//! Allocator-agnostic buffer reuse for DSSIM.
//!
//! DSSIM allocates many large, short-lived image-sized `Vec<f32>` buffers per
//! comparison (LAB planes, blurred means, squared-image blurs, scratch). For a
//! workload that compares many independent image pairs, those buffers are
//! allocated and freed on every pair; with the system allocator each large
//! block is `mmap`/`munmap`'d, re-faulting pages every time.
//!
//! A [`DssimPool`] recycles those buffers across calls. It is fully
//! allocator-agnostic: it neither depends on nor installs any
//! `#[global_allocator]`, so it composes with whatever allocator the final
//! binary chooses (or doesn't).

use std::collections::BTreeMap;
use std::sync::Mutex;

const ELEM: usize = std::mem::size_of::<f32>();

/// A thread-safe pool of reusable `f32` buffers.
///
/// Hold one `DssimPool` and pass it to [`Dssim::create_image_in`],
/// [`Dssim::compare_in`], or [`Dssim::compare_pair_in`], reusing it across many
/// comparisons. Buffers taken for a comparison are returned to the pool (via
/// [`DssimPool::reclaim`] or automatically by `compare_pair_in`) and reused by
/// the next one, avoiding repeated large allocations.
///
/// [`Dssim::create_image_in`]: crate::Dssim::create_image_in
/// [`Dssim::compare_in`]: crate::Dssim::compare_in
/// [`Dssim::compare_pair_in`]: crate::Dssim::compare_pair_in
pub struct DssimPool {
    inner: Mutex<Inner>,
    /// Maximum number of idle bytes retained in the free list; gives beyond
    /// this are dropped (returned to the allocator) to bound memory use.
    cap_bytes: usize,
}

struct Inner {
    /// Free buffers keyed by their capacity in elements, so a best-fit buffer
    /// can be found with a single `BTreeMap` range query.
    free: BTreeMap<usize, Vec<Vec<f32>>>,
    held_bytes: usize,
}

impl Default for DssimPool {
    fn default() -> Self {
        Self::new()
    }
}

impl DssimPool {
    /// Create a pool with a default idle-memory cap (1 GiB).
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity_bytes(1 << 30)
    }

    /// Create a pool that retains at most `cap_bytes` of idle (freed) buffers.
    /// Buffers returned beyond this are released to the allocator.
    #[must_use]
    pub fn with_capacity_bytes(cap_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner { free: BTreeMap::new(), held_bytes: 0 }),
            cap_bytes,
        }
    }

    /// Take a cleared buffer with `capacity() >= len`, reusing a freed one when
    /// available (best fit), otherwise allocating a fresh `Vec`.
    pub(crate) fn take(&self, len: usize) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
        }
        let mut inner = self.inner.lock().unwrap();
        // Smallest capacity bucket that still satisfies `len`.
        if let Some((&cap, _)) = inner.free.range(len..).next() {
            let bucket = inner.free.get_mut(&cap).expect("range key exists");
            let mut buf = bucket.pop().expect("non-empty bucket");
            if bucket.is_empty() {
                inner.free.remove(&cap);
            }
            inner.held_bytes -= cap * ELEM;
            buf.clear();
            debug_assert!(buf.capacity() >= len);
            return buf;
        }
        Vec::with_capacity(len)
    }

    /// Return a buffer to the pool for reuse. The contents are discarded; the
    /// allocation (capacity) is kept unless the idle cap would be exceeded.
    pub(crate) fn give(&self, mut buf: Vec<f32>) {
        let cap = buf.capacity();
        if cap == 0 {
            return;
        }
        buf.clear();
        let bytes = cap * ELEM;
        let mut inner = self.inner.lock().unwrap();
        if inner.held_bytes + bytes > self.cap_bytes {
            return; // let `buf` drop and free
        }
        inner.held_bytes += bytes;
        inner.free.entry(cap).or_default().push(buf);
    }

    /// Number of idle bytes currently retained (for tests/diagnostics).
    #[cfg(test)]
    fn held_bytes(&self) -> usize {
        self.inner.lock().unwrap().held_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_from_empty_allocates() {
        let pool = DssimPool::new();
        let buf = pool.take(100);
        assert!(buf.capacity() >= 100);
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn give_then_take_reuses_same_allocation() {
        let pool = DssimPool::new();
        let mut buf = Vec::with_capacity(1000);
        buf.push(1.0);
        let cap = buf.capacity();
        let ptr = buf.as_ptr();
        pool.give(buf);
        assert_eq!(pool.held_bytes(), cap * ELEM);

        let reused = pool.take(1000);
        assert_eq!(reused.as_ptr(), ptr, "should reuse the same allocation");
        assert_eq!(reused.len(), 0, "buffer is cleared");
        assert!(reused.capacity() >= 1000);
        assert_eq!(pool.held_bytes(), 0, "removed from free list on take");
    }

    #[test]
    fn take_is_best_fit() {
        let pool = DssimPool::new();
        pool.give(Vec::with_capacity(1000));
        pool.give(Vec::with_capacity(100));
        // Request 50: should get the 100-capacity buffer, not the 1000 one.
        let small = pool.take(50);
        assert!(small.capacity() >= 100 && small.capacity() < 1000);
        // The 1000 buffer is still available.
        let big = pool.take(600);
        assert!(big.capacity() >= 1000);
    }

    #[test]
    fn take_skips_too_small() {
        let pool = DssimPool::new();
        pool.give(Vec::with_capacity(10));
        // Need 100; the 10-capacity buffer cannot satisfy it -> fresh alloc.
        let buf = pool.take(100);
        assert!(buf.capacity() >= 100);
        // The 10-capacity buffer is untouched.
        assert_eq!(pool.held_bytes(), 10 * ELEM);
    }

    #[test]
    fn cap_bytes_limits_retention() {
        // Cap at 100 elements worth of bytes.
        let pool = DssimPool::with_capacity_bytes(100 * ELEM);
        pool.give(Vec::with_capacity(80));
        assert_eq!(pool.held_bytes(), 80 * ELEM);
        // Adding 80 more would exceed 100 -> dropped.
        pool.give(Vec::with_capacity(80));
        assert_eq!(pool.held_bytes(), 80 * ELEM);
    }

    #[test]
    fn zero_len_take_and_empty_give_are_noops() {
        let pool = DssimPool::new();
        assert_eq!(pool.take(0).capacity(), 0);
        pool.give(Vec::new());
        assert_eq!(pool.held_bytes(), 0);
    }
}
