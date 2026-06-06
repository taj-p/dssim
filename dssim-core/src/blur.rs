// 1D kernel from separable decomposition of the original 3×3 Gaussian
// (KERNEL = [0.095332, 0.118095, 0.095332, …, 0.146293, …]).
// Symmetric 1D form: K1D = [K_SIDE, K_CENTER, K_SIDE].
const K_SIDE: f32 = 0.308_758_86;
const K_CENTER: f32 = 0.382_482_8;

// Fused double-blur 5-tap kernel: convolving K1D with itself.
// K5 = [K5_OUTER, K5_INNER, K5_MID, K5_INNER, K5_OUTER]
// This makes H→V→H→V (two 3-tap blurs) equivalent to a single H5→V5 pass,
// halving memory traffic.
const K5_OUTER: f32 = K_SIDE * K_SIDE;
const K5_INNER: f32 = 2.0 * K_SIDE * K_CENTER;
const K5_MID: f32 = 2.0 * K_SIDE * K_SIDE + K_CENTER * K_CENTER;

// Edge-pixel coefficients chosen to make this pass *bit-equivalent* to two
// successive 1D 3-tap clamped passes (the upstream double-3×3 form). Derived
// by composing two H1·H1 clamped operations at j=0:
//
//   pass1 at 0: (K_SIDE+K_CENTER)·p[0] + K_SIDE·p[1]
//   pass1 at 1: K_SIDE·p[0] + K_CENTER·p[1] + K_SIDE·p[2]
//   pass2 at 0 = (K_SIDE+K_CENTER)·pass1[0] + K_SIDE·pass1[1]
//              = (K_M + K_I)·p[0] + (K_O + K_I)·p[1] + K_O·p[2]
//
// The single 5-tap with replicated clamps would over-weight p[0] by K_O. The
// inner pixels (j ∈ {1, w-2}) still match the upstream double-3×3 with the
// plain 5-tap form.
const K5_EDGE_CENTER: f32 = K5_MID + K5_INNER;
const K5_EDGE_NEAR: f32 = K5_OUTER + K5_INNER;
const K5_EDGE_FAR: f32 = K5_OUTER;

mod portable {
    use super::{K5_EDGE_CENTER, K5_EDGE_FAR, K5_EDGE_NEAR, K5_INNER, K5_MID, K5_OUTER};
    use imgref::*;
    use std::mem::MaybeUninit;

    /// True when the CPU supports the AVX2+FMA hot path. `is_x86_feature_detected!`
    /// caches its result internally, so after the first call this is just an
    /// atomic load + branch.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn has_avx2_fma() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    /// Plain 5-tap elementwise combine over six equal-length slices:
    /// `out[i] = (m2+p2)·OUTER + (m1+p1)·INNER + c·MID`.
    /// Shared interior body of both `blur_h5` (offset sub-slices) and
    /// `blur_v5` (full-width rows). Re-slicing to `out.len()` lets LLVM hoist
    /// the bounds checks and vectorize the loop.
    #[inline(always)]
    fn blur5_inner(
        m2: &[f32], m1: &[f32], c: &[f32], p1: &[f32], p2: &[f32],
        out: &mut [MaybeUninit<f32>],
    ) {
        let n = out.len();
        let (m2, m1, c, p1, p2) = (&m2[..n], &m1[..n], &c[..n], &p1[..n], &p2[..n]);
        for i in 0..n {
            out[i].write((m2[i] + p2[i]) * K5_OUTER + (m1[i] + p1[i]) * K5_INNER + c[i] * K5_MID);
        }
    }

    /// AVX2/FMA build of `blur5_inner`. Because `blur5_inner` is
    /// `#[inline(always)]`, LLVM re-codegens it here with 256-bit (8×f32)
    /// vectors instead of the baseline SSE2 (4×f32). The arithmetic is
    /// elementwise (no reduction), so results are bit-identical to the
    /// portable path; we do not enable fast-math.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn blur5_inner_avx2(
        m2: &[f32], m1: &[f32], c: &[f32], p1: &[f32], p2: &[f32],
        out: &mut [MaybeUninit<f32>],
    ) {
        blur5_inner(m2, m1, c, p1, p2, out);
    }

    /// Runtime-dispatched `blur5_inner`.
    #[inline]
    fn blur5(
        m2: &[f32], m1: &[f32], c: &[f32], p1: &[f32], p2: &[f32],
        out: &mut [MaybeUninit<f32>],
    ) {
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: only reached after confirming AVX2+FMA are present.
            unsafe { blur5_inner_avx2(m2, m1, c, p1, p2, out) };
            return;
        }
        blur5_inner(m2, m1, c, p1, p2, out);
    }

    /// 5-tap elementwise combine fused with a per-pixel product
    /// (`q[i] = a[i]·b[i]`), then the same 5-tap form over `q`. Shared interior
    /// body of `blur_h5_mul`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn blur5_inner_mul(
        a_m2: &[f32], a_m1: &[f32], a_c: &[f32], a_p1: &[f32], a_p2: &[f32],
        b_m2: &[f32], b_m1: &[f32], b_c: &[f32], b_p1: &[f32], b_p2: &[f32],
        out: &mut [MaybeUninit<f32>],
    ) {
        let n = out.len();
        let (a_m2, a_m1, a_c, a_p1, a_p2) = (&a_m2[..n], &a_m1[..n], &a_c[..n], &a_p1[..n], &a_p2[..n]);
        let (b_m2, b_m1, b_c, b_p1, b_p2) = (&b_m2[..n], &b_m1[..n], &b_c[..n], &b_p1[..n], &b_p2[..n]);
        for i in 0..n {
            let pm2 = a_m2[i] * b_m2[i];
            let pm1 = a_m1[i] * b_m1[i];
            let pc = a_c[i] * b_c[i];
            let pp1 = a_p1[i] * b_p1[i];
            let pp2 = a_p2[i] * b_p2[i];
            out[i].write((pm2 + pp2) * K5_OUTER + (pm1 + pp1) * K5_INNER + pc * K5_MID);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn blur5_inner_mul_avx2(
        a_m2: &[f32], a_m1: &[f32], a_c: &[f32], a_p1: &[f32], a_p2: &[f32],
        b_m2: &[f32], b_m1: &[f32], b_c: &[f32], b_p1: &[f32], b_p2: &[f32],
        out: &mut [MaybeUninit<f32>],
    ) {
        blur5_inner_mul(a_m2, a_m1, a_c, a_p1, a_p2, b_m2, b_m1, b_c, b_p1, b_p2, out);
    }

    /// Runtime-dispatched `blur5_inner_mul`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn blur5_mul(
        a_m2: &[f32], a_m1: &[f32], a_c: &[f32], a_p1: &[f32], a_p2: &[f32],
        b_m2: &[f32], b_m1: &[f32], b_c: &[f32], b_p1: &[f32], b_p2: &[f32],
        out: &mut [MaybeUninit<f32>],
    ) {
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: only reached after confirming AVX2+FMA are present.
            unsafe {
                blur5_inner_mul_avx2(a_m2, a_m1, a_c, a_p1, a_p2, b_m2, b_m1, b_c, b_p1, b_p2, out)
            };
            return;
        }
        blur5_inner_mul(a_m2, a_m1, a_c, a_p1, a_p2, b_m2, b_m1, b_c, b_p1, b_p2, out);
    }

    /// Horizontal 5-tap blur, bit-equivalent to two sequential clamped 1D
    /// 3-tap blurs. Edges (`j=0` and `j=w-1`) use the legacy-equivalent
    /// 3-coefficient form derived from H1·H1 clamping; `j=1` and `j=w-2`
    /// already match H1·H1 with the plain clamped 5-tap.
    fn blur_h5(src: &[f32], dst: &mut [MaybeUninit<f32>], width: usize, height: usize, src_stride: usize) {
        debug_assert!(width >= 1);
        let last = width - 1;
        for y in 0..height {
            let row = &src[y * src_stride..][..width];
            let out = &mut dst[y * width..][..width];

            // Edge: j=0. Reads p[0], p[min(1, last)], p[min(2, last)] with
            // the H1·H1-derived weights. Works for any width ≥ 1.
            let p0 = row[0];
            let p1 = row[1.min(last)];
            let p2 = row[2.min(last)];
            out[0].write(K5_EDGE_CENTER * p0 + K5_EDGE_NEAR * p1 + K5_EDGE_FAR * p2);

            // Edge: j=w-1 (mirror of j=0). Skipped when width==1 because
            // j=0 already covered it.
            if width >= 2 {
                let pl = row[last];
                let pl1 = row[last - 1];
                let pl2 = row[last.saturating_sub(2)];
                out[last].write(K5_EDGE_FAR * pl2 + K5_EDGE_NEAR * pl1 + K5_EDGE_CENTER * pl);
            }

            // Near-edge: j=1 — plain clamped 5-tap (matches H1·H1).
            if width >= 3 {
                // i-2 → 0, i-1 → 0, i = 1, i+1 = 2, i+2 = min(3, last)
                let m2 = row[0];
                let m1 = row[0];
                let c = row[1];
                let p1n = row[2.min(last)];
                let p2n = row[3.min(last)];
                out[1].write((m2 + p2n) * K5_OUTER + (m1 + p1n) * K5_INNER + c * K5_MID);
            }

            // Near-edge: j=w-2 — plain clamped 5-tap (matches H1·H1).
            if width >= 4 {
                let i = last - 1;
                let m2 = row[i - 2];
                let m1 = row[i - 1];
                let c = row[i];
                let p1n = row[i + 1];           // = last
                let p2n = row[(i + 2).min(last)]; // i+2 == last+1 → clamp to last
                out[i].write((m2 + p2n) * K5_OUTER + (m1 + p1n) * K5_INNER + c * K5_MID);
            }

            // Interior: j ∈ [2, w-2). Five aligned sub-slices so LLVM hoists
            // bounds checks once per row and emits AVX2/NEON SIMD over the body.
            if width >= 5 {
                let inner_len = width - 4;
                let r_m2 = &row[..inner_len];
                let r_m1 = &row[1..1 + inner_len];
                let r_c  = &row[2..2 + inner_len];
                let r_p1 = &row[3..3 + inner_len];
                let r_p2 = &row[4..4 + inner_len];
                let (_, out_rest) = out.split_at_mut(2);
                let out_inner = &mut out_rest[..inner_len];
                blur5(r_m2, r_m1, r_c, r_p1, r_p2, out_inner);
            }
        }
    }

    /// Vertical 5-tap blur, bit-equivalent to two sequential clamped 1D
    /// 3-tap blurs. Same edge-handling structure as `blur_h5`. `src` must
    /// be tightly packed (stride == width).
    fn blur_v5(src: &[f32], dst: &mut [MaybeUninit<f32>], width: usize, height: usize, dst_stride: usize) {
        debug_assert!(height >= 1);
        let last_y = height - 1;

        // Helper: row slice at index y (clamped within [0, last_y]).
        let row = |y: usize| &src[y * width..][..width];

        // Edge: y=0 — H1·H1-derived 3-coefficient form (vertical).
        {
            let r0 = row(0);
            let r1 = row(1.min(last_y));
            let r2 = row(2.min(last_y));
            let out = &mut dst[..width];
            for x in 0..width {
                out[x].write(
                    K5_EDGE_CENTER * r0[x] + K5_EDGE_NEAR * r1[x] + K5_EDGE_FAR * r2[x],
                );
            }
        }

        // Edge: y=h-1 (mirror of y=0).
        if height >= 2 {
            let rl = row(last_y);
            let rl1 = row(last_y - 1);
            let rl2 = row(last_y.saturating_sub(2));
            let out = &mut dst[last_y * dst_stride..][..width];
            for x in 0..width {
                out[x].write(
                    K5_EDGE_FAR * rl2[x] + K5_EDGE_NEAR * rl1[x] + K5_EDGE_CENTER * rl[x],
                );
            }
        }

        // Near-edge: y=1 — plain clamped 5-tap.
        if height >= 3 {
            let rm = row(0);
            let rc = row(1);
            let rp1 = row(2.min(last_y));
            let rp2 = row(3.min(last_y));
            let out = &mut dst[dst_stride..][..width];
            for x in 0..width {
                // m2 and m1 both clamp to row 0.
                out[x].write(
                    (rm[x] + rp2[x]) * K5_OUTER
                    + (rm[x] + rp1[x]) * K5_INNER
                    + rc[x] * K5_MID,
                );
            }
        }

        // Near-edge: y=h-2 — plain clamped 5-tap.
        if height >= 4 {
            let y = last_y - 1;
            let rm2 = row(y - 2);
            let rm1 = row(y - 1);
            let rc = row(y);
            let rp1 = row(y + 1);                // = last_y
            let rp2 = row((y + 2).min(last_y));  // y+2 == last_y+1 → clamp to last_y
            let out = &mut dst[y * dst_stride..][..width];
            for x in 0..width {
                out[x].write(
                    (rm2[x] + rp2[x]) * K5_OUTER
                    + (rm1[x] + rp1[x]) * K5_INNER
                    + rc[x] * K5_MID,
                );
            }
        }

        // Interior: y ∈ [2, h-2). Plain 5-tap; this is the SIMD-friendly hot loop.
        if height >= 5 {
            for y in 2..height - 2 {
                let rm2 = row(y - 2);
                let rm1 = row(y - 1);
                let rc  = row(y);
                let rp1 = row(y + 1);
                let rp2 = row(y + 2);
                let out = &mut dst[y * dst_stride..][..width];
                blur5(rm2, rm1, rc, rp1, rp2, out);
            }
        }
    }

    /// Horizontal 5-tap blur with fused element-wise multiply, bit-equivalent
    /// to clamped H1·H1 applied to `src1 * src2`. Same edge-handling structure
    /// as `blur_h5`.
    #[allow(clippy::too_many_arguments)]
    fn blur_h5_mul(
        src1: &[f32],
        src2: &[f32],
        dst: &mut [MaybeUninit<f32>],
        width: usize,
        height: usize,
        stride1: usize,
        stride2: usize,
    ) {
        debug_assert!(width >= 1);
        let last = width - 1;
        for y in 0..height {
            let r1 = &src1[y * stride1..][..width];
            let r2 = &src2[y * stride2..][..width];
            let out = &mut dst[y * width..][..width];
            let prod = |i: usize| r1[i] * r2[i];

            // Edge: j=0. H1·H1-derived 3-coefficient form on q[i] = r1[i]·r2[i].
            let q0 = prod(0);
            let q1 = prod(1.min(last));
            let q2 = prod(2.min(last));
            out[0].write(K5_EDGE_CENTER * q0 + K5_EDGE_NEAR * q1 + K5_EDGE_FAR * q2);

            // Edge: j=w-1.
            if width >= 2 {
                let ql = prod(last);
                let ql1 = prod(last - 1);
                let ql2 = prod(last.saturating_sub(2));
                out[last].write(K5_EDGE_FAR * ql2 + K5_EDGE_NEAR * ql1 + K5_EDGE_CENTER * ql);
            }

            // Near-edge: j=1.
            if width >= 3 {
                let m2 = prod(0);
                let m1 = prod(0);
                let c = prod(1);
                let p1n = prod(2.min(last));
                let p2n = prod(3.min(last));
                out[1].write((m2 + p2n) * K5_OUTER + (m1 + p1n) * K5_INNER + c * K5_MID);
            }

            // Near-edge: j=w-2.
            if width >= 4 {
                let i = last - 1;
                let m2 = prod(i - 2);
                let m1 = prod(i - 1);
                let c = prod(i);
                let p1n = prod(i + 1);
                let p2n = prod((i + 2).min(last));
                out[i].write((m2 + p2n) * K5_OUTER + (m1 + p1n) * K5_INNER + c * K5_MID);
            }

            // Interior: j ∈ [2, w-2). Build five pairs of aligned sub-slices.
            if width >= 5 {
                let inner_len = width - 4;
                let s1_m2 = &r1[..inner_len];
                let s1_m1 = &r1[1..1 + inner_len];
                let s1_c  = &r1[2..2 + inner_len];
                let s1_p1 = &r1[3..3 + inner_len];
                let s1_p2 = &r1[4..4 + inner_len];
                let s2_m2 = &r2[..inner_len];
                let s2_m1 = &r2[1..1 + inner_len];
                let s2_c  = &r2[2..2 + inner_len];
                let s2_p1 = &r2[3..3 + inner_len];
                let s2_p2 = &r2[4..4 + inner_len];
                let (_, out_rest) = out.split_at_mut(2);
                let out_inner = &mut out_rest[..inner_len];
                blur5_mul(
                    s1_m2, s1_m1, s1_c, s1_p1, s1_p2,
                    s2_m2, s2_m1, s2_c, s2_p1, s2_p2,
                    out_inner,
                );
            }
        }
    }

    /// Promote `&mut [MaybeUninit<f32>]` to `&[f32]` once every cell is written.
    /// SAFETY: every cell of `slice` must have been initialized.
    unsafe fn assume_init_ref(slice: &[MaybeUninit<f32>]) -> &[f32] {
        // SAFETY: f32 and MaybeUninit<f32> have identical layout; caller guarantees init.
        unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<f32>(), slice.len()) }
    }

    pub fn blur(src: ImgRef<'_, f32>, tmp: &mut [MaybeUninit<f32>]) -> ImgVec<f32> {
        let width = src.width();
        let height = src.height();
        assert!(width > 0 && width < 1 << 24);
        assert!(height > 0 && height < 1 << 24);
        debug_assert!(src.pixels().all(|p| p.is_finite()));

        let pixels = width * height;
        assert!(tmp.len() >= pixels);
        let tmp = &mut tmp[..pixels];

        let mut dst_vec: Vec<f32> = Vec::with_capacity(pixels);
        let dst_uninit: &mut [MaybeUninit<f32>] = &mut dst_vec.spare_capacity_mut()[..pixels];

        blur_h5(src.buf(), tmp, width, height, src.stride());
        // SAFETY: blur_h5 wrote every cell of tmp[..pixels].
        let tmp_init: &[f32] = unsafe { assume_init_ref(tmp) };
        blur_v5(tmp_init, dst_uninit, width, height, width);

        // SAFETY: blur_v5 wrote every cell of dst_vec.spare_capacity_mut().
        unsafe { dst_vec.set_len(pixels); }
        ImgVec::new(dst_vec, width, height)
    }

    pub fn blur_in_place(mut srcdst: ImgRefMut<'_, f32>, tmp: &mut [MaybeUninit<f32>]) {
        let width = srcdst.width();
        let height = srcdst.height();
        let stride = srcdst.stride();
        assert!(width > 0 && width < 1 << 24);
        assert!(height > 0 && height < 1 << 24);

        let pixels = width * height;
        assert!(tmp.len() >= pixels);
        let tmp = &mut tmp[..pixels];

        blur_h5(srcdst.buf(), tmp, width, height, stride);
        // SAFETY: blur_h5 wrote every cell of tmp[..pixels].
        let tmp_init: &[f32] = unsafe { assume_init_ref(tmp) };

        // Reinterpret the (initialized) destination buffer as MaybeUninit so blur_v5
        // can reuse its `&mut [MaybeUninit<f32>]` write path. Every pixel inside the
        // (width,height) window will be overwritten before any further read.
        let dst_buf = srcdst.buf_mut();
        // SAFETY: f32 and MaybeUninit<f32> have the same layout; we overwrite every cell.
        let dst_uninit: &mut [MaybeUninit<f32>] = unsafe {
            std::slice::from_raw_parts_mut(
                dst_buf.as_mut_ptr().cast::<MaybeUninit<f32>>(),
                dst_buf.len(),
            )
        };

        blur_v5(tmp_init, dst_uninit, width, height, stride);
    }

    /// Blur the element-wise product of two images: `blur(src1 * src2)`.
    /// Fuses the multiply into the horizontal pass, then does a single vertical pass.
    pub fn blur_mul(src1: ImgRef<'_, f32>, src2: ImgRef<'_, f32>, tmp: &mut [MaybeUninit<f32>]) -> Vec<f32> {
        let width = src1.width();
        let height = src1.height();
        debug_assert_eq!(width, src2.width());
        debug_assert_eq!(height, src2.height());
        assert!(width > 0 && width < 1 << 24);
        assert!(height > 0 && height < 1 << 24);

        let pixels = width * height;
        assert!(tmp.len() >= pixels);
        let tmp = &mut tmp[..pixels];

        let mut dst_vec: Vec<f32> = Vec::with_capacity(pixels);
        let dst_uninit: &mut [MaybeUninit<f32>] = &mut dst_vec.spare_capacity_mut()[..pixels];

        blur_h5_mul(
            src1.buf(),
            src2.buf(),
            tmp,
            width,
            height,
            src1.stride(),
            src2.stride(),
        );
        // SAFETY: blur_h5_mul wrote every cell of tmp[..pixels].
        let tmp_init: &[f32] = unsafe { assume_init_ref(tmp) };
        blur_v5(tmp_init, dst_uninit, width, height, width);

        // SAFETY: blur_v5 wrote every cell.
        unsafe { dst_vec.set_len(pixels); }
        dst_vec
    }
}

pub use self::portable::*;

#[cfg(test)]
use imgref::*;

#[test]
fn blur_zero() {
    use std::mem::MaybeUninit;
    let src = vec![0.25];
    let mut src2 = src.clone();

    let mut tmp = vec![MaybeUninit::uninit(); 1];
    let dst = blur(ImgRef::new(&src[..], 1, 1), &mut tmp[..]);
    blur_in_place(ImgRefMut::new(&mut src2[..], 1, 1), &mut tmp[..]);

    assert_eq!(&src2, dst.buf());
    assert!((0.25 - dst.buf()[0]).abs() < 0.00001);
}

#[test]
fn blur_one() {
    blur_one_compare(Img::new(vec![
        0.,0.,0.,0.,0.,
        0.,0.,0.,0.,0.,
        0.,0.,1.,0.,0.,
        0.,0.,0.,0.,0.,
        0.,0.,0.,0.,0.,
    ], 5, 5));
}

#[test]
fn blur_one_stride() {
    let nan = 1./0.;
    blur_one_compare(Img::new_stride(vec![
        0.,0.,0.,0.,0., nan, -11.,
        0.,0.,0.,0.,0., 333., nan,
        0.,0.,1.,0.,0., nan, -11.,
        0.,0.,0.,0.,0., 333., nan,
        0.,0.,0.,0.,0., nan,
    ], 5, 5, 7));
}

#[cfg(test)]
fn blur_one_compare(src: ImgVec<f32>) {
    use std::mem::MaybeUninit;
    let mut src2 = src.clone();

    let mut tmp = vec![MaybeUninit::uninit(); 5 * 5];
    let dst = blur(src.as_ref(), &mut tmp[..]);
    blur_in_place(src2.as_mut(), &mut tmp[..]);

    assert_eq!(&src2.pixels().collect::<Vec<_>>(), dst.buf());

    assert!((1. / 110. - dst.buf()[0]).abs() < 0.0001, "{dst:?}");
    assert!((1. / 110. - dst.buf()[5 * 5 - 1]).abs() < 0.0001, "{dst:?}");
    assert!((0.11354011 - dst.buf()[2 * 5 + 2]).abs() < 0.0001);
}

#[test]
fn blur_1x1() {
    use std::mem::MaybeUninit;
    let src = vec![1.];
    let mut src2 = src.clone();

    let mut tmp = vec![MaybeUninit::uninit(); 1];
    let dst = blur(ImgRef::new(&src[..], 1, 1), &mut tmp[..]);
    blur_in_place(ImgRefMut::new(&mut src2[..], 1, 1), &mut tmp[..]);

    assert!((dst.buf()[0] - 1.).abs() < 0.00001);
    assert!((src2[0] - 1.).abs() < 0.00001);
}

#[test]
fn blur_two() {
    use std::mem::MaybeUninit;
    // 4×4 image with a 0 at corner (0,0) and 1s elsewhere. The blur should:
    //   - keep the far corners at 1 (kernel sums to 1, all neighbors are 1);
    //   - pull corner (0,0) up toward 1 by exactly the amount the legacy
    //     double-3×3 blur would (this branch's blur is bit-equivalent to it).
    let src = vec![
        0., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
    ];
    let mut src2 = src.clone();

    let mut tmp = vec![MaybeUninit::uninit(); 4 * 4];
    let dst = blur(ImgRef::new(&src[..], 4, 4), &mut tmp[..]);
    blur_in_place(ImgRefMut::new(&mut src2[..], 4, 4), &mut tmp[..]);

    assert_eq!(&src2, dst.buf());

    // All-1 corners should remain 1.0 (kernel is normalized).
    assert!((1. - dst.buf()[3]).abs() < 0.0001, "{}", dst.buf()[3]);
    assert!((1. - dst.buf()[3 * 4]).abs() < 0.0001, "{}", dst.buf()[3 * 4]);
    assert!((1. - dst.buf()[4 * 4 - 1]).abs() < 0.0001, "{}", dst.buf()[4 * 4 - 1]);

    // Locked corner-(0,0) value: this is exactly what two clamped 1D 3-tap
    // passes (i.e. the upstream double-3×3 blur) produce on this fixture, to
    // four decimal places. The new fused 5-tap with the H1·H1-derived edge
    // weights matches it bit-for-bit modulo FP reordering. The
    // `blur_equiv` tests further down validate the equivalence over a
    // wider battery of inputs.
    let expected = 0.671_504_5_f32;
    assert!(
        (f64::from(expected) - f64::from(dst.buf()[0])).abs() < 0.0001,
        "expected {expected}, got {}",
        dst.buf()[0]
    );
}

// Equivalence test against the upstream double-3×3 blur lives in a separate
// file to keep this module focused on the algorithm. See
// `blur/equiv_tests.rs` for the legacy reference impl and per-pixel parity
// sweep.
#[cfg(test)]
mod equiv_tests;

// Microbenchmarks for the blur hot paths. Kept inside the crate because the
// `blur` module is private. Run with:
//   RUSTC_BOOTSTRAP=1 cargo bench -p dssim-core
// (matches the repo's existing nightly-`test` bench convention.)
#[cfg(test)]
mod blur_bench {
    extern crate test;
    use super::{blur, blur_in_place, blur_mul};
    use imgref::*;
    use std::mem::MaybeUninit;
    use test::Bencher;

    // Deterministic pseudo-random fill so the benched data is representative
    // (autovectorizable but not degenerate) without a `rand` dependency.
    fn make_image(w: usize, h: usize, seed: u32) -> ImgVec<f32> {
        let mut s = seed | 1;
        let buf: Vec<f32> = (0..w * h)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect();
        ImgVec::new(buf, w, h)
    }

    fn bench_blur(b: &mut Bencher, w: usize, h: usize) {
        let src = make_image(w, h, 0x1234_5678);
        let mut tmp: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); w * h];
        b.iter(|| test::black_box(blur(test::black_box(src.as_ref()), &mut tmp)));
    }

    fn bench_blur_in_place(b: &mut Bencher, w: usize, h: usize) {
        // Blur the same buffer in place repeatedly: this measures pure blur
        // cost (no per-iteration clone). Re-blurring a blurred image is
        // same-shape, finite work, so the timing is representative.
        let mut img = make_image(w, h, 0x1234_5678);
        let mut tmp: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); w * h];
        b.iter(|| {
            blur_in_place(test::black_box(img.as_mut()), &mut tmp);
            test::black_box(&img);
        });
    }

    fn bench_blur_mul(b: &mut Bencher, w: usize, h: usize) {
        let a = make_image(w, h, 0x1234_5678);
        let c = make_image(w, h, 0x9E37_79B9);
        let mut tmp: Vec<MaybeUninit<f32>> = vec![MaybeUninit::uninit(); w * h];
        b.iter(|| {
            test::black_box(blur_mul(
                test::black_box(a.as_ref()),
                test::black_box(c.as_ref()),
                &mut tmp,
            ))
        });
    }

    #[bench] fn blur_320x200(b: &mut Bencher) { bench_blur(b, 320, 200); }
    #[bench] fn blur_1024x768(b: &mut Bencher) { bench_blur(b, 1024, 768); }
    #[bench] fn blur_1920x1080(b: &mut Bencher) { bench_blur(b, 1920, 1080); }

    #[bench] fn blur_in_place_320x200(b: &mut Bencher) { bench_blur_in_place(b, 320, 200); }
    #[bench] fn blur_in_place_1024x768(b: &mut Bencher) { bench_blur_in_place(b, 1024, 768); }
    #[bench] fn blur_in_place_1920x1080(b: &mut Bencher) { bench_blur_in_place(b, 1920, 1080); }

    #[bench] fn blur_mul_320x200(b: &mut Bencher) { bench_blur_mul(b, 320, 200); }
    #[bench] fn blur_mul_1024x768(b: &mut Bencher) { bench_blur_mul(b, 1024, 768); }
    #[bench] fn blur_mul_1920x1080(b: &mut Bencher) { bench_blur_mul(b, 1920, 1080); }
}
