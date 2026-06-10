//! The library interface is awfully abstract, because it strives to efficiently, and very accurately,
//! support several pixel types. It also allows replacing some parts of the algorithm with different implementations
//! (if you need higher accuracy or higher speed).
#![doc(html_logo_url = "https://kornel.ski/dssim/logo.png")]
#![cfg_attr(test, feature(test))]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::new_without_default)]

// Measurement-only: use mimalloc as the global allocator under tests/benches
// when the `bench-mimalloc` feature is on, to bound allocation overhead.
#[cfg(all(test, feature = "bench-mimalloc"))]
#[global_allocator]
static GLOBAL_MIMALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod blur;
mod c_api;
mod dssim;
/// cbindgen:ignore
mod ffi;
mod image;
#[cfg(not(feature = "threads"))]
mod lieon;
mod linear;
mod pool;
mod tolab;
mod val;

pub use crate::dssim::*;
pub use crate::image::*;
pub use crate::linear::*;
pub use crate::pool::DssimPool;

/// Re-export of the `imgref` crate, whose `ImgVec`/`ImgRef` types are the input
/// bitmaps accepted by [`Dssim::create_image`] and the pooled
/// [`Dssim::compare_pair_in`]. Re-exported so callers can construct inputs
/// without taking their own (version-matched) `imgref` dependency.
pub use imgref;
