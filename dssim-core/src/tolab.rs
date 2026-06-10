#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use crate::image::ToRGB;
use crate::image::RGBAPLU;
use crate::image::RGBLU;
use crate::pool::DssimPool;
use imgref::*;
use std::mem::MaybeUninit;
#[cfg(not(feature = "threads"))]
use crate::lieon as rayon;
use rayon::prelude::*;

const D65x: f32 = 0.9505;
const D65y: f32 = 1.0;
const D65z: f32 = 1.089;

pub type GBitmap = ImgVec<f32>;

// 1.05 (vs the usual 1.16) on L boosts color importance without pushing values
// outside 0..1; the 86.2/220 and 107.9/220 offsets keep a*/b* positive. The
// per-pixel RGB->LAB math now lives in the vectorized `lab_transform` below.
#[inline(always)]
fn fma_matrix(r: f32, rx: f32, g: f32, gx: f32, b: f32, bx: f32) -> f32 {
    b.mul_add(bx, g.mul_add(gx, r * rx))
}

const EPSILON: f32 = 216. / 24389.;
const K: f32 = 24389. / (27. * 116.); // http://www.brucelindbloom.com/LContinuity.html

#[inline]
fn cbrt_poly(x: f32) -> f32 {
    // Polynomial approximation
    let poly = [0.2f32, 1.51, -0.5];
    let y = poly[2].mul_add(x, poly[1]).mul_add(x, poly[0]);

    // 2x Halley's Method
    let y3 = y * y * y;
    let y = y * 2.0f32.mul_add(x, y3) / 2.0f32.mul_add(y3, x);
    let y3 = y * y * y;
    let y = y * 2.0f32.mul_add(x, y3) / 2.0f32.mul_add(y3, x);
    debug_assert!(y < 1.001);
    debug_assert!(x < 216. / 24389. || y >= 16. / 116.);
    y
}

/// Branchless form of the L*a*b* `f(t)` companding used per channel:
/// `if t > EPSILON { cbrt_poly(t) - 16/116 } else { K*t }`. Both arms are
/// computed and blended so the loop vectorizes; for `t <= EPSILON` the
/// discarded `cbrt_poly(t)` stays finite for all inputs in `[0, 1]`. The
/// selected value is bit-identical to the original branched scalar code.
#[inline(always)]
fn lab_f(t: f32) -> f32 {
    let cbrt = cbrt_poly(t) - 16. / 116.;
    let lin = K * t;
    if t > EPSILON { cbrt } else { lin }
}

/// In-place RGB(linear) -> L*a*b* transform over three equal-length planar
/// slices (`r`/`g`/`b` hold the input on entry and the output L/a/b on exit).
/// Each element is independent, so this both vectorizes and runs in place.
/// Bit-identical to `RGBLU::to_lab` (same op order, fused `mul_add`s).
#[inline(always)]
fn lab_transform_scalar(r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let n = r.len();
    let (r, g, b) = (&mut r[..n], &mut g[..n], &mut b[..n]);
    for i in 0..n {
        let (rr, gg, bb) = (r[i], g[i], b[i]);
        let fx = fma_matrix(rr, 0.4124 / D65x, gg, 0.3576 / D65x, bb, 0.1805 / D65x);
        let fy = fma_matrix(rr, 0.2126 / D65y, gg, 0.7152 / D65y, bb, 0.0722 / D65y);
        let fz = fma_matrix(rr, 0.0193 / D65z, gg, 0.1192 / D65z, bb, 0.9505 / D65z);
        let X = lab_f(fx);
        let Y = lab_f(fy);
        let Z = lab_f(fz);
        r[i] = Y * 1.05f32;
        g[i] = (500.0 / 220.0f32).mul_add(X - Y, 86.2 / 220.0f32);
        b[i] = (200.0 / 220.0f32).mul_add(Y - Z, 107.9 / 220.0f32);
    }
}

/// AVX2/FMA build of `lab_transform_scalar`. The default x86-64 target only
/// has SSE2 and no FMA, so `mul_add` there lowers to a (slow) libm `fmaf`
/// call; under `avx2,fma` it becomes hardware `vfmadd` over 8 lanes. Results
/// are bit-identical (both `mul_add`s are correctly-rounded fused ops; no
/// fast-math).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn lab_transform_avx2(r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    lab_transform_scalar(r, g, b);
}

/// Runtime-dispatched `lab_transform_scalar`.
#[inline]
fn lab_transform(r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: only reached after confirming AVX2+FMA are present.
        unsafe { lab_transform_avx2(r, g, b) };
        return;
    }
    lab_transform_scalar(r, g, b);
}

/// Grayscale companding: `if fy > EPSILON { (cbrt_poly(fy)-16/116)*1.16 } else { (K*1.16)*fy }`,
/// branchless and in place. Bit-identical to the original `GBitmap::to_lab` closure.
#[inline(always)]
fn gray_lab_scalar(v: &mut [f32]) {
    for x in v.iter_mut() {
        let fy = *x;
        let cbrt = (cbrt_poly(fy) - 16. / 116.) * 1.16;
        let lin = (K * 1.16) * fy;
        *x = if fy > EPSILON { cbrt } else { lin };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gray_lab_avx2(v: &mut [f32]) {
    gray_lab_scalar(v);
}

/// Runtime-dispatched `gray_lab_scalar`.
#[inline]
fn gray_lab(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: only reached after confirming AVX2+FMA are present.
        unsafe { gray_lab_avx2(v) };
        return;
    }
    gray_lab_scalar(v);
}

/// Promote `&mut [MaybeUninit<f32>]` to `&mut [f32]` once every cell is written.
/// SAFETY: every cell of `s` must have been initialized.
#[inline(always)]
unsafe fn assume_init_mut(s: &mut [MaybeUninit<f32>]) -> &mut [f32] {
    // SAFETY: f32 and MaybeUninit<f32> share layout; caller guarantees init.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<f32>(), s.len()) }
}

/// Convert image to L\*a\*b\* planar
///
/// It should return 1 (gray) or 3 (color) planes.
pub trait ToLABBitmap {
    fn to_lab(&self) -> Vec<GBitmap>;

    /// Like [`ToLABBitmap::to_lab`], but draws its output planes from `pool`
    /// for reuse across calls. The default ignores the pool, so custom pixel
    /// types keep working unchanged; the built-in types override it.
    #[inline]
    fn to_lab_pooled(&self, _pool: &DssimPool) -> Vec<GBitmap> {
        self.to_lab()
    }
}

impl ToLABBitmap for ImgVec<RGBAPLU> {
    #[inline(always)]
    fn to_lab(&self) -> Vec<GBitmap> {
        self.as_ref().to_lab()
    }
    #[inline(always)]
    fn to_lab_pooled(&self, pool: &DssimPool) -> Vec<GBitmap> {
        self.as_ref().to_lab_pooled(pool)
    }
}

impl ToLABBitmap for ImgVec<RGBLU> {
    #[inline(always)]
    fn to_lab(&self) -> Vec<GBitmap> {
        self.as_ref().to_lab()
    }
    #[inline(always)]
    fn to_lab_pooled(&self, pool: &DssimPool) -> Vec<GBitmap> {
        self.as_ref().to_lab_pooled(pool)
    }
}
impl ToLABBitmap for GBitmap {
    #[inline(always)]
    fn to_lab(&self) -> Vec<GBitmap> {
        self.to_lab_pooled(&DssimPool::new())
    }
    fn to_lab_pooled(&self, pool: &DssimPool) -> Vec<GBitmap> {
        let width = self.width();
        let height = self.height();
        debug_assert!(width > 0);
        let area = width * height;

        let mut out: Vec<f32> = pool.take(area);
        out.spare_capacity_mut()
            .par_chunks_exact_mut(width)
            .take(height)
            .enumerate()
            .for_each(|(y, out_row)| {
                let in_row = &self[y][0..width];
                let out_row = &mut out_row[0..width];
                for x in 0..width {
                    out_row[x].write(in_row[x]);
                }
                // SAFETY: every cell of out_row was written above.
                gray_lab(unsafe { assume_init_mut(out_row) });
            });
        // SAFETY: every row (hence every cell) was written above.
        unsafe { out.set_len(area) };

        vec![Self::new(out, width, height)]
    }
}

/// `to_rgb` produces the (possibly dithered) linear RGB triplet for one input
/// pixel; the heavy RGB->LAB math is then applied per row by the vectorized,
/// AVX2-dispatched `lab_transform`. The cheap, branchy deinterleave/dither
/// stays scalar so the SIMD kernel is branch-free.
#[inline(never)]
fn rgb_to_lab<T: Copy + Sync + Send + 'static, F>(img: ImgRef<'_, T>, to_rgb: F, pool: &DssimPool) -> Vec<GBitmap>
    where F: Fn(T, usize) -> RGBLU + Sync + Send + 'static
{
    let width = img.width();
    assert!(width > 0);
    let height = img.height();
    let area = width * height;

    // Output planes come from the pool so they are reused across calls.
    let mut out_l = pool.take(area);
    let mut out_a = pool.take(area);
    let mut out_b = pool.take(area);

    // For output width == stride
    out_l.spare_capacity_mut().par_chunks_exact_mut(width).take(height).zip(
        out_a.spare_capacity_mut().par_chunks_exact_mut(width).take(height).zip(
            out_b.spare_capacity_mut().par_chunks_exact_mut(width).take(height))
    ).enumerate()
    .for_each(|(y, (l_row, (a_row, b_row)))| {
        let in_row = &img.rows().nth(y).unwrap()[0..width];
        let l_row = &mut l_row[0..width];
        let a_row = &mut a_row[0..width];
        let b_row = &mut b_row[0..width];
        // Phase 1 (scalar): deinterleave + dither into the output rows as r/g/b.
        for x in 0..width {
            let n = (x + 11) ^ (y + 11);
            let rgb = to_rgb(in_row[x], n);
            l_row[x].write(rgb.r);
            a_row[x].write(rgb.g);
            b_row[x].write(rgb.b);
        }
        // Phase 2 (SIMD): in-place RGB->LAB over the row.
        // SAFETY: phase 1 wrote every cell of each row.
        let r = unsafe { assume_init_mut(l_row) };
        let g = unsafe { assume_init_mut(a_row) };
        let b = unsafe { assume_init_mut(b_row) };
        lab_transform(r, g, b);
    });

    unsafe { out_l.set_len(area) };
    unsafe { out_a.set_len(area) };
    unsafe { out_b.set_len(area) };

    vec![
        Img::new(out_l, width, height),
        Img::new(out_a, width, height),
        Img::new(out_b, width, height),
    ]
}

impl ToLABBitmap for ImgRef<'_, RGBAPLU> {
    #[inline]
    fn to_lab(&self) -> Vec<GBitmap> {
        self.to_lab_pooled(&DssimPool::new())
    }
    #[inline]
    fn to_lab_pooled(&self, pool: &DssimPool) -> Vec<GBitmap> {
        rgb_to_lab(*self, |px, n| px.to_rgb(n), pool)
    }
}

impl ToLABBitmap for ImgRef<'_, RGBLU> {
    #[inline]
    fn to_lab(&self) -> Vec<GBitmap> {
        self.to_lab_pooled(&DssimPool::new())
    }
    #[inline]
    fn to_lab_pooled(&self, pool: &DssimPool) -> Vec<GBitmap> {
        rgb_to_lab(*self, |px, _n| px, pool)
    }
}

#[test]
fn cbrts1() {
    let mut totaldiff = 0.;
    let mut maxdiff: f64 = 0.;
    for i in (0..=10001).rev() {
        let x = (f64::from(i) / 10001.) as f32;
        let a = cbrt_poly(x);
        let actual = a * a * a;
        let expected = x;
        let absdiff = (f64::from(expected) - f64::from(actual)).abs();
        assert!(absdiff < 0.0002, "{expected} - {actual} = {} @ {x}", expected - actual);
        if i % 400 == 0 {
            println!("{:+0.3}", (expected - actual) * 255.);
        }
        totaldiff += absdiff;
        maxdiff = maxdiff.max(absdiff);
    }
    println!("1={totaldiff:0.6}; {maxdiff:0.8}");
    assert!(totaldiff < 0.0025, "{totaldiff}");
}

#[test]
fn cbrts2() {
    let mut totaldiff = 0.;
    let mut maxdiff: f64 = 0.;
    for i in (2000..=10001).rev() {
        let x = f64::from(i) / 10001.;
        let actual = f64::from(cbrt_poly(x as f32));
        let expected = x.cbrt();
        let absdiff = (expected - actual).abs();
        totaldiff += absdiff;
        maxdiff = maxdiff.max(absdiff);
        assert!(absdiff < 0.0000005, "{expected} - {actual} = {} @ {x}", expected - actual);
    }
    println!("2={totaldiff:0.6}; {maxdiff:0.8}");
    assert!(totaldiff < 0.0025, "{totaldiff}");
}

// Microbenchmarks for the non-blur pipeline hot paths (gamma->linear,
// RGB->LAB, downsample). Run with:
//   RUSTC_BOOTSTRAP=1 cargo bench -p dssim-core
#[cfg(test)]
mod perf_bench {
    extern crate test;
    use crate::image::RGBAPLU;
    use crate::linear::ToRGBAPLU;
    use crate::tolab::ToLABBitmap;
    use crate::image::Downsample;
    use imgref::*;
    use rgb::RGBA;
    use test::Bencher;

    fn xorshift(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    }

    fn rgba_u8(w: usize, h: usize) -> Vec<RGBA<u8>> {
        let mut s = 0x1234_5678u32;
        (0..w * h)
            .map(|_| {
                let v = xorshift(&mut s);
                RGBA::new(v as u8, (v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8)
            })
            .collect()
    }

    fn rgbaplu_img(w: usize, h: usize) -> ImgVec<RGBAPLU> {
        let buf = rgba_u8(w, h).to_rgbaplu();
        ImgVec::new(buf, w, h)
    }

    fn bench_to_rgbaplu(b: &mut Bencher, w: usize, h: usize) {
        let src = rgba_u8(w, h);
        b.iter(|| test::black_box(test::black_box(&src[..]).to_rgbaplu()));
    }

    fn bench_to_lab(b: &mut Bencher, w: usize, h: usize) {
        let img = rgbaplu_img(w, h);
        b.iter(|| test::black_box(test::black_box(&img).to_lab()));
    }

    fn bench_downsample(b: &mut Bencher, w: usize, h: usize) {
        let img = rgbaplu_img(w, h);
        b.iter(|| test::black_box(test::black_box(&img).downsample()));
    }

    #[bench] fn to_rgbaplu_320x200(b: &mut Bencher) { bench_to_rgbaplu(b, 320, 200); }
    #[bench] fn to_rgbaplu_1024x768(b: &mut Bencher) { bench_to_rgbaplu(b, 1024, 768); }
    #[bench] fn to_rgbaplu_1920x1080(b: &mut Bencher) { bench_to_rgbaplu(b, 1920, 1080); }

    #[bench] fn to_lab_320x200(b: &mut Bencher) { bench_to_lab(b, 320, 200); }
    #[bench] fn to_lab_1024x768(b: &mut Bencher) { bench_to_lab(b, 1024, 768); }
    #[bench] fn to_lab_1920x1080(b: &mut Bencher) { bench_to_lab(b, 1920, 1080); }

    #[bench] fn downsample_320x200(b: &mut Bencher) { bench_downsample(b, 320, 200); }
    #[bench] fn downsample_1024x768(b: &mut Bencher) { bench_downsample(b, 1024, 768); }
    #[bench] fn downsample_1920x1080(b: &mut Bencher) { bench_downsample(b, 1920, 1080); }
}
