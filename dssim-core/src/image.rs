#![allow(dead_code)]

use imgref::*;
use rgb::*;
#[cfg(not(feature = "threads"))]
use crate::lieon as rayon;
use rayon::prelude::*;

/// RGBA, but: premultiplied alpha, linear (using sRGB primaries, but not its gamma curve), f32 unit scale 0..1
pub type RGBAPLU = RGBA<f32>;
/// RGB, but: linear (using sRGB primaries, but not its gamma curve), f32 unit scale 0..1
pub type RGBLU = RGB<f32>;

/// L\*a\*b\*b, but using float units (values are 100× smaller than in usual integer representation)
#[derive(Debug, Copy, Clone)]
pub struct LAB {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl std::ops::Mul<Self> for LAB {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            l: self.l * other.l,
            a: self.a * other.a,
            b: self.b * other.b,
        }
    }
}

impl std::ops::Mul<LAB> for f32 {
    type Output = LAB;

    fn mul(self, other: LAB) -> Self::Output {
        LAB {
            l: self * other.l,
            a: self * other.a,
            b: self * other.b,
        }
    }
}

impl std::ops::Mul<f32> for LAB {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        Self {
            l: self.l * other,
            a: self.a * other,
            b: self.b * other,
        }
    }
}

impl std::ops::Add<Self> for LAB {
    type Output = Self;

    fn add(self, other: Self::Output) -> Self::Output {
        Self {
            l: self.l + other.l,
            a: self.a + other.a,
            b: self.b + other.b,
        }
    }
}

impl std::ops::Add<f32> for LAB {
    type Output = Self;

    fn add(self, other: f32) -> Self::Output {
        Self {
            l: self.l + other,
            a: self.a + other,
            b: self.b + other,
        }
    }
}

impl std::ops::Sub<Self> for LAB {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            l: self.l - other.l,
            a: self.a - other.a,
            b: self.b - other.b,
        }
    }
}

impl LAB {
    pub(crate) fn avg(self) -> f32 {
        (self.l + self.a + self.b) * (1. / 3.)
    }
}

impl From<LAB> for f64 {
    fn from(other: LAB) -> Self {
        (Self::from(other.l) + Self::from(other.a) + Self::from(other.b)) * (1. / 3.)
    }
}

impl From<LAB> for f32 {
    fn from(other: LAB) -> Self {
        other.avg()
    }
}

impl std::ops::Div<Self> for LAB {
    type Output = Self;

    fn div(self, other: Self::Output) -> Self::Output {
        Self {
            l: self.l / other.l,
            a: self.a / other.a,
            b: self.b / other.b,
        }
    }
}

/// Component-wise averaging of pixel values used by `Downsample` to support arbitrary pixel types
///
/// Used to naively resample 4 high-res pixels into one low-res pixel
#[doc(hidden)]
pub trait Average4 {
    fn average4(a: Self, b: Self, c: Self, d: Self) -> Self;
}

impl Average4 for f32 {
    fn average4(a: Self, b: Self, c: Self, d: Self) -> Self {
        (a + b + c + d) * 0.25
    }
}

impl Average4 for RGBAPLU {
    fn average4(a: Self, b: Self, c: Self, d: Self) -> Self {
        RGBAPLU {
            r: Average4::average4(a.r, b.r, c.r, d.r),
            g: Average4::average4(a.g, b.g, c.g, d.g),
            b: Average4::average4(a.b, b.b, c.b, d.b),
            a: Average4::average4(a.a, b.a, c.a, d.a),
        }
    }
}

impl Average4 for RGBLU {
    fn average4(a: Self, b: Self, c: Self, d: Self) -> Self {
        RGBLU {
            r: Average4::average4(a.r, b.r, c.r, d.r),
            g: Average4::average4(a.g, b.g, c.g, d.g),
            b: Average4::average4(a.b, b.b, c.b, d.b),
        }
    }
}

pub(crate) trait ToRGB {
    fn to_rgb(self, n: usize) -> RGBLU;
}

impl ToRGB for RGBAPLU {
    fn to_rgb(self, n: usize) -> RGBLU {
        let mut r = self.r;
        let mut g = self.g;
        let mut b = self.b;
        let a = self.a;
        if a < 255.0 {
            if (n & 16) != 0 {
                r += 1.0 - a;
            }
            if (n & 8) != 0 {
                g += 1.0 - a; // assumes premultiplied alpha
            }
            if (n & 32) != 0 {
                b += 1.0 - a;
            }
        }

        RGBLU { r, g, b }
    }
}

/// You can customize how images are downsampled
///
/// Multi-scale DSSIM needs to scale images down. This is it. It's supposed to return the same type of image, but half the size.
///
/// There is a default implementation that just averages 4 neighboring pixels.
#[doc(hidden)]
pub trait Downsample {
    type Output;
    fn downsample(&self) -> Option<Self::Output>;
}

impl<T> Downsample for ImgVec<T> where T: Average4 + Copy + Sync + Send {
    type Output = Self;

    fn downsample(&self) -> Option<Self::Output> {
        self.as_ref().downsample()
    }
}

impl<T> Downsample for ImgRef<'_, T> where T: Average4 + Copy + Sync + Send {
    type Output = ImgVec<T>;

    fn downsample(&self) -> Option<Self::Output> {
        let stride = self.stride();
        let width = self.width();
        let height = self.height();

        if width < 8 || height < 8 {
            return None;
        }

        let half_height = height / 2;
        let half_width = width / 2;
        let buf = self.buf();

        // Each output row y averages input rows 2y and 2y+1, so rows are
        // independent. Write them in parallel into the uninitialized tail.
        let mut scaled = Vec::with_capacity(half_width * half_height);
        scaled.spare_capacity_mut()
            .par_chunks_exact_mut(half_width)
            .take(half_height)
            .enumerate()
            .for_each(|(y, out_row)| {
                let top = &buf[(2 * y) * stride..][..half_width * 2];
                let bot = &buf[(2 * y + 1) * stride..][..half_width * 2];
                for x in 0..half_width {
                    out_row[x].write(Average4::average4(top[2 * x], top[2 * x + 1], bot[2 * x], bot[2 * x + 1]));
                }
            });
        // SAFETY: every cell of scaled[..half_width*half_height] was written above.
        unsafe { scaled.set_len(half_width * half_height); }
        Some(Img::new(scaled, half_width, half_height))
    }
}

#[allow(dead_code)]
pub(crate) fn worst(input: ImgRef<'_, f32>) -> ImgVec<f32> {
    let stride = input.stride();
    let half_height = input.height() / 2;
    let half_width = input.width() / 2;

    if half_height < 4 || half_width < 4 {
        return input.new_buf(input.buf().to_vec());
    }

    let mut scaled = Vec::with_capacity(half_width * half_height);
    scaled.extend(input.buf().chunks(stride * 2).take(half_height).flat_map(|pair| {
        let (top, bot) = pair.split_at(stride);
        let top = &top[0..half_width * 2];
        let bot = &bot[0..half_width * 2];

        top.chunks_exact(2).zip(bot.chunks_exact(2)).map(|(a,b)| {
            a[0].min(a[1]).min(b[0].min(b[1]))
        })
    }));

    assert_eq!(half_width * half_height, scaled.len());
    Img::new(scaled, half_width, half_height)
}

#[allow(dead_code)]
pub(crate) fn avgworst(input: ImgRef<'_, f32>) -> ImgVec<f32> {
    let stride = input.stride();
    let half_height = input.height() / 2;
    let half_width = input.width() / 2;

    if half_height < 4 || half_width < 4 {
        return input.new_buf(input.buf().to_vec());
    }

    let mut scaled = Vec::with_capacity(half_width * half_height);
    scaled.extend(input.buf().chunks(stride * 2).take(half_height).flat_map(|pair| {
        let (top, bot) = pair.split_at(stride);
        let top = &top[0..half_width * 2];
        let bot = &bot[0..half_width * 2];

        top.chunks_exact(2)
            .zip(bot.chunks_exact(2))
            .map(|(a, b)| (a[0] + a[1] + b[0] + b[1]).mul_add(0.25, a[0].min(a[1]).min(b[0].min(b[1]))) * 0.5)
    }));

    assert_eq!(half_width * half_height, scaled.len());
    Img::new(scaled, half_width, half_height)
}

#[allow(dead_code)]
pub(crate) fn avg(input: ImgRef<'_, f32>) -> ImgVec<f32> {
    let stride = input.stride();
    let half_height = input.height() / 2;
    let half_width = input.width() / 2;

    if half_height < 4 || half_width < 4 {
        return input.new_buf(input.buf().to_vec());
    }

    let mut scaled = Vec::with_capacity(half_width * half_height);
    scaled.extend(input.buf().chunks(stride * 2).take(half_height).flat_map(|pair| {
        let (top, bot) = pair.split_at(stride);
        let top = &top[0..half_width * 2];
        let bot = &bot[0..half_width * 2];

        top.chunks_exact(2).zip(bot.chunks_exact(2)).map(|(a,b)| {
            (a[0] + a[1] + b[0] + b[1]) * 0.25
        })
    }));

    assert_eq!(half_width * half_height, scaled.len());
    Img::new(scaled, half_width, half_height)
}
