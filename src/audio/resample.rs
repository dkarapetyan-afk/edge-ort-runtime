//! Lightweight linear resampler (no FFT dependency).

/// Downmix interleaved multi-channel f32 to mono by averaging channels.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / channels;
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += interleaved[i * channels + c];
        }
        out.push(sum / channels as f32);
    }
    out
}

/// Linear-interpolation resample from `from_rate` to `to_rate` (mono).
pub fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    if from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let frac = (src - i0 as f64) as f32;
        let s = input[i0] * (1.0 - frac) + input[i1] * frac;
        out.push(s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo() {
        let interleaved = vec![1.0, 3.0, 2.0, 4.0];
        let mono = downmix_to_mono(&interleaved, 2);
        assert_eq!(mono, vec![2.0, 3.0]);
    }

    #[test]
    fn resample_identity() {
        let x = vec![0.0, 1.0, 0.0, -1.0];
        assert_eq!(resample_linear(&x, 16_000, 16_000), x);
    }

    #[test]
    fn resample_upsample_length() {
        let x: Vec<f32> = (0..160).map(|i| (i as f32 / 160.0).sin()).collect();
        let y = resample_linear(&x, 8_000, 16_000);
        assert!((y.len() as i32 - 320).abs() <= 1);
    }

    #[test]
    fn resample_downsample_length() {
        let x: Vec<f32> = (0..320).map(|i| (i as f32).sin() * 0.01).collect();
        let y = resample_linear(&x, 48_000, 16_000);
        assert!((y.len() as i32 - 106).abs() <= 2);
    }
}
