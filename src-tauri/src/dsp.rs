// 说 · Shuo — 重采样（rubato sinc，离线整段）
//
// SPDX-License-Identifier: GPL-3.0-or-later

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
  Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

const CHUNK: usize = 1024;

/// 任意采样率 → 任意采样率（单声道）。延迟已补偿，输出长度按比例截断。
pub fn resample(input: &[f32], in_rate: u32, out_rate: u32) -> Result<Vec<f32>, String> {
  if in_rate == out_rate || input.is_empty() {
    return Ok(input.to_vec());
  }
  let ratio = out_rate as f64 / in_rate as f64;
  let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
    .oversampling_factor(256)
    .interpolation(SincInterpolationType::Quadratic);
  let mut rs = Async::<f32>::new_sinc(ratio, 1.1, &params, CHUNK, 1, FixedAsync::Input)
    .map_err(|e| format!("创建重采样器失败：{e}"))?;
  let delay = rs.output_delay();
  let expected = ((input.len() as f64) * ratio).round() as usize;

  let mut out: Vec<f32> = Vec::with_capacity(expected + delay + CHUNK);
  let mut in_pos = 0usize;
  let mut buf_out = vec![0.0f32; rs.output_frames_max()];
  let mut zero_rounds = 0usize;
  // 多喂一轮静音，把 sinc 尾音冲出来
  let max_rounds = input.len().div_ceil(CHUNK) + 4;
  while out.len() < expected + delay && zero_rounds < max_rounds {
    let mut chunk = vec![0.0f32; CHUNK];
    if in_pos < input.len() {
      let n = (input.len() - in_pos).min(CHUNK);
      chunk[..n].copy_from_slice(&input[in_pos..in_pos + n]);
      in_pos += n;
    } else {
      zero_rounds += 1;
    }
    let inp = InterleavedSlice::new(&chunk, 1, CHUNK).map_err(|e| e.to_string())?;
    let cap = buf_out.len();
    let mut outbuf = InterleavedSlice::new_mut(&mut buf_out, 1, cap).map_err(|e| e.to_string())?;
    let (_, n) = rs
      .process_into_buffer(&inp, &mut outbuf, None)
      .map_err(|e| format!("重采样失败：{e}"))?;
    out.extend_from_slice(&buf_out[..n]);
  }

  if out.len() > delay {
    out.drain(..delay);
  }
  out.truncate(expected);
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn length_ratio() {
    let input: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.01).sin()).collect();
    let out = resample(&input, 48000, 16000).unwrap();
    assert!((out.len() as i64 - 16000).abs() <= 2, "len={}", out.len());
  }
}
