// 说 · Shuo — Qwen3.5-2B ONNX 文本后处理（Rust 端口）
//
// 对应 Python 版 `onnx_infer/llm.py`：embed_tokens + decoder（混合线性注意力层，
// past_conv / past_recurrent / past_key_values 三种状态）→ 采样 → 去掉 thinking 段。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use half::f16;

pub const VOCAB_SIZE: usize = 248_320;
pub const HIDDEN: usize = 2048;
pub const NUM_LAYERS: usize = 24;
pub const EOS_ID: i64 = 248_044;
pub const EOS_ID_ALT: i64 = 248_046;
pub const THINK_START: i64 = 248_068;
pub const THINK_END: i64 = 248_069;

pub const PRESET_STANDARD: &str = "standard";
pub const PRESET_MODERATE: &str = "moderate";
pub const PRESET_AGGRESSIVE: &str = "aggressive";
pub const PRESET_AGGRESSIVE_NO_PUNC: &str = "aggressive_no_punc";
pub const PRESET_TRANSLATE: &str = "translate";

pub fn preset_prompt(key: &str) -> Option<&'static str> {
  Some(match key {
    PRESET_STANDARD => {
      "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n\
规则：\n\
- 输入可能已带有部分标点，也可能完全没有标点。仅根据语义补充缺少的必要标点（逗号、句号、问号、感叹号），已有标点检查并改正；\n\
- 不补全缺失的内容或改写用词；\n\
- 标点宁少勿多，不确定时不加。\n\
要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n\
示例：\n\
输入：下周去北京的机票订了吗？酒店我看了一下有三家备选你倾向哪个。\n\
输出：下周去北京的机票订了吗？酒店我看了一下有三家备选，你倾向哪个？\n\
输入：昨晚加班到九点回到家已经十点半了吃完饭又处理了几封邮件然后十二点才睡。\n\
输出：昨晚加班到九点，回到家已经十点半了，吃完饭又处理了几封邮件，然后十二点才睡。\n\n\
请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    }
    PRESET_MODERATE => {
      "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n\
规则：\n\
- 输入可能已带有部分标点，也可能没有。根据语义补充缺少的标点，已有标点检查并改正，不补全缺失的内容；\n\
- 删除口癖和犹豫词（如嗯、啊、呃、那个、这个、就是说、然后、对吧、反正、大概、好像、就是、的话、之类的、这样的、对不对等）；\n\
- 不替换任何实词，保持原意。\n\
要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n\
示例：\n\
输入：嗯那个我觉得就是说这个方案好像不太行然后可能得改一下，呃我再想想有没有更好的办法呃那个我们的服务器大概可能下周二吧要维护你提前备份一下数据\n\
输出：我觉得这个方案不太行，可能得改一下，我再想想有没有更好的办法。我们的服务器下周二要维护，你提前备份一下数据。\n\
输入：啊对了我忘了说。昨天的会议纪要我发你了你先看一下有什么问题明天再讨论 这个文档我大概看了一下，然后觉得内容方面还要补充。你那边方便加一下吗\n\
输出：对了，我忘了说。昨天的会议纪要我发你了，你先看一下，有什么问题明天再讨论。我大概看了一下，觉得内容还要补充。你那边方便加一下吗？\n\n\
请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    }
    PRESET_AGGRESSIVE => {
      "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n\
规则：\n\
- 补充缺失标点。\n\
- 删除口癖和犹豫词嗯、啊、那个、就是说、然后、反正、大概、好像、就是、的话等。\n\
- 修正明显的同音或近音识别错误。\n\
- 将中文数字（包括幺、两、拐、洞等读法）转为阿拉伯数字并合并，金额只转数字（如三块五→3.5元）。\n\
- 将日期时间转为标准格式（如二零二六年→2026年，下午三点半→15:30）。\n\
- 将常见物理单位替换为符号（如米→m，平方米→m²，摄氏度→℃，千瓦→kW等）。\n\
- 将数学口头表达式转为书写形式（如x的平方→x²，根号→√，f x等于x平方加一→f(x)=x²+1）。\n\
- 将邮箱的at和点替换为@和.，中文数字合并（如一二三四五六@qq点com→123456@qq.com）。\n\n\
要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n\
示例：\n\
输入：嗯那个函数f x等于x平方加一，下周二下午三点维护，温度二十五度三。\n\
输出：函数f(x)=x²+1，下周二15:00维护，温度25.3℃。\n\
请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    }
    PRESET_AGGRESSIVE_NO_PUNC => {
      "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n\
规则：\n\
- 句之间标点改为空格，句尾标点去除，符号系统保留。\n\
- 删除口癖和犹豫词（嗯、啊、那个、就是说、然后、反正、大概、好像、就是、的话等）。\n\
- 修正明显的同音或近音识别错误。\n\
- 将中文数字（包括幺、两、拐、洞等读法）转为阿拉伯数字并合并，金额只转数字（如三块五→3.5元）。\n\
- 将日期时间转为标准格式（如二零二六年→2026年，下午三点半→15:30）。\n\
- 将常见物理单位替换为符号（如米→m，平方米→m²，摄氏度→℃，千瓦→kW等）。\n\
- 将数学口头表达式转为书写形式（如x的平方→x²，根号→√，f x等于x平方加一→f(x)=x²+1）。\n\
- 将邮箱的at和点替换为@和.，中文数字合并（如一二三四五六@qq点com→123456@qq.com）。\n\n\
要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n\
示例：\n\
输入：嗯那个函数f x等于x平方加一，下周二下午三点维护，温度二十五度三。\n\
输出：函数f(x)=x²+1 下周二15:00维护 温度25.3℃\n\n\
请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    }
    PRESET_TRANSLATE => {
      "你是一个文本转写工具，将口语化语音识别结果直接转为书面规范文本。不添加解释，不补全内容。\n\n\
规则：\n\
- 将文本翻译成英文，保持原意和语气；\n\
- 如果原文已经是英文，则原样输出；\n\
- 文本中的中文数字先转为阿拉伯数字再翻译（如三点→3:00、一百二十→120）。\n\
要求：只输出处理后的文本，不要加引号，不要任何解释或问候。\n\n\
示例：\n\
输入：早上好你今天怎么样。下午三点有个会议大概一个小时你参加吗\n\
输出：Good morning, how are you today? I have a meeting at 3 PM, about one hour. Will you join?\n\
输入：这个项目下周五截止。请把文件发到我的邮箱然后抄送给李经理\n\
输出：This project is due next Friday. Please send the file to my email and cc Manager Li.\n\
输入：帮我订一张明天上午十点到上海的火车票。我的电话是幺三九零零八六拐五二一\n\
输出：Please book a train ticket to Shanghai for tomorrow at 10 AM. My phone number is 13900867521.\n\
输入：房间温度二十五度空调功率两千瓦开两个小时\n\
输出：Room temperature is 25 degrees, air conditioner power is 2000W, run for 2 hours. \n\n\
请根据以上规则处理输入文本，不要多或少内容，语句不完整、不通顺也不自由发挥。"
    }
    _ => return None,
  })
}

pub fn preset_list() -> Vec<(&'static str, &'static str, &'static str)> {
  vec![
    (PRESET_STANDARD, "标准 - 只修标点", "Standard — punctuation only"),
    (PRESET_MODERATE, "适中 - 去口癖", "Moderate — remove fillers"),
    (PRESET_AGGRESSIVE, "激进 - 数字/单位/日期", "Aggressive — numbers/units/dates"),
    (PRESET_AGGRESSIVE_NO_PUNC, "激进 - 去标点", "Aggressive — no punctuation"),
    (PRESET_TRANSLATE, "翻译成英文", "Translate to English"),
  ]
}

pub struct LlmPipeline {
  embed: Session,
  decoder: Session,
  tokenizer: Tokenizer,
}

/// 采样参数（与 Python 默认一致）
#[derive(Clone, Copy)]
pub struct Sampling {
  pub max_new: usize,
  pub temperature: f32,
  pub top_k: usize,
  pub presence_penalty: f32,
}

impl Default for Sampling {
  fn default() -> Self {
    Self { max_new: 512, temperature: 1.0, top_k: 20, presence_penalty: 2.0 }
  }
}

impl LlmPipeline {
  pub fn load(model_root: &Path, threads: usize) -> Result<Self, String> {
    let onnx = model_root.join("onnx");
    let embed = Session::builder()
      .map_err(|e| e.to_string())?
      .with_intra_threads(threads.max(1))
      .map_err(|e| e.to_string())?
      .with_inter_threads(1)
      .map_err(|e| e.to_string())?
      .with_intra_op_spinning(false)
      .map_err(|e| e.to_string())?
      .commit_from_file(onnx.join("embed_tokens_q4f16.onnx"))
      .map_err(|e| format!("加载 embed_tokens 失败：{e}"))?;
    let decoder = Session::builder()
      .map_err(|e| e.to_string())?
      .with_intra_threads(threads.max(1))
      .map_err(|e| e.to_string())?
      .with_inter_threads(1)
      .map_err(|e| e.to_string())?
      .with_intra_op_spinning(false)
      .map_err(|e| e.to_string())?
      .commit_from_file(onnx.join("decoder_model_merged_q4f16.onnx"))
      .map_err(|e| format!("加载 decoder 失败：{e}"))?;
    let tok_path = model_root.join("tokenizer.json");
    let tokenizer =
      Tokenizer::from_file(&tok_path).map_err(|e| format!("加载 tokenizer 失败：{e}"))?;
    Ok(LlmPipeline { embed, decoder, tokenizer })
  }

  pub fn is_available(model_root: &Path) -> bool {
    let onnx = model_root.join("onnx");
    onnx.join("embed_tokens_q4f16.onnx").is_file()
      && onnx.join("embed_tokens_q4f16.onnx_data").is_file()
      && onnx.join("decoder_model_merged_q4f16.onnx").is_file()
      && onnx.join("decoder_model_merged_q4f16.onnx_data").is_file()
      && model_root.join("tokenizer.json").is_file()
  }

  /// Qwen chat 模板（`enable_thinking=false`）渲染结果。
  ///
  /// 与 `Qwen3.5-2B-ONNX-OPT/chat_template.jinja` 在 `enable_thinking=false` 时逐字一致：
  /// system / user 内容先 `trim`，结尾固定带 ` thinking\n\n</think>\n\n`。
  /// （缺这段会让模型行为与 Python 版不同，见 `tests::chat_prompt_matches_template`）
  pub fn chat_prompt(system: &str, user: &str) -> String {
    let mut out = String::new();
    let sys = system.trim();
    if !sys.is_empty() {
      out.push_str("<|im_start|>system\n");
      out.push_str(sys);
      out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user.trim());
    out.push_str("<|im_end|>\n<|im_start|>assistant\n");
    out.push_str("<think>\n\n</think>\n\n");
    out
  }

  fn empty_state(&self) -> Vec<(String, SessionInputValue<'static>)> {
    let mut out: Vec<(String, SessionInputValue<'static>)> = Vec::new();
    for i in 0..NUM_LAYERS {
      if i % 4 == 3 {
        for kind in ["key", "value"] {
          let t = Tensor::from_array((
            [1i64, 2, 0, 256],
            vec![f16::ZERO; 0],
          ))
          .expect("空 KV");
          out.push((format!("past_key_values.{i}.{kind}"), t.into()));
        }
      } else {
        let conv = Tensor::from_array(([1i64, 6144, 3], vec![f16::ZERO; 6144 * 3])).expect("空 conv");
        out.push((format!("past_conv.{i}"), conv.into()));
        let rec = Tensor::from_array(([1i64, 16, 128, 128], vec![f16::ZERO; 16 * 128 * 128]))
          .expect("空 recurrent");
        out.push((format!("past_recurrent.{i}"), rec.into()));
      }
    }
    out
  }

  /// 把 `present_*` 输出映射回 `past_*` 输入名。
  fn state_from_outputs(
    outputs: &mut ort::session::SessionOutputs<'_>,
  ) -> Vec<(String, SessionInputValue<'static>)> {
    let names: Vec<String> = outputs.keys().map(|s| s.to_string()).collect();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
      let input_name = if let Some(rest) = name.strip_prefix("present_key_values.") {
        format!("past_key_values.{rest}")
      } else if let Some(rest) = name.strip_prefix("present_conv.") {
        format!("past_conv.{rest}")
      } else if let Some(rest) = name.strip_prefix("present_recurrent.") {
        format!("past_recurrent.{rest}")
      } else if let Some(rest) = name.strip_prefix("present.") {
        format!("past_key_values.{rest}")
      } else {
        continue;
      };
      if let Some(v) = outputs.remove(&name) {
        out.push((input_name, v.into()));
      }
    }
    out
  }

  pub fn generate(&mut self, system: &str, user: &str, cfg: Sampling) -> Result<String, String> {
    let prompt = Self::chat_prompt(system, user);
    let enc = self.tokenizer.encode(prompt.as_str(), false).map_err(|e| e.to_string())?;
    let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
    if ids.is_empty() {
      return Ok(String::new());
    }
    let seq = ids.len();
    let ids_i64: Vec<i64> = ids.clone();

    let emb_t = Tensor::from_array(([1i64, seq as i64], ids_i64))
      .map_err(|e| format!("构造 input_ids 失败：{e}"))?;
    let emb_out = self
      .embed
      .run(ort::inputs!["input_ids" => emb_t])
      .map_err(|e| format!("embed 推理失败：{e}"))?;
    let (_, emb_data) = emb_out["inputs_embeds"]
      .try_extract_tensor::<f32>()
      .map_err(|e| format!("读取 inputs_embeds 失败：{e}"))?;
    let embeds: Vec<f32> = emb_data.to_vec();
    drop(emb_out);

    let mask = Tensor::from_array(([1i64, seq as i64], vec![1i64; seq]))
      .map_err(|e| e.to_string())?;
    let pos: Vec<i64> = (0..seq as i64).collect();
    let mut pos3: Vec<i64> = Vec::with_capacity(seq * 3);
    for _ in 0..3 {
      pos3.extend_from_slice(&pos);
    }
    let pos_t = Tensor::from_array(([3i64, 1, seq as i64], pos3)).map_err(|e| e.to_string())?;
    let keep = Tensor::from_array(((), vec![1i64])).map_err(|e| e.to_string())?;
    let emb_t = Tensor::from_array(([1i64, seq as i64, HIDDEN as i64], embeds))
      .map_err(|e| e.to_string())?;

    let mut inputs: Vec<(String, SessionInputValue)> = vec![
      ("inputs_embeds".into(), emb_t.into()),
      ("attention_mask".into(), mask.into()),
      ("position_ids".into(), pos_t.into()),
      ("num_logits_to_keep".into(), keep.into()),
    ];
    inputs.extend(self.empty_state());

    let mut out = self
      .decoder
      .run(inputs)
      .map_err(|e| format!("decoder 预填充失败：{e}"))?;
    let logits = out
      .get("logits")
      .ok_or_else(|| "缺少 logits".to_string())?
      .try_extract_tensor::<f16>()
      .map_err(|e| format!("读取 logits 失败：{e}"))?
      .1
      .to_vec();
    let mut state = Self::state_from_outputs(&mut out);
    drop(out);

    let mut generated: Vec<i64> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();
    let mut next = sample_token_f16(&logits, cfg, &mut seen, &mut generated)?;
    generated.push(next);
    seen.insert(next);
    let mut cur_pos = seq;

    while generated.len() < cfg.max_new {
      if next == EOS_ID || next == EOS_ID_ALT {
        break;
      }
      let id_t = Tensor::from_array(([1i64, 1], vec![next])).map_err(|e| e.to_string())?;
      let emb = self
        .embed
        .run(ort::inputs!["input_ids" => id_t])
        .map_err(|e| format!("embed step 失败：{e}"))?;
      let (_, emb_data) = emb["inputs_embeds"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
      let embed_vec = emb_data.to_vec();
      drop(emb);

      let emb_t = Tensor::from_array(([1i64, 1, HIDDEN as i64], embed_vec)).map_err(|e| e.to_string())?;
      let mask = Tensor::from_array(([1i64, cur_pos as i64 + 1], vec![1i64; cur_pos + 1]))
        .map_err(|e| e.to_string())?;
      let pos_t = Tensor::from_array(([3i64, 1, 1], vec![cur_pos as i64; 3])).map_err(|e| e.to_string())?;
      let keep = Tensor::from_array(((), vec![1i64])).map_err(|e| e.to_string())?;

      let mut inputs: Vec<(String, SessionInputValue)> = vec![
        ("inputs_embeds".into(), emb_t.into()),
        ("attention_mask".into(), mask.into()),
        ("position_ids".into(), pos_t.into()),
        ("num_logits_to_keep".into(), keep.into()),
      ];
      inputs.extend(std::mem::take(&mut state));

      let mut out = self.decoder.run(inputs).map_err(|e| format!("decoder step 失败：{e}"))?;
      let logits = out
        .get("logits")
        .ok_or_else(|| "缺少 logits".to_string())?
        .try_extract_tensor::<f16>()
        .map_err(|e| format!("读取 logits 失败：{e}"))?
        .1
        .to_vec();
      state = Self::state_from_outputs(&mut out);
      drop(out);

      next = sample_token_f16(&logits, cfg, &mut seen, &mut generated)?;
      generated.push(next);
      seen.insert(next);
      cur_pos += 1;
    }

    // 去掉 thinking 段
    let mut out_ids = &generated[..];
    if let Some(pos) = generated.iter().rposition(|&v| v == THINK_END) {
      out_ids = &generated[pos + 1..];
    }
    if out_ids.first() == Some(&THINK_START) {
      out_ids = &out_ids[1..];
    }
    let u32_ids: Vec<u32> = out_ids.iter().map(|&v| v as u32).collect();
    self
      .tokenizer
      .decode(&u32_ids, true)
      .map_err(|e| format!("解码失败：{e}"))
  }
}

fn sample_token_f16(
  logits_f16: &[f16],
  cfg: Sampling,
  seen: &mut HashSet<i64>,
  generated: &mut [i64],
) -> Result<i64, String> {
  if logits_f16.is_empty() {
    return Err("logits 为空".into());
  }
  let mut logits: Vec<f32> = logits_f16.iter().map(|v| v.to_f32()).collect();
  if cfg.presence_penalty != 0.0 {
    for &id in seen.iter() {
      if let Some(v) = logits.get_mut(id as usize) {
        *v -= cfg.presence_penalty;
      }
    }
  }
  if cfg.temperature <= 0.0 {
    return Ok(argmax(&logits) as i64);
  }
  if (cfg.temperature - 1.0).abs() > f32::EPSILON {
    for v in logits.iter_mut() {
      *v /= cfg.temperature;
    }
  }
  if cfg.top_k > 0 && cfg.top_k < logits.len() {
    let mut copy = logits.clone();
    let (_, kth, _) = copy.select_nth_unstable_by(cfg.top_k - 1, |a, b| b.partial_cmp(a).unwrap());
    let threshold = *kth;
    let kept: Vec<usize> = (0..logits.len()).filter(|&i| logits[i] >= threshold).collect();
    // 与 Python 版一致：并列（f16 logits 里很常见）全部保留，不硬截到 k 个
    if kept.is_empty() {
      return Ok(argmax(&logits) as i64);
    }
    let max = kept.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = Vec::with_capacity(kept.len());
    let mut sum = 0.0f32;
    for &i in &kept {
      let p = (logits[i] - max).exp();
      sum += p;
      probs.push((i, p));
    }
    if !(sum.is_finite()) || sum <= 0.0 {
      return Ok(kept[0] as i64);
    }
    let r = rand_f32() * sum;
    let mut acc = 0.0f32;
    for (i, p) in probs {
      acc += p;
      if acc >= r {
        return Ok(i as i64);
      }
    }
    return Ok(kept[kept.len() - 1] as i64);
  }
  // temperature 采样（无 top-k）
  let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
  let mut probs: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
  let sum: f32 = probs.iter().sum();
  if !sum.is_finite() || sum <= 0.0 {
    return Ok(argmax(&logits) as i64);
  }
  let r = rand_f32() * sum;
  let mut acc = 0.0f32;
  for (i, p) in probs.iter_mut().enumerate() {
    acc += *p;
    if acc >= r {
      return Ok(i as i64);
    }
  }
  let _ = generated;
  Ok((probs.len() - 1) as i64)
}

fn argmax(v: &[f32]) -> usize {
  let mut best = 0usize;
  let mut best_v = f32::NEG_INFINITY;
  for (i, &x) in v.iter().enumerate() {
    if x > best_v {
      best_v = x;
      best = i;
    }
  }
  best
}

/// 轻量随机数（xorshift64*），避免引入 rand 依赖。
fn rand_f32() -> f32 {
  use std::sync::atomic::{AtomicU64, Ordering};
  static STATE: AtomicU64 = AtomicU64::new(0);
  let mut s = STATE.load(Ordering::Relaxed);
  if s == 0 {
    let seed = std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_nanos() as u64)
      .unwrap_or(0x9E37_79B9_7F4A_7C15);
    s = seed | 1;
  }
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  STATE.store(s, Ordering::Relaxed);
  let v = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
  ((v >> 40) as f32) / (1u32 << 24) as f32
}

/// 自定义提示词（与 Python 版共用 `~/.shuo/custom_prompts.json`）
pub fn custom_prompts_path() -> PathBuf {
  crate::config::user_dir().join("custom_prompts.json")
}

pub fn load_custom_prompts() -> std::collections::BTreeMap<String, String> {
  std::fs::read_to_string(custom_prompts_path())
    .ok()
    .and_then(|s| serde_json::from_str(&s).ok())
    .unwrap_or_default()
}

pub fn save_custom_prompts(p: &std::collections::BTreeMap<String, String>) -> Result<(), String> {
  let path = custom_prompts_path();
  if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
  }
  let s = serde_json::to_string_pretty(p).map_err(|e| e.to_string())?;
  std::fs::write(path, s).map_err(|e| e.to_string())
}

#[allow(dead_code)]
pub fn now() -> Instant {
  Instant::now()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn chat_prompt_matches_python_template() {
    let p = LlmPipeline::chat_prompt(" 系统提示 \n", "  你好  ");
    assert_eq!(p, "<|im_start|>system\n系统提示<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    // 没有 system 时不能出现空的 system 段
    let p = LlmPipeline::chat_prompt("   ", "你好");
    assert!(p.starts_with("<|im_start|>user\n你好"));
  }

  /// 与 Python `apply_chat_template(..., enable_thinking=False)` 的 token id 逐项对齐。
  /// 参考值由 `tokenizers.Tokenizer.encode(text)` 生成。
  #[test]
  fn chat_prompt_token_ids_match_python() {
    let root = std::path::Path::new("../Qwen3.5-2B-ONNX-OPT");
    if !root.join("tokenizer.json").is_file() {
      return; // 模型没下载时跳过（CI）
    }
    let tok = Tokenizer::from_file(root.join("tokenizer.json")).expect("tokenizer");
    let prompt = LlmPipeline::chat_prompt(
      "你是一个文本转写工具：只补标点。",
      "  嗯那个我觉得就是说这个方案好像不太行然后可能得改一下  ",
    );
    let ids: Vec<u32> = tok.encode(prompt.as_str(), false).expect("encode").get_ids().to_vec();
    #[rustfmt::skip]
    let expected: Vec<u32> = vec![
      248045, 8678, 198, 133653, 109120, 96156, 96674, 99445, 4960, 96227, 96770, 141111, 1710,
      248046, 198, 248045, 846, 198, 107680, 103703, 108747, 124360, 96488, 97791, 100617,
      104000, 95835, 98118, 96539, 95930, 96332, 97434, 248046, 198, 248045, 74455, 198, 248068,
      271, 248069, 271,
    ];
    assert_eq!(ids, expected);
  }

  #[test]
  fn sampling_respects_top_k() {
    // 构造一个 top-2 很明确的分布，反复采样不应跑到冷门 token
    let mut logits = vec![f16::from_f32(-10.0); 100];
    logits[3] = f16::from_f32(5.0);
    logits[7] = f16::from_f32(4.0);
    let mut seen = HashSet::new();
    let mut gen_ids = Vec::new();
    for _ in 0..200 {
      let id = sample_token_f16(&logits, Sampling::default(), &mut seen, &mut gen_ids).unwrap();
      assert!(id == 3 || id == 7, "采样跑出 top-k：{id}");
    }
  }

  /// 端到端跑一次真实模型（需要本地 1.4G 模型）：
  /// `cargo test --release -- --ignored llm_end_to_end --nocapture`
  #[test]
  #[ignore = "需要本地 LLM 模型"]
  fn llm_end_to_end() {
    let root = std::path::Path::new("../Qwen3.5-2B-ONNX-OPT");
    if !LlmPipeline::is_available(root) {
      eprintln!("跳过：本地没有 LLM 模型");
      return;
    }
    let mut p = LlmPipeline::load(root, 8).expect("加载 LLM");
    let t0 = Instant::now();
    let out = p
      .generate(
        preset_prompt(PRESET_MODERATE).expect("preset"),
        "嗯那个我觉得就是说这个方案好像不太行然后可能得改一下",
        Sampling { max_new: 200, ..Default::default() },
      )
      .expect("生成");
    println!("OUT({:.2}s): {out}", t0.elapsed().as_secs_f64());
    assert!(!out.trim().is_empty());
  }
}
