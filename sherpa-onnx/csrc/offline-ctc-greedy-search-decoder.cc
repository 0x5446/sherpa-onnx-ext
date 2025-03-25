// sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

/// modified by tf @2025-03-24
/// modified by tf @2025-03-25: remove avg_logprob
std::vector<OfflineCtcDecoderResult> OfflineCtcGreedySearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value log_probs_length) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t vocab_size = static_cast<int32_t>(shape[2]);

  const int64_t *p_log_probs_length = log_probs_length.GetTensorData<int64_t>();

  std::vector<OfflineCtcDecoderResult> ans;
  ans.reserve(batch_size);

  for (int32_t b = 0; b != batch_size; ++b) {
    const float *p_log_probs =
        log_probs.GetTensorData<float>() + b * num_frames * vocab_size;

    OfflineCtcDecoderResult r;
    int64_t prev_id = -1;

    for (int32_t t = 0; t != static_cast<int32_t>(p_log_probs_length[b]); ++t) {
      // the start position of the logits of the current time step
      const float *current_logits = p_log_probs;
      
      // find the maximum value and its index
      auto max_elem = std::max_element(current_logits, current_logits + vocab_size);
      auto y = static_cast<int64_t>(std::distance(current_logits, max_elem));
      float max_val = *max_elem;
      
      if (y != blank_id_ && y != prev_id) {
        // step 1: calculate sum_exp = sum(exp(x_i - max_val)) to prevent overflow
        float sum_exp = 0.0f;
        for (int32_t i = 0; i < vocab_size; ++i) {
          sum_exp += std::exp(current_logits[i] - max_val);
        }
        
        // step 2: log_softmax(x_i) = x_i - max_val - log(sum_exp)
        float log_prob_value = current_logits[y] - max_val - std::log(sum_exp);
        
        r.tokens.push_back(y);
        r.timestamps.push_back(t);
        r.log_probs.push_back(log_prob_value);
      }
      
      // move to the next frame
      p_log_probs += vocab_size;
      prev_id = y;
    }  // for (int32_t t = 0; ...)

    ans.push_back(std::move(r));
  }
  return ans;
}

}  // namespace sherpa_onnx
