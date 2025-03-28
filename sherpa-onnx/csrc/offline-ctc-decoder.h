// sherpa-onnx/csrc/offline-ctc-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_DECODER_H_

#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

struct OfflineCtcDecoderResult {
  /// The decoded token IDs
  std::vector<int64_t> tokens;

  /// The decoded word IDs
  /// Note: tokens.size() is usually not equal to words.size()
  /// words is empty for greedy search decoding.
  /// it is not empty when an HLG graph or an HLG graph is used.
  std::vector<int32_t> words;

  /// timestamps[i] contains the output frame index where tokens[i] is decoded.
  /// Note: The index is after subsampling
  ///
  /// tokens.size() == timestamps.size()
  std::vector<int32_t> timestamps;


  /// added by tf @2025-03-24 BOF
  /// log_probs[i] contains the log probability of tokens[i] at
  /// timestamps[i]
  std::vector<float> log_probs;

  /// avg_logprob: the average log probability of the decoded tokens
  // float avg_logprob; // modified by tf @2025-03-25: remove it
  /// added by tf @2025-03-24 EOF
};

class OfflineCtcDecoder {
 public:
  virtual ~OfflineCtcDecoder() = default;

  /** Run CTC decoding given the output from the encoder model.
   *
   * @param log_probs A 3-D tensor of shape (N, T, vocab_size) containing
   *                  lob_probs.
   * @param log_probs_length A 1-D tensor of shape (N,) containing number
   *                         of valid frames in log_probs before padding.
   *
   * @return Return a vector of size `N` containing the decoded results.
   */
  virtual std::vector<OfflineCtcDecoderResult> Decode(
      Ort::Value log_probs, Ort::Value log_probs_length) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_DECODER_H_
