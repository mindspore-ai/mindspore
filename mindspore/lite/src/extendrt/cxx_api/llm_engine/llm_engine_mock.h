/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_MOCK_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_MOCK_H_
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <map>
#include "ge/ge_ir_build.h"
#include "ge/ge_api_types.h"

namespace llm {

constexpr uint64_t kInvalidReqId = UINT64_MAX;
constexpr uint64_t kInvalidPrefixId = UINT64_MAX;

struct LLMEngineStatus {
  uint64_t empty_max_prompt_kv;  // 全量集群可容纳的KV
};

class LLMReq {
 public:
  LLMReq() = default;
  ~LLMReq() = default;

  void SetReqId(const uint64_t req_id) { req_id_ = req_id; }

  uint64_t GetReqId() const { return req_id_; }

  void SetPromptLength(const uint64_t prompt_length) { prompt_length_ = prompt_length; }

  uint64_t GetPromptLength() const { return prompt_length_; }

  void SetPromptClusterId(const uint64_t prompt_cluster_id) { prompt_cluster_id_ = prompt_cluster_id; }

  uint64_t GetPromptClusterId() const { return prompt_cluster_id_; }

  void SetDecoderClusterId(const uint64_t decoder_cluster_id) { decoder_cluster_id_ = decoder_cluster_id; }

  uint64_t GetDecoderClusterId() const { return decoder_cluster_id_; }

  void SetPrefixId(const uint64_t prefix_id) { prefix_id_ = prefix_id; }

  uint64_t GetPrefixId() const { return prefix_id_; }

 private:
  uint64_t req_id_{kInvalidReqId};
  // 请求prompt的句子长度，做完padding的值， 用于申请prompt的KVCache
  uint64_t prompt_length_{0UL};
  uint64_t prompt_cluster_id_{0UL};  // in/out， runPrompt的输出， runecoder的输入
  uint64_t decoder_cluster_id_{0UL};
  uint64_t prefix_id_{kInvalidPrefixId};
  int8_t reserved_[128];
};
class DecoderManager;
class PromptManager;
class LLMEngine {
 public:
  explicit LLMEngine(uint64_t cluster_id) : cluster_id_(cluster_id) {}
  ~LLMEngine();
  ge::Status LLMEngineInitialize(const std::vector<ge::ModelBufferData> &,
                                 const std::map<ge::AscendString, ge::AscendString> &);

  ge::Status LLMEngineInitializeV2(const std::map<ge::AscendString, std::vector<ge::ModelBufferData>> &,
                                   const std::map<ge::AscendString, ge::AscendString> &);

  static LLMEngineStatus FetchLLMEngineStatus();
  int64_t FetchLlmEngineQueueStatus();
  // API2：execute prompt
  ge::Status RunPromptAsync(const LLMReq &, const std::vector<ge::Tensor> &, ge::RunAsyncCallback);
  ge::Status RunPrompt(const LLMReq &, const std::vector<ge::Tensor> &, std::vector<ge::Tensor> &);

  // API3: Execute the Decoder calculation
  // a. Assign an idle index for the request
  // b. Fetch KVCache from the specified prompt cluster based on the request and write it to the corresponding idle
  // index c. Perform Decoder computation and asynchronously return the calculation result
  ge::Status RunDecoderAsync(const LLMReq &, const std::vector<ge::Tensor> &, ge::RunAsyncCallback);
  ge::Status RunDecoder(const LLMReq &, const std::vector<ge::Tensor> &, std::vector<ge::Tensor> &);

  // Externally notifies that the request has ended. If the request has already started execution, release the
  // placeholders associated with incremental inference. If the request has not yet started execution, remove it from
  // the queue.
  ge::Status LLMReqComplete(const LLMReq &);
  ge::Status LLMEngineFinalize();

  // Preload prompt prefix model to generate kv cache
  ge::Status PreloadPromptPrefix(const LLMReq &, const std::vector<ge::Tensor> &);

  // Release kv cache of prompt prefix model
  ge::Status ReleasePromptPrefix(const LLMReq &);

  ge::Status PullKv(const LLMReq &);
  ge::Status MergeKv(const uint64_t req_id, const int32_t batch_index);

  ge::Status RunDecoder(const std::vector<uint64_t> &, const std::vector<ge::Tensor> &, std::vector<ge::Tensor> &);

 private:
  std::shared_ptr<PromptManager> prompt_manager_;
  std::shared_ptr<DecoderManager> decoder_manager_;
  uint64_t cluster_id_;
  std::string role_;
  std::atomic<bool> is_initialized_{false};
  std::atomic<bool> is_finalized_{false};
};
}  // namespace llm
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_MOCK_H_
