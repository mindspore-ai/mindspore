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
#include "external/llm_engine_types.h"

namespace llm {
class DecoderManager;
class PromptManager;
class LLMEngine {
 public:
  explicit LLMEngine(uint64_t cluster_id) : cluster_id_(cluster_id) {}
  ~LLMEngine();
  ge::Status LLMEngineInitialize(const std::vector<ge::ModelBufferData> &,
                                 const std::map<ge::AscendString, ge::AscendString> &);
  static LLMEngineStatus fetchLLMEngineStatus();
  int64_t FetchLlmEngineQueueStatus();

  ge::Status RunPromptAsync(const LLMReq &, const std::vector<ge::Tensor> &, ge::RunAsyncCallback);
  ge::Status RunPrompt(const LLMReq &, const std::vector<ge::Tensor> &, std::vector<ge::Tensor> &);

  ge::Status RunDecoderAsync(const LLMReq &, const std::vector<ge::Tensor> &, ge::RunAsyncCallback);
  ge::Status RunDecoder(const LLMReq &, const std::vector<ge::Tensor> &, std::vector<ge::Tensor> &);

  ge::Status LLMReqComplete(const LLMReq &);
  ge::Status LLMEngineFinalize();

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
