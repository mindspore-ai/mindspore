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
#include "external/llm_engine.h"
namespace llm {
LLMEngine::~LLMEngine() {}

ge::Status LLMEngine::LLMEngineInitialize(const std::vector<ge::ModelBufferData> &,
                                          const std::map<ge::AscendString, ge::AscendString> &) {
  return ge::GRAPH_SUCCESS;
}

ge::Status LLMEngine::LLMEngineFinalize() { return ge::GRAPH_SUCCESS; }

ge::Status LLMEngine::RunPromptAsync(const LLMReq &, const std::vector<ge::Tensor> &, ge::RunAsyncCallback) {
  return ge::GRAPH_SUCCESS;
}

ge::Status LLMEngine::RunPrompt(const LLMReq &, const std::vector<ge::Tensor> &, std::vector<ge::Tensor> &) {
  return ge::GRAPH_SUCCESS;
}

ge::Status LLMEngine::RunDecoderAsync(const LLMReq &, const std::vector<ge::Tensor> &, ge::RunAsyncCallback) {
  return ge::GRAPH_SUCCESS;
}

ge::Status LLMEngine::RunDecoder(const LLMReq &, const std::vector<ge::Tensor> &, std::vector<ge::Tensor> &) {
  return ge::GRAPH_SUCCESS;
}

ge::Status LLMEngine::LLMReqComplete(const LLMReq &) { return ge::GRAPH_SUCCESS; }
}  // namespace llm
