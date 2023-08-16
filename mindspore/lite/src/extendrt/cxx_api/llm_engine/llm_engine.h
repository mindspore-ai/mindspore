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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_H_
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/types.h"
#include "include/api/status.h"

namespace mindspore {
struct LLMReq {
  uint64_t req_id = 0;
  uint64_t prompt_length = 0;
  uint64_t prompt_cluster_id = 0;
  uint64_t decoder_cluster_id = 0;
};

struct LLMEngineStatus {
  uint64_t empty_max_prompt_kv = 0;
};

enum LLMRole {
  kLLMRolePrompt = 0,
  kLLMRoleDecoder = 1,
};

class LLMEnginePluginBase;
class MS_API LLMEngine {
 public:
  LLMEngine();
  ~LLMEngine() = default;
  Status Init(const std::vector<std::string> &model_paths, LLMRole role, uint64_t cluster_id,
              const std::map<std::string, std::string> &options);
  void Finalize();
  Status Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);
  Status CompleteRequest(const LLMReq &req);
  LLMEngineStatus FetchStatus();

 private:
  std::shared_ptr<LLMEnginePluginBase> plugin_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_H_
