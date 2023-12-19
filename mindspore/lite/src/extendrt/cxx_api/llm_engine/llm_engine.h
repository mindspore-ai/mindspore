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
  uint64_t req_id = UINT64_MAX;
  uint64_t prompt_length = 0;
  uint64_t prompt_cluster_id = 0;
  uint64_t decoder_cluster_id = 0;
  uint64_t prefix_id = UINT64_MAX;
};

struct LLMIpInfo {
  uint32_t ip;
  uint16_t port;
};

struct LLMClusterInfo {
  uint64_t remote_cluster_id;
  int32_t remote_role_type;
  std::vector<LLMIpInfo> local_ip_infos;
  std::vector<LLMIpInfo> remote_ip_infos;
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
              const std::map<std::string, std::string> &options, const std::string &batch_mode,
              const std::string &postprocess_model_path);
  void Finalize();
  Status Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);
  Status Predict(const std::vector<LLMReq> &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);
  Status CompleteRequest(const LLMReq &req);
  LLMEngineStatus FetchStatus();
  Status PreloadPromptPrefix(const LLMReq &req, const std::vector<MSTensor> &inputs);
  Status ReleasePromptPrefix(const LLMReq &req);
  Status PullKV(const LLMReq &req);
  Status MergeKV(const LLMReq &req, uint32_t batch_index);

  Status LinkClusters(const std::vector<LLMClusterInfo> &clusters, std::vector<Status> *rets, int32_t timeout = -1);
  Status UnlinkClusters(const std::vector<LLMClusterInfo> &clusters, std::vector<Status> *rets, int32_t timeout = -1);

 private:
  std::shared_ptr<LLMEnginePluginBase> plugin_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_H_
