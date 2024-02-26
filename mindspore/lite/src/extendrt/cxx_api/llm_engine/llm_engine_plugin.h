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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_PLUGIN_H_
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/serialization.h"
#include "extendrt/cxx_api/graph/graph_data.h"
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
#include "extendrt/infer_session.h"
#include "src/common/config_infos.h"
#include "mindapi/ir/common.h"
#include "extendrt/cxx_api/llm_engine/llm_engine.h"

namespace mindspore {
struct LLMEngineModelInfo {
  std::string name;
  tensor::TensorPtr om_data = nullptr;
  std::vector<std::string> input_names;
  std::vector<ShapeVector> input_shapes;
  std::vector<TypeId> input_dtypes;
  std::vector<ShapeVector> ref_input_shapes;
  std::vector<TypeId> ref_input_dtypes;
  size_t output_count = 0;
  std::string weight_dir;
};

class LLMEnginePluginBase {
 public:
  LLMEnginePluginBase(LLMRole role, uint64_t cluster_id, const std::string &batch_mode)
      : role_(role), cluster_id_(cluster_id), batch_mode_(batch_mode) {}
  virtual ~LLMEnginePluginBase() = default;

  virtual Status AddModel(const std::vector<LLMEngineModelInfo> &model_infos,
                          const std::map<std::string, std::string> &options,
                          const LLMEngineModelInfo &postprocess_model, uint64_t *model_id) = 0;
  virtual Status Init(const std::map<std::string, std::string> &options) = 0;
  virtual void Finalize() = 0;
  virtual Status Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                         uint64_t model_id) = 0;
  virtual Status Predict(const std::vector<LLMReq> &req, const std::vector<MSTensor> &inputs,
                         std::vector<MSTensor> *outputs, uint64_t model_id) = 0;
  virtual Status CompleteRequest(const LLMReq &req) = 0;
  virtual Status PreloadPromptPrefix(const LLMReq &req, const std::vector<MSTensor> &inputs, uint64_t model_id) = 0;
  virtual Status ReleasePromptPrefix(const LLMReq &req, uint64_t model_id) = 0;
  virtual Status PullKV(const LLMReq &req, uint64_t model_id) = 0;
  virtual Status MergeKV(const LLMReq &req, uint32_t batch_index, uint32_t batch_id, uint64_t model_id) = 0;

  virtual LLMEngineStatus FetchStatus() = 0;
  virtual Status LinkClusters(const std::vector<LLMClusterInfo> &, std::vector<Status> *rets, int32_t timeout) = 0;
  virtual Status UnlinkClusters(const std::vector<LLMClusterInfo> &, std::vector<Status> *rets, int32_t timeout) = 0;

 protected:
  LLMRole role_ = kLLMRolePrompt;
  uint64_t cluster_id_ = 0;
  std::string batch_mode_;
};

extern "C" MS_API LLMEnginePluginBase *CreateLLMEnginePlugin(LLMRole, uint64_t, const std::string &);
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_PLUGIN_H_
