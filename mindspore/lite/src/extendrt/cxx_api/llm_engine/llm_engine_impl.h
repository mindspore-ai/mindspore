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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_IMPL_H_
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/types.h"
#include "include/api/status.h"
#include "extendrt/cxx_api/llm_engine/llm_engine.h"
#include "mindspore/core/base/base.h"
#include "extendrt/cxx_api/llm_engine/llm_engine_plugin.h"

namespace mindspore {
class MS_API LLMEngineImpl {
 public:
  LLMEngineImpl(LLMRole role, uint64_t cluster_id, const std::string &batch_mode = "auto");
  ~LLMEngineImpl() = default;
  Status AddModel(const std::vector<std::string> &model_paths, const std::map<std::string, std::string> &options,
                  const std::string &postprocess_model_path, uint64_t *model_id);
  Status Init(const std::map<std::string, std::string> &options);

  void Finalize();
  Status Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 uint64_t model_id);
  Status Predict(const std::vector<LLMReq> &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 uint64_t model_id);
  Status CompleteRequest(const LLMReq &req);
  Status PreloadPromptPrefix(const LLMReq &req, const std::vector<MSTensor> &inputs, uint64_t model_id);
  Status ReleasePromptPrefix(const LLMReq &req, uint64_t model_id);
  Status PullKV(const LLMReq &req, uint64_t model_id);
  Status MergeKV(const LLMReq &req, uint32_t batch_index, uint32_t batch_id, uint64_t model_id);
  std::vector<LLMTensorInfo> GetInputInfos(uint64_t model_id);

  LLMEngineStatus FetchStatus();
  Status LinkClusters(const std::vector<LLMClusterInfo> &clusters, std::vector<Status> *rets, int32_t timeout = -1);
  Status UnlinkClusters(const std::vector<LLMClusterInfo> &clusters, std::vector<Status> *rets, int32_t timeout = -1);

 private:
  LLMRole role_;
  uint64_t cluster_id_;
  std::string batch_mode_;
  std::map<uint64_t, std::vector<LLMTensorInfo>> model_infos_;
  bool inited_ = false;
  std::shared_ptr<LLMEnginePluginBase> plugin_ = nullptr;

  Status GetModelInfo(const FuncGraphPtr &func_graph, LLMEngineModelInfo *model_info);
  Status LoadAndGetModelInfo(const std::string &model_path, LLMEngineModelInfo *model_info_ptr);
  FuncGraphPtr LoadMindIR(const std::string &model_path);
  Status InitPlugin();
};

typedef LLMEnginePluginBase *(*CreateLLMEnginePluginFunc)(LLMRole, uint64_t, const std::string &);

class LLEnginePluginLoader {
 public:
  static LLEnginePluginLoader &Instance() {
    static LLEnginePluginLoader instance;
    return instance;
  }
  std::shared_ptr<LLMEnginePluginBase> CreatePlugin(LLMRole role, uint64_t cluster_id, const std::string &batch_mode);

 private:
  void *handle_ = nullptr;
  CreateLLMEnginePluginFunc create_plugin_func_ = nullptr;
  bool Register();
};

class LLMModelImpl {
 public:
  LLMModelImpl() = default;
  ~LLMModelImpl() = default;

  void SetModelId(uint64_t model_id) { model_id_ = model_id; }
  uint64_t GetModelId() const { return model_id_; }

  void SetLLMEngine(const std::shared_ptr<LLMEngineImpl> &llm_engine) { engine_impl_ = llm_engine; }
  std::shared_ptr<LLMEngineImpl> GetLLMEngine() const { return engine_impl_; }

 private:
  uint64_t model_id_ = 0;
  std::shared_ptr<LLMEngineImpl> engine_impl_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_LLM_ENGINE_IMPL_H_
