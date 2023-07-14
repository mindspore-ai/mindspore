/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>

#include "include/api/context.h"
#include "include/model.h"
#include "include/transform/graph_ir/types.h"
#include "extendrt/session/lite_graph_executor.h"
#include "common/config_infos.h"
#include "include/transform/graph_ir/utils.h"
#include "extendrt/delegate/ascend_ge/ge_device_context.h"
#include "src/tensor.h"

namespace mindspore {
class GeAllocator : public Allocator {
 public:
  explicit GeAllocator(const std::function<void(uint8_t *)> &deleter) : deleter_(deleter) {}
  ~GeAllocator() override{};
  void *Malloc(size_t size) override { return nullptr; };
  void Free(void *ptr) override {
    if (ptr != nullptr && deleter_ != nullptr) {
      deleter_(static_cast<uint8_t *>(ptr));
    }
  }
  int RefCount(void *ptr) override { return std::atomic_load(&ref_count_); }
  int SetRefCount(void *ptr, int ref_count) override {
    std::atomic_store(&ref_count_, ref_count);
    return ref_count;
  }
  int DecRefCount(void *ptr, int ref_count) override {
    if (ptr == nullptr) {
      return 0;
    }
    auto ref = std::atomic_fetch_sub(&ref_count_, ref_count);
    return (ref - ref_count);
  }
  int IncRefCount(void *ptr, int ref_count) override {
    auto ref = std::atomic_fetch_add(&ref_count_, ref_count);
    return (ref + ref_count);
  }

 private:
  const std::function<void(uint8_t *)> deleter_ = nullptr;
  std::atomic_int ref_count_ = {0};
};

class GeGraphExecutor : public LiteGraphExecutor {
 public:
  GeGraphExecutor(const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos)
      : context_(context), config_infos_(config_infos) {}
  ~GeGraphExecutor();

  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options,
                    uint32_t *graph_id) override;

  bool RunGraph(uint32_t graph_id, const std::vector<lite::Tensor *> &inputs, std::vector<lite::Tensor *> *outputs,
                const std::map<string, string> &compile_options) override;

  bool Resize(uint32_t graph_id, const std::vector<lite::Tensor *> &inputs,
              const std::vector<ShapeVector> &dims) override {
    return true;
  }

  std::vector<mindspore::lite::Tensor *> GetInputInfos(uint32_t graph_id) override;
  std::vector<mindspore::lite::Tensor *> GetOutputInfos(uint32_t graph_id) override;
  bool Init();
  bool AoeTuning(const FuncGraphPtr &graph);

 private:
  const std::shared_ptr<mindspore::Context> context_;
  ConfigInfos config_infos_;
  std::shared_ptr<ge::Session> ge_session_ = nullptr;
  int64_t session_id_ = -1;
  std::vector<uint32_t> init_graph_id_list_;
  std::vector<uint32_t> compute_graph_id_list_;

  std::shared_ptr<GeDeviceContext> ge_global_context_ = nullptr;

  std::shared_ptr<AscendDeviceInfo> GetAscendDeviceInfo();
  void GetGeGraphOptions(const FuncGraphPtr &anf_graph, std::map<std::string, std::string> *ge_options);
  void GetGeSessionOptions(std::map<std::string, std::string> *ge_options);
  bool CreateSession();
  int64_t GetSessionId();
  transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph);
  bool UpdateInputsOutputs(uint32_t graph_id, const FuncGraphPtr &anf_graph);

  bool AddGraph(const transform::DfGraphPtr &graph, const std::map<std::string, std::string> &options,
                uint32_t *graph_id);
  bool RunGeInitGraph(uint32_t init_graph_id, const std::vector<tensor::TensorPtr> &init_tensors);
  bool GeMoveOutputData(::ge::Tensor *ge_tensor_ptr, lite::Tensor *output, uint32_t graph_id, size_t idx);

  static std::atomic_uint32_t global_graph_idx_;
  static uint32_t GetNextGraphIdx();

  bool is_data_flow_graph_ = false;
  bool RunGeGraphAsync(uint32_t graph_id, const std::vector<::ge::Tensor> &inputs, std::vector<::ge::Tensor> *outputs);
  bool RunDataFlowGraphAsync(uint32_t graph_id, const std::vector<::ge::Tensor> &inputs,
                             std::vector<::ge::Tensor> *outputs);
  bool AoeTuningGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options, uint32_t *graph_id);

  transform::DfGraphPtr CompileGraphCommon(const FuncGraphPtr &graph, const std::map<string, string> &compile_options,
                                           std::map<std::string, std::string> *ge_options_ptr);
  bool UpdateGraphInputs(const FuncGraphPtr &graph);

  std::map<uint32_t, std::vector<lite::Tensor *>> graph_inputs_;
  std::map<uint32_t, std::vector<lite::Tensor *>> graph_outputs_;
  std::map<uint32_t, std::vector<TensorPtr>> original_graph_outputs_;
  std::map<uint32_t, std::vector<TensorPtr>> original_graph_inputs_;
};

struct GeSessionContext {
  std::weak_ptr<ge::Session> ge_session;
  std::map<std::string, std::string> session_options;
  std::set<std::string> session_variables;
};

class GeSessionManager {
 public:
  static std::shared_ptr<ge::Session> CreateGeSession(int64_t session_id,
                                                      const std::map<std::string, std::string> &session_options);
  // return new Variables not in session
  static std::set<std::string> UpdateSessionVariables(int64_t session_id,
                                                      const std::vector<std::string> &graph_variables);
  static void TryReleaseGeSessionContext(int64_t session_id);

 private:
  static std::map<int64_t, GeSessionContext> ge_session_map_;
  static std::mutex session_mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_
