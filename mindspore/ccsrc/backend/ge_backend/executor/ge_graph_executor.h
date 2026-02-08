/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_GRAPH_EXECUTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_GRAPH_EXECUTOR_H_
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <set>
#include <unordered_set>
#include "utils/ms_context.h"
#include "backend/ge_backend/graph_ir/types.h"
#include "backend/ge_backend/executor/ge_device_res_manager.h"
#include "backend/ge_backend/executor/ge_summary.h"
#include "backend/ge_backend/executor/ge_memory_manager.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
struct GeInputData {
  std::vector<GeTensor> ge_inputs;
  std::vector<kernel::KernelTensor *> kernel_tensors;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> need_update_input;
};

struct GeOutputData {
  std::vector<GeTensor> ge_outputs;
  std::vector<kernel::KernelTensor *> kernel_tensors;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> graph_outputs;
};

class GeMessageManager {
 public:
  void SetFeatureMemory(const std::string &name, size_t size) { feature_memorys[name] = size; }
  void SetStream(const std::string &name, size_t size) { streams[name] = size; }
  bool SummaryExist(const std::string &name) const {
    auto iter = summarys.find(name);
    if (iter == summarys.end()) {
      return false;
    }
    return true;
  }
  void SetSummary(const std::string &name, const GraphSummary &summary) { summarys[name] = summary; }
  size_t GetFeatureMemory(const std::string &name) const {
    auto iter = feature_memorys.find(name);
    if (iter == feature_memorys.end()) {
      MS_LOG(EXCEPTION) << "Feature memory " << name << " not found.";
    }
    return iter->second;
  }
  size_t GetStream(const std::string &name) const {
    auto iter = streams.find(name);
    if (iter == streams.end()) {
      MS_LOG(EXCEPTION) << "Stream " << name << " not found.";
    }
    return iter->second;
  }
  GraphSummary GetSummary(const std::string &name) const {
    auto iter = summarys.find(name);
    if (iter == summarys.end()) {
      MS_LOG(EXCEPTION) << "Summary " << name << " not found.";
    }
    return iter->second;
  }

 private:
  HashMap<std::string, size_t> feature_memorys;
  HashMap<std::string, size_t> streams;
  HashMap<std::string, GraphSummary> summarys;
};

class GeGraphExecutor {
 public:
  ~GeGraphExecutor() = default;
  void Initialize();
  void Finalize();
  void OptimizeBeforeCompileGraph(const KernelGraphPtr &graph);
  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options);
  bool RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::TensorPtr> &inputs,
                std::vector<tensor::TensorPtr> *outputs, const std::map<string, string> &compile_options);

  FuncGraphPtr BuildDFGraph(const FuncGraphPtr &anf_graph,
                            const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_inputs_map,
                            bool export_air);
  string ExportDFGraph(const std::string &file_name, const FuncGraphPtr &anf_graph, bool is_save_to_file);
  size_t GetGraphFeatureMemory(const FuncGraphPtr &graph) const;
  void InitGraphInfo(const FuncGraphPtr &graph);

  // For run as kernelmod.
  std::vector<std::pair<uint32_t, uint32_t>> GetGraphRefIndexes(const KernelGraphPtr &graph) const;
  void SetGraphWorkspaceMemory(const KernelGraphPtr &graph, void *device_ptr, size_t size);
  size_t GetGraphWorkSpaceMemory(const KernelGraphPtr &graph) const;
  bool CompileGraphForKernel(const KernelGraphPtr &graph);
  bool RunGraphRefModeForKernel(const KernelGraphPtr &graph, const AnfNodeWeakPtr &node,
                                const std::vector<kernel::KernelTensor *> &inputs,
                                const std::vector<kernel::KernelTensor *> &outputs, void *stream);
  void AllocGEFixMemory() const;
  void InitGEFixMemory(const KernelGraphPtr &graph, size_t stream_id) const;
  void FreeGeTensorMemory() {
    input_datas_.clear();
    output_datas_.clear();
  }

  // for ge
  void AllocGEInputOutputMemory(const KernelGraphPtr &graph) const;
  void AllocGERefreshableFeatureMemory(const KernelGraphPtr &graph);
  void FreeGERefreshableFeatureMemory(const KernelGraphPtr &graph);
  void FreeInputOutputMemory(const KernelGraphPtr &graph) const;
  device::DeviceAddressPtr CreateDeviceAddress(const kernel::KernelTensorPtr &kernel_tensor,
                                               bool is_need_alloc_mem) const;
  void AllocInputMemory(const device::DeviceAddressPtr &input_address) const;

  std::unordered_set<std::string> GetInferParameterNames();

 private:
  // for ge_init
  bool initialized_ = false;
  std::shared_ptr<GeAllocator> ge_allocator_;
  void CreateSessionAndGraphRunner() const;

  // for ge_finalize
  bool FinalizeGe();
  void UnregisterExternalAllocator();

  bool RunGraphRefMode(const FuncGraphPtr &graph, const std::vector<tensor::TensorPtr> &inputs);
  bool RunGraphRefModeInnner(const FuncGraphPtr &graph, const std::vector<GeTensor> &inputs,
                             std::vector<GeTensor> *outputs, void *stream);
  void BuildInputDataGeTensor(const KernelGraphPtr &kernel_graph);
  void BuildOutputDataGeTensor(const KernelGraphPtr &kernel_graph);
  bool CompileGraph(const KernelGraphPtr &graph, const std::map<string, string> &compile_options);
  int64_t CurGraphSinkSize(std::string graph_name);
  std::vector<GeTensor> GenerateInputGeTensor(const KernelGraphPtr &kernel_graph) const;
  std::vector<GeTensor> GenerateOutputGeTensor(const KernelGraphPtr &kernel_graph) const;
  // for GEGraphOp Run
  std::vector<GeTensor> CreateInputGeTensorList(const std::vector<kernel::KernelTensor *> &tensorsz,
                                                const KernelGraphPtr &graph);
  std::vector<GeTensor> CreateOutputGeTensorList(const std::vector<kernel::KernelTensor *> &tensorsz,
                                                 const KernelGraphPtr &graph);
  void RunInitGraph(const std::string &graph_name);
  void AddRefCorrespondPairs(const KernelGraphPtr &graph, const std::vector<std::pair<uint32_t, uint32_t>> &io_indexes);
  bool BuildGraph(const KernelGraphPtr &graph, const backend::ge_backend::TensorOrderMap &tensor_order_map);
  void DoAsyncCkpt(const FuncGraphPtr &graph);
  void SetFlagIgnoreDevicePtr(const FuncGraphPtr &graph);
  mindspore::HashMap<session::KernelGraph *, GeInputData> input_datas_;
  mindspore::HashMap<session::KernelGraph *, GeOutputData> output_datas_;
  // io_index for kernel_graph
  mindspore::HashMap<std::string, std::vector<std::pair<uint32_t, uint32_t>>> io_indexes_;
  std::map<std::string, int64_t> graph_sink_size_;
  int64_t pre_sink_size_{-1};
  bool disable_ge_kernel_ = IsDisableGeKernel();
  GeMessageManager ge_message_manager_;
  mindspore::HashMap<FuncGraphPtr, bool> is_init_graph_run_;
  // <graph_ptr, refreshable_feature_mem_ptr>
  std::map<KernelGraphPtr, void *> graph_refreshable_feature_mem_;
  std::shared_ptr<GeDeviceResManager> ge_res_manager_;
};
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_GRAPH_EXECUTOR_H_
