/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GE_BACKEND_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GE_BACKEND_H_

#include <memory>
#include <map>
#include <vector>
#include <string>
#include <set>
#include <unordered_set>
#include "include/backend/backend_manager/backend_manager.h"
#include "include/backend/backend_manager/backend_jit_config.h"
#include "ir/tensor.h"
#include "include/backend/common/kernel_graph/kernel_graph_mgr.h"
#include "abstract/abstract_value.h"
#include "backend/ge_backend/runtime/graph_partition.h"
#include "backend/ge_backend/runtime/graph_compiler.h"
#include "backend/ge_backend/runtime/actor/actor_set.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
enum CompileType { NotSupport = 0, SubGraph = 1, WholeGraph = 2 };
// The base class of all supported backend.
class GEBackend : public BackendBase {
 public:
  GEBackend();
  ~GEBackend() = default;

  // The backend graph Build interface, the return value is the built graph id.
  BackendGraphId Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) override;

  // The backend graph Run interface by the graph_id which are generated through the graph Build interface above.
  RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) override;

  std::string ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file,
                       IRFormat ir_format) override;

  void ConvertIR(const FuncGraphPtr &anf_graph,
                 const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                 IRFormat ir_format) override;

  void Clear() override;

 private:
  // for init
  void Init();
  bool OpenTsd(const std::shared_ptr<MsContext> &ms_context_ptr);
  bool CloseTsd(bool force);

  // for Build
  BackendGraphId CompileWholeGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config);
  BackendGraphId CompileSubGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config);
  // 0: not support; 1: subgraph; 2: whole graph
  CompileType CheckGraph(const FuncGraphPtr &func_graph) const;
  FuncGraphPtr WrapPrimitives(const FuncGraphPtr &graph);
  void TraverseGraphMap(
    const FuncGraphManagerPtr &manager_ptr, FuncGraphTransaction *tr, const FuncGraphSet &fgs,
    const std::function<std::shared_ptr<FuncGraph>(const PrimitivePtr, const abstract::AbstractFunctionPtr)>
      &get_prim_graph);
  std::map<std::string, std::vector<CNodePtr>> CollectCommOps(const FuncGraphPtr &root_graph);
  int GetHcclBuffsizeFromEnv(const std::string &env_name);
  void InitCommGroup(const FuncGraphPtr &root_graph);
  void UnifyMindIR(const FuncGraphPtr &root_graph) const;
  // for compile subgraph
  void CompileGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config);
  void CompileGraphFromSegment(const GraphSegmentPtr &segment, const BackendJitConfig &backend_jit_config);
  std::shared_ptr<mindspore::ge_backend::runtime::GraphCompilerInfo> ConstructGraphCompilerInfo(
    const FuncGraphPtr &root_graph, const BackendJitConfig &backend_jit_config);
  void ParseControlNodes(const mindspore::ge_backend::runtime::GraphCompilerInfo &graph_compile_info,
                         const FuncGraphPtr &root_graph);
  // Clear the temp members at the end of graph building.
  void ClearGraphBuildMember();

  // for run graph
  void RunWholeGraph(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs);
  void RunSubGraph(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs);
  void WaitTaskFinish() const;
  void WaitMultiStream();
  // inputs
  void ConstructInputs(const KernelGraphPtr &func_graph, const VectorRef &args,
                       std::vector<tensor::TensorPtr> *inputs_tensor);
  void ConstructInputsRefMode(const KernelGraphPtr &func_graph, const VectorRef &args,
                              std::vector<tensor::TensorPtr> *inputs_tensor);
  void UpdateInputsShapeAndSize(const ParameterPtr &input_node, const mindspore::kernel::KernelTensorPtr &kernel_tensor,
                                const tensor::TensorPtr &input_tensor);
  void SetTensorUpdateCallback(const tensor::TensorPtr &update_tensor);
  bool Copy(KernelTensor *const dst_kernel_tensor, const tensor::TensorPtr &src_tensor) const;
  bool Copy(KernelTensor *const dst_kernel_tensor, KernelTensor *const src_kernel_tensor) const;
  // outputs
  void ConstructOutputs(const KernelGraphPtr &func_graph, std::vector<tensor::TensorPtr> *outputs,
                        std::vector<TypePtr> *output_types);
  void ConstructOutputs(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                        size_t *output_position, VectorRef *outputs, std::vector<tensor::TensorPtr> *tuple_tensors,
                        const std::vector<TypePtr> &output_types);
  void ConstructOutputs(mindspore::ge_backend::runtime::ActorSet *actor_set, VectorRef *outputs,
                        const FuncGraphPtr &root_graph);
  void ConstructOutputByTupleTensor(tensor::TensorPtr output_tensor, const abstract::SequenceShapePtr &tensor_shape,
                                    VectorRef *outputs, std::vector<tensor::TensorPtr> *tuple_tensors,
                                    const TypePtr &output_type) const;
  BaseRef ConstructOutputByAbstract(const abstract::AbstractBasePtr &abstract,
                                    const std::vector<tensor::TensorPtr> &output_tensors, size_t *output_position,
                                    std::vector<tensor::TensorPtr> *tuple_tensors,
                                    const std::vector<TypePtr> &output_types);

  void ClearGraph(BackendGraphId backend_graph_id) override;

  // for acl dump
  bool DebugOnStepBegin(const KernelGraphPtr &func_graph);
  void DebugOnStepEnd(const KernelGraphPtr &graph, bool dump_flag);

  // for profiling
  bool ProfilerOnStepBegin(const KernelGraphPtr &graph);
  void ProfilerOnStepEnd(bool profile_started);

  // The temp members for backend graph building and will be reset at the end of graph building, can't be used in the
  // backend graph running. Do not allow adding new temporary members.
  std::map<FuncGraphPtr, std::vector<std::vector<GraphId>>> func_graph_to_kernel_graph_ids_;
  std::set<GraphId> graph_ids_;
  std::vector<AnfNodePtr> control_nodes_;

  // All the backend graphs shared the members and status in the graph building and running. Need clear the object
  // when the graph destroy.
  mindspore::HashMap<BackendGraphId, KernelGraphPtr> graph_map_;
  mindspore::HashMap<BackendGraphId, FuncGraphPtr> root_graph_map_;
  // if param init in device, for refmode
  mindspore::HashMap<ParameterPtr, bool> is_weight_init_;
  // if weight value update in python, it records the tensor
  static mindspore::HashSet<const tensor::Tensor *> weights_need_reprepare_;
  // graph running step
  mindspore::HashMap<FuncGraphPtr, uint32_t> graph_run_iter_;
  // <BackendGraphId, compile_type> : comile&run in whole or sub graph
  mindspore::HashMap<BackendGraphId, CompileType> graph_compile_type_;
  mindspore::HashMap<BackendGraphId, std::shared_ptr<mindspore::ge_backend::runtime::GraphCompilerInfo>>
    graph_id_to_graph_compiler_info_;

  std::shared_ptr<mindspore::ge_backend::runtime::GraphCompiler> graph_compiler_;
  static BackendGraphId backend_graph_id_;
  std::shared_ptr<GeGraphExecutor> graph_executor_;
  inline static std::mutex init_mutex_;
  bool is_initialized_ = false;
};

using GEBackendPtr = std::shared_ptr<GEBackend>;
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GE_BACKEND_H_
