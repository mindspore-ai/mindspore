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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MSBACKENDBASE_H_
#define MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MSBACKENDBASE_H_

#include <memory>
#include <list>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <unordered_set>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "abstract/abstract_function.h"
#include "include/backend/backend_manager/backend_manager.h"
#include "backend/ms_backend/graph_partition.h"
#include "include/backend/backend_manager/backend_jit_config.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "backend/ms_backend/runtime/actors/base/actor_set.h"
#include "tools/profiler/profiler.h"

namespace mindspore {
namespace backend {
namespace ms_backend {
using DeviceContext = device::DeviceContext;
using ActorInfo = runtime::ActorInfo;
using GraphCompiler = runtime::GraphCompiler;
using GraphCompilerInfo = runtime::GraphCompilerInfo;
using ControlNodeParser = runtime::ControlNodeParser;
using FuncGraphToKernelGraphGroup = runtime::FuncGraphToKernelGraphGroup;
using ControlNodeParserPtr = runtime::ControlNodeParserPtr;
using KernelWithIndex = session::KernelWithIndex;
using GraphPartition = compile::GraphPartition;
using GraphPartitionPtr = compile::GraphPartitionPtr;
using mindspore::abstract::AbstractFunction;
using AbstractFunctionPtr = std::shared_ptr<AbstractFunction>;

class MSBackendBase : public BackendBase {
 public:
  MSBackendBase();
  virtual ~MSBackendBase() = default;

  // The backend graph Build interface, the return value is the built graph id.
  std::vector<GraphFragmentPtr> Split(const FuncGraphPtr &func_graph) override;
  // The backend graph Build interface, the return value is the built graph id.
  BackendGraphId Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) override;

  // The backend graph Run interface by the graph_id which are generated through the graph Build interface above.
  RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) override;

  virtual void WaitTaskFinish() const {}
  virtual void RunGraphByCondition(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info,
                                   const VectorRef &args, VectorRef *outputs) {}
#ifdef ENABLE_DEBUGGER
  void SetDebuggerInit() const;
#endif

 protected:
  // Clear the temp members at the end of graph building.
  void ClearGraphBuildMember();

  // Convert the nodes which are not supported in the backend.
  void UnifyMindIR(const FuncGraphPtr &func_graph) const;

  // The parameter func_graph is a graph, it can be either a root graph or a sub graph,
  // The result of graph compiler is stored in graph_id_to_device_context_ and control_nodes_.
  void CompileGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config);

  // Compile the kernel graph by the segment which is from the function graph partition.
  void CompileGraphFromSegment(const FuncGraphPtr &func_graph, const GraphSegmentPtr &segment,
                               const BackendJitConfig &backend_jit_config);

  // Compile the kernel graph which generated directly from front end(PyNative), and no need do graph partition.
  void CompileKernelGraph(const KernelGraphPtr &kernel_graph, const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                          DeviceContext *device_context);

  void CacheFuncGraphWithKernelGraphId(const FuncGraphPtr &func_graph, const GraphId &graph_id,
                                       DeviceContext *device_context);

  std::vector<std::vector<tensor::TensorPtr>> GetRunGraphInputs(const GraphCompilerInfo &graph_compiler_info,
                                                                const VectorRef &args);

  void ConstructOutputs(runtime::ActorSet *actor_set, VectorRef *outputs, const FuncGraphPtr &root_graph,
                        bool enable_graph_pipeline);

  // Restore the outputs tuple by the origin funcGraph output node and output tensors.
  void ConstructOutputs(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                        size_t *output_position, VectorRef *outputs, std::vector<tensor::TensorPtr> *tuple_tensors,
                        const std::vector<TypePtr> &output_types);
  // Spit the tuple tensor to multi tensors for restoring the tuple output.
  void ConstructOutputByTupleTensor(tensor::TensorPtr output_tensor, const abstract::SequenceShapePtr &tensor_shape,
                                    VectorRef *outputs, std::vector<tensor::TensorPtr> *tuple_tensors,
                                    const TypePtr &output_type) const;
  // In the control flow, the output of the call node needs to be created by abstract.
  BaseRef ConstructOutputByAbstract(const abstract::AbstractBasePtr &abstract,
                                    const std::vector<tensor::TensorPtr> &output_tensors, size_t *output_position,
                                    std::vector<tensor::TensorPtr> *tuple_tensors,
                                    const std::vector<TypePtr> &output_types);
  // Construct the GraphCompilerInfo by the compilation results of graph, used in Graph mode.
  std::shared_ptr<GraphCompilerInfo> ConstructGraphCompilerInfo(const FuncGraphPtr &root_graph,
                                                                const BackendJitConfig &backend_jit_config);

  void ParseControlNodes(const GraphCompilerInfo &graph_compile_info);

  void CreateTensorArgs(const VectorRef &args, const GraphCompilerInfo &);

  // Wait multi stream finish.
  void WaitMultiStream(const GraphCompilerInfo &graph_compiler_info);

  // Backend compile cache interface, handle the control node and graph id in this class.
  bool DumpBackendInfo();
  bool LoadBackendInfo(const nlohmann::json &data_json);
  bool CheckBackendInfoValid(nlohmann::json *data_json);

  // Check whether this root_graph can enable single op and graph pipeline or not.
  bool CheckEnableGraphPipeline(const std::shared_ptr<GraphCompilerInfo> &graph_compiler_info);

  // Bind a specific core to the main thread.
  void BindCoreForMainThread();

  void TransformGraphToActorDAG(const GraphCompilerInfo &graph_compiler_info);

  void CompileSubGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config);
  void ProcessNotSupportCnode(const FuncGraphPtr &func_graph, const device::DeviceType &old_target,
                              const device::DeviceType &new_target) const;
  bool CompileGraphsByKbkCache(const FuncGraphPtr &func_graph, DeviceContext *device_context);
  bool CacheCompileGraphs();

  void ClearGraph(BackendGraphId backend_graph_id) override;

  FuncGraphPtr WrapPrimitives(const FuncGraphPtr &graph);

  // The temp members for backend graph building and will be reset at the end of graph building, can't be used in the
  // backend graph running. Do not allow adding new temporary members.
  FuncGraphPtr root_graph_;
  std::vector<AnfNodePtr> control_nodes_;
  // When compiling FuncGraph, it is divided according to the control nodes, and obtain the control nodes and several
  // node segments. Node segments will be compiled into kernelGraphs which are expressed as GraphId and bound to
  // the corresponding device_context.
  std::map<GraphId, DeviceContext *> graph_id_to_device_context_;
  // Funcgraph will be cut into multiple kernel graphs, and the map is used to save the correspondence.
  // The kernel graphs which not cut by control flow are placed in the same group.
  std::map<FuncGraphPtr, std::vector<std::vector<GraphId>>> func_graph_to_kernel_graph_ids_;
  std::map<FuncGraphPtr, std::vector<std::variant<AnfNodePtr, GraphId>>> func_graph_to_sub_segments_;

  // All the backend graphs shared the members and status in the graph building and running. Need clear the object when
  // the graph destroy.
  mindspore::HashMap<BackendGraphId, std::shared_ptr<GraphCompilerInfo>> actor_to_graph_compiler_info_;
  std::shared_ptr<GraphCompiler> graph_compiler_;
  std::string device_name_;
  uint32_t device_id_;
};

using MSBackendBasePtr = std::shared_ptr<MSBackendBase>;
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MSBACKENDBASE_H_
