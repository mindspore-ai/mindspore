/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_VM_BACKENDBASE_H_
#define MINDSPORE_CCSRC_VM_BACKENDBASE_H_

#include <list>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "backend/common/session/session_basic.h"
#include "runtime/hardware/device_context.h"
#include "backend/graph_compiler/segment_runner.h"
#include "runtime/graph_scheduler/actor/actor_set.h"

namespace mindspore {
namespace compile {
using GraphOutputInfo = session::GraphOutputInfo;
using DeviceContext = device::DeviceContext;
using ActorInfo = runtime::ActorInfo;
using GraphCompiler = runtime::GraphCompiler;
using GraphCompilerInfo = runtime::GraphCompilerInfo;
using ControlNodeParser = runtime::ControlNodeParser;
using FuncGraphToKernelGraphGroup = runtime::FuncGraphToKernelGraphGroup;
using ControlNodeParserPtr = runtime::ControlNodeParserPtr;
using KernelWithIndex = session::KernelWithIndex;

enum SwitchCondStatus {
  kCondOk = 0,
  kCondAlreadyRun,
};

class BACKEND_EXPORT Backend {
 public:
  explicit Backend(const std::string &name);

  virtual ~Backend() = default;

  LinkFuncType convert_fn() { return convert_fn_; }
  std::string name() { return name_; }
  virtual bool GetCond(const BaseRef &c, bool *value);
  virtual bool GetIndex(const BaseRef &c, int64_t *value);
  virtual GraphId CompileGraph(NotNull<FuncGraphPtr> fg) { return kInvalidGraphId; }
  virtual void SetDebugger() {}

  bool is_multi_graph_sink() const { return is_multi_graph_sink_; }
  void set_is_multi_graph_sink(bool flag) { is_multi_graph_sink_ = flag; }

 protected:
  std::string name_;
  LinkFuncType convert_fn_;
  bool is_multi_graph_sink_;
};

void PushInputTensor(const BaseRef &arg, std::vector<tensor::TensorPtr> *inputs, const AnfNodePtr &node = nullptr);
std::vector<std::vector<tensor::TensorPtr>> GetRunGraphInputs(const GraphCompilerInfo &graph_compiler_info,
                                                              const VectorRef &args);

class BACKEND_EXPORT MindRTBackendBase : public Backend {
 public:
  MindRTBackendBase(const std::string &backend_name, const std::string &device_name, uint32_t device_id);
  ~MindRTBackendBase() override = default;

  // The parameter root_graph is a root graph, and the root graph maybe contain multiple sub graphs, It will traverse
  // all sub graphs to call CompileGraph.
  const ActorInfo &CompileGraphs(const FuncGraphPtr &func_graph);

  // Run Graph in the graph mode.
  void RunGraph(const ActorInfo &actor_info, const VectorRef &args, VectorRef *outputs);

#ifdef ENABLE_DEBUGGER
  void SetDebuggerInit() const;
#endif

  // Get the device target.
  std::string GetDeviceTarget() { return device_name_; }

  virtual void WaitTaskFinish() const {}
  virtual void RunGraphByCondition(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                                   const VectorRef &args, VectorRef *outputs) {}

 protected:
  // Convert the nodes which are not supported in the backend.
  void UnifyMindIR(const FuncGraphPtr &func_graph);

  // The parameter func_graph is a graph, it can be either a root graph or a sub graph,
  // The result of graph compiler is stored in graph_id_to_device_context_ and control_nodes_.
  void CompileGraph(const FuncGraphPtr &func_graph, device::RunMode run_mode);

  // Compile the kernel graph by the segment which is from the function graph partition.
  void CompileGraph(const GraphSegmentPtr &segment, device::RunMode run_mode);

  void ConstructOutputs(runtime::ActorSet *actor_set, VectorRef *outputs, const FuncGraphPtr &root_graph);

  // Restore the outputs tuple by the origin funcGraph output node and output tensors.
  void ConstructOutputs(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                        size_t *output_position, VectorRef *outputs, std::vector<tensor::TensorPtr> *tuple_tensors);
  // Spit the tuple tensor to multi tensors for restoring the tuple output.
  void ConstructOutputByTupleTensor(tensor::TensorPtr output_tensor, const abstract::SequenceShapePtr &tensor_shape,
                                    VectorRef *outputs, std::vector<tensor::TensorPtr> *tuple_tensors);
  // In the control flow, the output of the call node needs to be created by abstract.
  BaseRef ConstructOutputByAbstract(const abstract::AbstractBasePtr &abstract,
                                    const std::vector<tensor::TensorPtr> &output_tensors, size_t *output_position,
                                    std::vector<tensor::TensorPtr> *tuple_tensors);
  // Construct the GraphCompilerInfo by the compilation results of graph, used in Graph mode.
  std::shared_ptr<GraphCompilerInfo> ConstructGraphCompilerInfo(const FuncGraphPtr &root_graph);

  void ParseControlNodes(const GraphCompilerInfo &graph_compile_info);

  // When compiling FuncGraph, it is divided according to the control nodes, and obtain the control nodes and several
  // node segments. Node segments will be compiled into kernelGraphs which are expressed as GraphId and bound to
  // the corresponding device_context.
  std::map<GraphId, DeviceContext *> graph_id_to_device_context_;
  // Funcgraph will be cut into multiple kernel graphs, and the map is used to save the correspondence.
  // The kernel graphs which not cut by control flow are placed in the same group.
  std::map<FuncGraphPtr, std::vector<std::vector<GraphId>>> func_graph_to_kernel_graph_ids_;
  std::map<GraphInfo, DeviceContext *> graph_info_to_device_context_;
  std::vector<AnfNodePtr> control_nodes_;

  mindspore::HashMap<ActorInfo, std::shared_ptr<GraphCompilerInfo>> actor_to_graph_compiler_info_;

  // Save the mapping between cell id and actor info.
  FuncGraphPtr root_graph_;
  GraphPartitionPtr graph_partition_;
  std::shared_ptr<GraphCompiler> graph_compiler_;
  std::string device_name_;
  uint32_t device_id_;
  int ms_execution_mode_{kGraphMode};
  void CompileSubGraph(const FuncGraphPtr &func_graph, device::RunMode run_mode = device::RunMode::kUnknown);
  void ProcessNotSupportCnode(const FuncGraphPtr &func_graph, const device::DeviceType &old_target,
                              const device::DeviceType &new_target) const;
};
}  // namespace compile
}  // namespace mindspore
#endif
