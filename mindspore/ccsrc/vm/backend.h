/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_VM_BACKEND_H_
#define MINDSPORE_CCSRC_VM_BACKEND_H_

#include <list>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "utils/contract.h"
#include "ir/anf.h"
#include "vm/segment_runner.h"
#include "vm/graph_partition.h"
#include "vm/vm.h"
#include "backend/session/session_basic.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/graph_scheduler.h"
#include "runtime/op_builder/op_lazy_builder.h"

namespace mindspore {
namespace compile {
using OpRunInfo = session::OpRunInfo;
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

class Backend {
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

class MsBackend : public Backend {
 public:
  MsBackend(const std::string &name, const std::string &target, uint32_t device_id);
  ~MsBackend() override = default;

  LinConvertResult MsConvert(const GraphSegmentPtr &segment, const std::string &target = "");
  virtual VectorRef MsRunGraph(const GraphId &g, const VectorRef &args, const std::string &target = "");

  VectorRef MsSimuRunGraph(const GraphId &g);
  GraphId CompileGraph(NotNull<FuncGraphPtr> fg) override;
  VectorRef RunGraph(GraphId graph_id, const VectorRef &args);
  void ClearSessionGraphs();
  void CreateOtherSession(const std::string &target);

#ifdef ENABLE_DEBUGGER
  void SetDebugger() override;
#endif

 protected:
  session::SessionPtr target_sess_;
  session::SessionPtr other_sess_;
  std::string target_device_;
  std::string other_device_;
  mindspore::HashMap<GraphId, LinConvertResult> graph_id_map_;
};

class MindRTBackend : public Backend {
 public:
  MindRTBackend(const std::string &backend_name, const std::string &device_name, uint32_t device_id);
  ~MindRTBackend() override = default;

  // The parameter root_graph is a root graph, and the root graph maybe contain multiple sub graphs, It will traverse
  // all sub graphs to call CompileGraph.
  const ActorInfo &CompileGraphs(const FuncGraphPtr &root_graph);

  // Run Graph in the graph mode.
  void RunGraph(const ActorInfo &actor_info, const VectorRef &args, VectorRef *outputs);
  // Run single op in the PyNative mode.
  void RunOp(OpRunInfo *op_run_info, VectorRef *outputs);
#ifdef ENABLE_DEBUGGER
  void SetDebuggerInit();
#endif

  // Execute all tasks in queue when lazy build is enabled in PyNative mode.
  void SyncLazyTasks() const;
  // Clear resource when python exit.
  void ClearOpBuilderResource() const;
  // Get the device target.
  std::string GetDeviceTarget() { return device_name_; }
  // Sync default stream in PyNative mode.
  void SyncStream();

 private:
  // The parameter func_graph is a graph, it can be either a root graph or a sub graph,
  // The result of graph compiler is stored in graph_id_to_device_context_ and control_nodes_.
  // The return value indicates whether the subgraph needs to be compiled recursively.
  bool CompileGraph(const FuncGraphPtr &func_graph);

  // Compile the kernel graph by the segment which is from the function graph partition.
  void CompileGraph(const GraphSegmentPtr &segment);

  // CreateKernel, Transform and Schedule have not been finished when LazyBuild is enabled in PyNative mode.
  void CompileSingleOpGraph(const KernelGraphPtr &graph, const DeviceContext *device_context,
                            GraphCompilerInfo *graph_compiler_info) const;

  // Get saved OpBuildTask in OpLazyBuilder and build all the kernels together in PyNative mode.
  void CompileSingleOpGraphs(const std::vector<std::shared_ptr<runtime::OpTask>> &build_tasks);

  // Restore the outputs tuple by the origin funcGraph output node and output tensors.
  void ConstructOutputs(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                        size_t *output_position, VectorRef *outputs);
  // In the control flow, the output of the call node needs to be created by abstract.
  BaseRef ConstructOutputByAbstract(const abstract::AbstractBasePtr &abstract,
                                    const std::vector<tensor::TensorPtr> &output_tensors, size_t *output_position);
  // Construct the GraphCompilerInfo by the compilation results of graph, used in Graph mode.
  std::unique_ptr<GraphCompilerInfo> ConstructGraphCompilerInfo(const FuncGraphPtr &root_graph);

  // Construct the GraphCompilerInfo by the compilation results of graph, used in PyNative mode.
  std::unique_ptr<GraphCompilerInfo> ConstructGraphCompilerInfo(const ActorInfo &actor_info,
                                                                const std::vector<int64_t> *tensors_mask,
                                                                const std::vector<tensor::TensorPtr> *input_tensors,
                                                                bool need_erase);

  // In PyNative mode, the size of single op cache list will be increasing, which lead to memory cost increasing,
  // so the latest single op cache should be erased when cache list size exceeds threshold value.
  void EraseSingleOpCache(const ActorInfo &actor_info, const KernelGraphPtr &graph);

  // Run op immediately when the single_op_cache hit and the queue of OpLazyBuilder is empty in PyNative mode.
  void RunSingleOpGraph(const KernelGraphPtr &graph, const OpRunInfo &op_run_info,
                        const GraphCompilerInfo *graph_compiler_info);

  // Execute OpBuildTask and OpRunTask when the OpLazyBuilder queue is full in PyNative mode.
  void LazyExecuteTaskCallback();

  // Run op immediately or save OpBuildTask and OpRunTask in OpLazyBuilder.
  void RunOpInternal(bool single_op_cache_hit, GraphCompilerInfo *graph_compiler_info, OpRunInfo *op_run_info,
                     VectorRef *outputs);

  // Split complete kernel graph to single op graph in PyNative back
  // propagation, then compile and run single op graph.
  void RunGraphBySingleOp(const std::vector<KernelGraphPtr> &graphs,
                          const std::vector<std::vector<tensor::TensorPtr>> &inputs, VectorRef *outputs);

  // When compiling FuncGraph, it is divided according to the control nodes, and obtain the control nodes and several
  // node segments. Node segments will be compiled into kernelGraphs which are expressed as GraphId and bound to
  // the corresponding device_context.
  std::map<GraphId, DeviceContext *> graph_id_to_device_context_;
  // Funcgraph will be cut into multiple kernel graphs, and the map is used to save the correspondence.
  // The kernel graphs which not cut by control flow are placed in the same group.
  std::map<FuncGraphPtr, std::vector<std::vector<GraphId>>> func_graph_to_kernel_graph_ids_;
  std::map<GraphInfo, DeviceContext *> graph_info_to_device_context_;
  std::vector<AnfNodePtr> control_nodes_;

  mindspore::HashMap<ActorInfo, std::unique_ptr<GraphCompilerInfo>> actor_to_graph_compiler_info_;

  // Cache output tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<GraphId, std::map<KernelWithIndex, size_t>> cnode_ref_counts_;

  // Cache forward op output value node tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<std::string, size_t> forward_op_output_tensor_id_;

  FuncGraph *root_graph_;
  GraphPartitionPtr graph_partition_;
  std::shared_ptr<GraphCompiler> graph_compiler_;
  std::string device_name_;
  uint32_t device_id_;
  int ms_execution_mode_{kGraphMode};
  int real_execution_mode_{kGraphMode};
};
using MindRTBackendPtr = std::shared_ptr<compile::MindRTBackend>;
}  // namespace compile
}  // namespace mindspore
#endif
