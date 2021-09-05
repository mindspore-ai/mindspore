/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <map>
#include <utility>
#include <vector>

#include "utils/contract.h"
#include "ir/anf.h"
#include "vm/segment_runner.h"
#include "vm/graph_partition.h"
#include "vm/vm.h"
#include "backend/session/session_basic.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/graph_scheduler.h"

namespace mindspore {
namespace compile {
using OpRunInfo = session::OpRunInfo;
using GraphOutputInfo = session::GraphOutputInfo;
using DeviceContext = device::DeviceContext;
using ActorInfo = runtime::ActorInfo;
using GraphCompiler = runtime::GraphCompiler;
using GraphCompilerInfo = runtime::GraphCompilerInfo;
using ControlNodeParser = runtime::ControlNodeParser;
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
  std::unordered_map<GraphId, LinConvertResult> graph_id_map_;
};

class MindRTBackend : public Backend {
 public:
  MindRTBackend(const std::string &backend_name, const std::string &device_name, uint32_t device_id);
  ~MindRTBackend() override = default;

  // The parameter root_graph is a root graph, and the root graph maybe contain multiple sub graphs, It will traverse
  // all sub graphs to call CompileGraph.
  const ActorInfo &CompileGraphs(const FuncGraphPtr &root_graph);

  // Compile single op kernel graph in the pyNative mode.
  const ActorInfo &CompileGraph(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                                const std::vector<int64_t> *tensors_mask,
                                std::vector<tensor::TensorPtr> *input_tensors);

  // Run Graph in the graph mode.
  void RunGraph(const ActorInfo &actor_info, const VectorRef &args, VectorRef *outputs);

  // Run Graph in the pyNative mode.
  void RunGraph(const ActorInfo &actor_info, OpRunInfo *op_run_info, const std::vector<int64_t> *tensors_mask,
                const std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs);
#ifdef ENABLE_DEBUGGER
  void SetDebuggerInit();
#endif

 private:
  // The parameter func_graph is a graph, it can be either a root graph or a sub graph,
  // The result of graph compiler is stored in graph_id_to_device_context_ and control_nodes_.
  void CompileGraph(const FuncGraphPtr &func_graph);

  // Restore the outputs tuple by the origin funcGraph output node and output tensors.
  void ConstructOutputs(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                        size_t *output_position, VectorRef *outputs);

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

  // Split complete kernel graph to single op graph in PyNative back
  // propagation, then compile and run single op graph.
  void RunGraphBySingleOp(const std::vector<KernelGraphPtr> &graphs,
                          const std::vector<std::vector<tensor::TensorPtr>> &inputs, VectorRef *outputs);

  // When compiling FuncGraph, it is divided according to the control nodes, and obtain the control nodes and several
  // node segments. Node segments will be compiled into kernelGraphs which are expressed as GraphId and bound to
  // the corresponding device_context.
  std::map<GraphId, DeviceContext *> graph_id_to_device_context_;
  std::map<GraphInfo, DeviceContext *> graph_info_to_device_context_;
  std::vector<AnfNodePtr> control_nodes_;

  std::unordered_map<ActorInfo, std::unique_ptr<GraphCompilerInfo>> actor_to_graph_compiler_info_;

  // Cache output tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<GraphId, std::map<KernelWithIndex, size_t>> cnode_ref_counts_;

  FuncGraph *root_graph_;
  GraphPartitionPtr graph_partition_;
  std::shared_ptr<GraphCompiler> graph_compiler_;
  std::string device_name_;
  uint32_t device_id_;
  int ms_execution_mode_{kGraphMode};
  int real_execution_mode_{kGraphMode};
};
}  // namespace compile
}  // namespace mindspore
#endif
