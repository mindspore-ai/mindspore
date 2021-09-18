/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_BASIC_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_BASIC_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <map>
#include <set>
#include "backend/session/session_context.h"
#include "backend/session/kernel_graph.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "utils/any.h"
#include "utils/contract.h"
#include "runtime/device/kernel_info.h"
#include "utils/ms_context.h"
#include "runtime/device/bucket.h"
#if defined(ENABLE_DEBUGGER) && !defined(_WIN32) && !defined(_WIN64)
#include "debug/debugger/debugger.h"
#endif
#include "runtime/hardware/device_context.h"
#include "backend/session/pynative_task_manager.h"

namespace mindspore {
namespace runtime {
class GraphCompiler;
}  // namespace runtime
}  // namespace mindspore

namespace mindspore {
using GraphId = uint32_t;
using GraphInfo = std::string;
const char kSessionBasic[] = "SessionBasic";

namespace session {
using CallBackFunc = uint32_t (*)(uint32_t graph_id,
                                  const std::map<std::string, mindspore::tensor::TensorPtr> &params_list);
using AnyList = std::vector<Any>;
using AnyListPtr = std::shared_ptr<AnyList>;

struct OpRunInfo {
  std::string op_name;
  PrimitivePtr primitive;
  AbstractBasePtr abstract;
  bool is_dynamic_shape = false;
  bool is_auto_mixed_precision = false;
  bool lazy_build = false;
  std::string next_op_name = "";
#if defined(__APPLE__)
  int next_input_index = 0;
#else
  size_t next_input_index = 0;
#endif
};

struct InputTensorInfo {
  std::vector<tensor::TensorPtr> input_tensors;
  std::vector<int64_t> input_tensors_mask;
  std::set<KernelWithIndex> input_kernel;
};

struct OutputTensorInfo {
  tensor::TensorPtr output_stub_tensor;
  bool is_weight;
};

struct GraphOutputInfo {
  VectorRef *graph_outputs;
  std::map<KernelWithIndex, std::vector<std::vector<size_t>>> output_indexes;
  std::vector<tensor::TensorPtr> graph_output_tensors;
};

class Executor;

class SessionBasic : public std::enable_shared_from_this<SessionBasic> {
 public:
  SessionBasic() : context_(nullptr), summary_callback_(nullptr), device_id_(0) {
#if defined(ENABLE_DEBUGGER) && !defined(_WIN32) && !defined(_WIN64)
    debugger_ = nullptr;
#endif
  }

  virtual void Init(uint32_t device_id) { device_id_ = device_id; }
  void InitExecutor(const std::string &device_name, uint32_t device_id);
  virtual void SyncStream() const {}
  virtual ~SessionBasic() { summary_callback_ = nullptr; }

  GraphId CompileGraph(const GraphSegmentPtr &segment, const AnfNodePtrList &outputs);
  GraphId CompileGraph(NotNull<FuncGraphPtr> func_graph);
  void BuildGraph(GraphId graphId);
  void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs);
  void RunGraphAsync(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs);
  void RunOp(OpRunInfo *, const GraphInfo &, std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
             const std::vector<int64_t> &tensors_mask);
  void RunOpsInGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs);

#ifndef ENABLE_SECURITY
  virtual void RegisterSummaryCallBackFunc(const CallBackFunc &callback);
#endif

  bool CreateCNodeOfKernelGraph(const AnfNodePtr &node, KernelGraph *graph);

  std::shared_ptr<KernelGraph> ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs,
                                                    bool common_opt = true);
  std::shared_ptr<KernelGraph> ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                    std::vector<KernelGraphPtr> *all_out_graph);

  CNodePtr CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph,
                          std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode);
  CNodePtr CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph);

  // get graph id in child graphs by ME front anf node pointer
  virtual GraphId GetGraphIdByNode(const AnfNodePtr &) const;
  virtual GraphId GetFinalRunGraph() const { return kInvalidGraphId; }
  void AssignParamKey(const KernelGraphPtr &kernel_graph);
  void InitPSParamAndOptim(const KernelGraphPtr &kernel_graph, const std::vector<tensor::TensorPtr> &inputs_const);
  bool IsGetNextGraph(const std::shared_ptr<KernelGraph> &kernel_graph, std::string *channel_name);
  virtual bool CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs,
                                std::string *error_msg) const {
    return true;
  }
  void GetModelInputsInfo(uint32_t graph_id, std::vector<tensor::TensorPtr> *inputs,
                          std::vector<std::string> *inputs_name) const;
  void GetModelOutputsInfo(uint32_t graph_id, std::vector<tensor::TensorPtr> *outputs,
                           std::vector<std::string> *outputs_name) const;
  std::vector<tensor::TensorPtr> GetInputNeedLockTensors(const GraphId &graph_id,
                                                         const std::vector<tensor::TensorPtr> &inputs);
  // Get graph by graph id, if not exist return null ptr
  KernelGraphPtr GetGraph(GraphId graph_id) const;
  void ClearGraph();
  // create a single run op graph
  std::shared_ptr<KernelGraph> ConstructSingleOpGraph(const OpRunInfo &op_run_info,
                                                      const std::vector<tensor::TensorPtr> &input_tensors,
                                                      const std::vector<int64_t> &tensors_mask, bool is_ascend = false);
  void EraseValueNodeTensor(const std::vector<int64_t> &tensors_mask,
                            std::vector<tensor::TensorPtr> *input_tensors) const;
  void RunOpRemoveNopNode(const KernelGraphPtr &kernel_graph) const;
  static void RunOpHideNopNode(const KernelGraphPtr &kernel_graph);
  virtual void ReportWarningMessage() {}
  virtual void ReportErrorMessage() {}
  virtual void SetThreadContext() {}
#ifdef ENABLE_DEBUGGER
  // set debugger
  void SetDebugger() {
    debugger_ = Debugger::GetInstance();
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    MS_EXCEPTION_IF_NULL(debugger_);
    debugger_->Init(device_id_, ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  }
#endif

 private:
  CNodePtr CreateSwitchInput(const CNodePtr &cnode, const AnfNodePtr &node_input, KernelGraph *graph);
  std::vector<AnfNodePtr> CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph);
  std::vector<AnfNodePtr> CreateValueNode(const CNodePtr &cnode, KernelGraph *graph);
  void CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs);
  std::vector<AnfNodePtr> CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph);
  void GetCNodeInfo(const CNodePtr &cnode, std::vector<AnfNodePtr> *cnode_inputs) const;
  void GetNewCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs,
                         std::unordered_map<AnfNodePtr, AnfNodePtr> *other_graph_cnode);
  std::vector<AnfNodePtr> CreateCallSwitchLayerInputs(const CNodePtr &cnode, KernelGraph *graph);
  void ProcessNodeRetFunc(const CNodePtr &cnode, KernelGraph *graph, const std::vector<AnfNodePtr> &real_inputs);
  void HandleInternalOutput(const AnfNodePtr &input_front_node, const AnfNodePtr &backend_node,
                            const FuncGraphManagerPtr &front_func_graph_manager,
                            const std::shared_ptr<KernelGraph> &backend_graph);
  std::string AddPartialParametersMap(const AnfNodePtr &partial_node);
  void GetParameterIndex(const KernelGraph *graph, const std::vector<tensor::TensorPtr> &inputs,
                         std::map<AnfNodePtr, size_t> *parameter_index);
  void CreateOutputPlaceholder(const KernelGraphPtr &kernel_graph, const std::vector<tensor::TensorPtr> &input_tensors,
                               VectorRef *const outputs,
                               std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes);
  void GetRefCount(const KernelGraph *graph, std::map<KernelWithIndex, size_t> *ref_count);
  void HandleOpInputs(const std::set<KernelWithIndex> &input_kernel, std::map<KernelWithIndex, size_t> *ref_count,
                      std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map);

  void HandleOpOutputs(const AnfNodePtr &kernel, const VectorRef &op_outputs,
                       const std::map<KernelWithIndex, size_t> &ref_count,
                       std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map,
                       GraphOutputInfo *const graph_output_info);

 protected:
  friend class Executor;
  friend class CompileNodesTask;
  friend class CompileGraphTask;
  friend class BuildGraphTask;
  friend class RunGraphTask;
  friend class RunOpTask;
  friend class RunOpsInGraphTask;
  friend class mindspore::runtime::GraphCompiler;
  virtual bool IsSupportSummary() { return true; }
  virtual void CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors,
                                   VectorRef *outputs,
                                   std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node);
  // When the device address of the node is used as the output of the graph, the device address will be passed
  // to the output tensor, and the output node will recreate a new device address. This third parameter records
  // the relationship between the new and old device address.
  virtual void UpdateOutputTensors(const VectorRef *outputs,
                                   const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                                   std::map<DeviceAddressPtr, DeviceAddressPtr> *);
  virtual void UnifyMindIR(const KernelGraphPtr &graph);
  virtual void FinalOptimize(const KernelGraphPtr &graph) const;
  virtual GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) { return 0; }
  virtual GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) { return kInvalidGraphId; }
  virtual void BuildGraphImpl(GraphId) {}
  virtual void PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                               const std::vector<tensor::TensorPtr> &inputs, VectorRef *const outputs) {}
  virtual void PostExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                const std::vector<tensor::TensorPtr> &inputs, VectorRef *const outputs) {}
  virtual void ExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph) {}
  void RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs);
  virtual KernelGraphPtr BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                                     const std::vector<tensor::TensorPtr> &input_tensors,
                                     const std::vector<int64_t> &tensors_mask) {
    return nullptr;
  }
  virtual void RunOpImpl(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                         std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                         const std::vector<int64_t> &tensors_mask) {}
  virtual void RunOpImplOrigin(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                               std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                               const std::vector<int64_t> &tensors_mask) {}
  void RunOpsInGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs);
  virtual void BuildOpsInGraph(const GraphId &graph_id, const std::map<AnfNodePtr, size_t> &parameter_index,
                               const std::vector<tensor::TensorPtr> &graph_inputs,
                               const std::map<KernelWithIndex, size_t> &cnode_refcount) {}
#ifndef ENABLE_SECURITY
  virtual void SetSummaryNodes(KernelGraph *graph);
#endif

  void LoadInputs(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs_const) {
    auto kernel_graph = GetGraph(graph_id);
    MS_EXCEPTION_IF_NULL(kernel_graph);
    if (!kernel_graph->executable()) {
      return;
    }
    MS_LOG(INFO) << "Load inputs";
    LoadInputData(kernel_graph, inputs_const);
  }

  virtual void ExecuteAllTaskInQueue() {}

  virtual void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                             const std::vector<tensor::TensorPtr> &inputs_const) const {}
  void UpdateOutputs(const std::shared_ptr<KernelGraph> &kernel_graph, VectorRef *const outputs,
                     const std::vector<tensor::TensorPtr> &input_tensors,
                     std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) const;
  void UpdateOutputAbstract(const std::shared_ptr<KernelGraph> &kernel_graph, OpRunInfo *op_run_info) const;
#ifndef ENABLE_SECURITY
  void Summary(KernelGraph *graph);
#endif
  // create graph output for RunOp
  void CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph);
  CNodePtr ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph);
  // Generate graph info for a single op graph
  GraphInfo GetSingleOpGraphInfo(const CNodePtr &kernel, const std::vector<tensor::TensorPtr> &input_tensors);
  void GetSingleOpRunInfo(const CNodePtr cnode, OpRunInfo *run_info);
  tensor::TensorPtr GetValueNodeOutputTensor(const AnfNodePtr &node, size_t output_index);
  tensor::TensorPtr GetParameterOutputTensor(const AnfNodePtr &node,
                                             const std::map<AnfNodePtr, size_t> &parameter_index,
                                             const std::vector<tensor::TensorPtr> &graph_inputs);
  tensor::TensorPtr GetCNodeOutputTensor(const KernelWithIndex &kernel_with_index,
                                         const std::map<KernelWithIndex, tensor::TensorPtr> &op_output);
  void GetOpInputTensors(const CNodePtr &cnode, const std::map<KernelWithIndex, tensor::TensorPtr> &op_output,
                         const std::map<AnfNodePtr, size_t> &parameter_index,
                         const std::vector<tensor::TensorPtr> &graph_inputs, InputTensorInfo *input_tensor_info);
  tensor::TensorPtr GetOpInputTensorByIndex(const CNodePtr &cnode,
                                            const std::map<KernelWithIndex, tensor::TensorPtr> &op_output,
                                            const std::map<AnfNodePtr, size_t> &parameter_index,
                                            const std::vector<tensor::TensorPtr> &graph_inputs,
                                            InputTensorInfo *const input_tensor_info, size_t input_index);

  // create a new kernel graph and update the graph sum
  KernelGraphPtr NewKernelGraph();
  AnfNodePtr CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph);
  virtual ParameterPtr CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph);
  ValueNodePtr CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph);
  ParameterPtr CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph);
  AnfNodePtr CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph);
  void AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph);
  void InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter);
  AnfNodePtr FindPullNode(const AnfNodePtr &push_node, const std::vector<AnfNodePtr> &node_list);
  void UpdateGraphDynamicShapeAttr(const NotNull<KernelGraphPtr> &root_graph);
  void UpdateAllGraphDynamicShapeAttr(const std::vector<KernelGraphPtr> &all_graphs);
  virtual std::shared_ptr<device::Bucket> CreateBucket(uint32_t bucket_id, uint32_t bucket_size) { return nullptr; }
  void InitAllBucket(const KernelGraphPtr &graph, const device::DeviceContext *device_context = nullptr);
  void AddGradAddrToBucket(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &grad_tensor);
  void ClearAllBucket(const GraphId &graph_id);
  std::vector<uint32_t> GetAllReduceSplitIndex();
  virtual std::string GetCommWorldGroup() { return std::string(); }
  void DumpGraph(const std::shared_ptr<KernelGraph> &kernel_graph);
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  void CheckPSModeConsistence(const KernelGraphPtr &kernel_graph) const;
  void GetBatchElements(const AnfNodePtr &kernel_node) const;
  void InitPsWorker(const KernelGraphPtr &kernel_graph);
#endif

  std::map<uint32_t, std::vector<std::shared_ptr<device::Bucket>>> bucket_map_;
  std::map<uint32_t, uint32_t> free_bucket_id_map_;
  std::unordered_map<GraphId, std::shared_ptr<KernelGraph>> graphs_;
  std::unordered_map<GraphInfo, std::shared_ptr<KernelGraph>> run_op_graphs_;
  std::unordered_map<FuncGraph *, KernelGraphPtr> front_backend_graph_map_;
  std::unordered_map<AnfNodePtr, AnfNodePtr> partial_parameters_map_;
  std::unordered_map<AnfNodePtr, std::string> partial_target_map_;
  std::shared_ptr<Context> context_;
  CallBackFunc summary_callback_;
  static GraphId graph_sum_;
  uint32_t device_id_;
  // rank id of physical device
  uint32_t rank_id_{0};
  std::shared_ptr<Executor> executor_;
#if defined(ENABLE_DEBUGGER) && !defined(_WIN32) && !defined(_WIN64)
  std::shared_ptr<Debugger> debugger_;
#endif
};

using SessionPtr = std::shared_ptr<session::SessionBasic>;
using NamedSummaryOutputs = std::map<std::string, std::pair<AnfNodePtr, int>>;
}  // namespace session
void DumpGraphExeOrder(const std::string &file_name, const std::string &target_dir,
                       const std::vector<CNodePtr> &execution_order);
uint32_t GetRankId();
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_SESSION_BASIC_H
