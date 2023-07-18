/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_MS_FUNCTION_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_MS_FUNCTION_GRAD_H_

#include <vector>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/grad/bprop_tensor_replace.h"
#include "pipeline/jit/pipeline.h"

namespace mindspore {
namespace pynative {
class GradExecutor;
const char kAddedValue[] = "added_value";

struct MsFunctionCompileInfo {
  bool is_control_flow_{false};
  bool is_dynamic_shape_{false};
};

class MsFunction {
 public:
  MsFunction() = default;
  ~MsFunction() = default;
  inline void set_graph_phase(const std::string &graph_phase) { graph_phase_ = graph_phase; }
  FuncGraphPtr ProcessMsFunctionFuncGraph(const FuncGraphPtr &ms_func_graph);
  py::object GradMsFunction(const py::object &out, const py::args &args);
  void SaveForwardOutputTensorInfoInBpropGraph(const FuncGraphPtr &func_graph);

 private:
  void GradMsFunctionInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                           const FuncGraphPtr &primal_func_graph, const FuncGraphPtr &grad_graph,
                           const CNodePtr &added_node, const ValuePtr &added_out_v);
  // Update device address of value node in grad graph by forward tensors.
  void RunReplace(const CNodePtr &added_node, const ValuePtrList &total_output_tensors) const;
  void ReplaceAddedCnodeActualOutput(const CNodePtr &added_node, const ValuePtrList &total_output_tensors) const;
  // Make CNode for ms_function forward graph.
  void GetInputArgsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                        AnfNodePtrList *input_nodes) const;
  void GetWeightsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                      const FuncGraphPtr &ms_func_graph, AnfNodePtrList *input_nodes) const;
  void MakeCNodeForMsFunction(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                              const FuncGraphPtr &ms_func_graph, CNodePtr *ms_function_cnode) const;
  // Make adjoint for ms_function fprop graph and connect it with previous op
  void MakeAdjointForMsFunction(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                const FuncGraphPtr &ms_func_graph, const FuncGraphPtr &grad_graph,
                                bool has_added_v) const;
  void KPynativeWithFProp(const GradExecutor *grad_executor, const autograd::AutoGradCellImplPtr &auto_grad_cell_ptr,
                          const autograd::GradParamPtr &grad_param) const;
  void RecordForwardGraphForMsFunction(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                       const FuncGraphPtr &ms_func_graph) const;
  void UpdateMsFunctionlForwardTensorInfoInBpropGraph(const std::string &op_info, const ValuePtr &v);
  bool IsGraphDynamic(const FuncGraphPtr &func_graph);
  void Reset();
  // The graph phase is used to obtain backend graph that is complied by ms_function
  std::string graph_phase_;
  MsFunctionCompileInfo compile_info_;
  mindspore::HashMap<std::string, TensorReplaceInfo> graph_phase_with_replace_info_;
  mindspore::HashMap<std::string, MsFunctionCompileInfo> ms_function_compile_info_;
};
using MsFunctionPtr = std::shared_ptr<MsFunction>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_MS_FUNCTION_GRAD_H_
