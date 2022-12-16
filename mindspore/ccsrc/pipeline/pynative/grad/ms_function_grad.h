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
#include "pipeline/jit/pipeline.h"

namespace mindspore {
namespace pynative {
class GradExecutor;

class MsFunction {
 public:
  MsFunction() = default;
  ~MsFunction() = default;
  inline void set_graph_phase(const std::string &graph_phase) { graph_phase_ = graph_phase; }
  void ModifyMsFunctionForwardOutput(const FuncGraphPtr &ms_func_graph);
  py::object GradMsFunction(const py::object &out, const py::args &args);

 private:
  void SetMsFuncGraphParameters(const FuncGraphPtr &ms_func_graph);
  void GradMsFunctionInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                           const ValuePtr &added_out_v, const FuncGraphPtr &ms_func_graph,
                           const FuncGraphPtr &grad_graph) const;
  void AsyncGradMsFunctionInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                const ValuePtr &added_out_v, const FuncGraphPtr &ms_func_graph,
                                const FuncGraphPtr &grad_graph) const;
  void AsyncKPynativeWithFProp(const GradExecutor *grad_executor,
                               const autograd::AutoGradCellImplPtr &auto_grad_cell_ptr,
                               const autograd::GradParamPtr &grad_param) const;
  // Update device address of value node in grad graph by forward tensors.
  void RunReplace(const CNodePtr &added_make_tuple, const std::vector<tensor::TensorPtr> &total_output_tensors,
                  const FuncGraphPtr &grad_graph, bool is_dynamic_shape) const;
  void ReplaceWithRealTensorsInGradGraph(const GradExecutor *grad_executor, const ValuePtr &added_out,
                                         const FuncGraphPtr &ms_func_graph, const FuncGraphPtr &grad_graph,
                                         const FrontendOpRunInfoPtr &op_run_info) const;
  void UpdateMsFunctionForwardTensors(const GradExecutor *grad_executor, const TopCellInfoPtr &top_cell,
                                      const string &op_info, const ValuePtr &new_forward_value) const;
  // Make CNode for ms_function forward graph.
  void GetInputArgsNode(const FrontendOpRunInfoPtr &op_run_info, AnfNodePtrList *input_nodes,
                        const GradExecutor *grad_executor) const;
  void GetWeightsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                      const FuncGraphPtr &ms_func_graph, AnfNodePtrList *input_nodes) const;
  void MakeCNodeForMsFunction(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                              const FuncGraphPtr &ms_func_graph, CNodePtr *ms_function_cnode) const;
  // Make adjoint for ms_function fprop graph and connect it with previous op
  CNodePtr MakeAdjointForMsFunction(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                    const FuncGraphPtr &ms_func_graph, const FuncGraphPtr &grad_graph) const;

  bool is_not_support_by_expander_{false};
  // The graph phase is used to obtain backend graph that is complied by ms_function
  std::string graph_phase_;
  // Stores parameter in ms_function
  std::vector<std::string> ms_function_params_;
};
using MsFunctionPtr = std::shared_ptr<MsFunction>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_MS_FUNCTION_GRAD_H_
