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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_JIT_JIT_GRAD_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_JIT_JIT_GRAD_H_

#include <vector>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/grad/auto_grad.h"
#include "pipeline/pynative/grad/ir/bprop_tensor_replace.h"
#include "pipeline/jit/ps/pipeline.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace pynative {
class GradExecutor;
struct JitCompileInfo {
  bool is_control_flow_{false};
  bool is_dynamic_shape_{false};
};

class Jit {
 public:
  Jit() = default;
  ~Jit() = default;
  inline void set_graph_phase(const std::string &graph_phase) { graph_phase_ = graph_phase; }
  py::object GradJit(const py::object &out, const py::args &args);
  void SaveForwardOutputTensorInfoInBpropGraph(const FuncGraphPtr &func_graph);
  void ProcessCnodeFromAdGrad(const CNodePtr &k_app, const CNodePtr &cnode_morph);
  bool GetJitGradGraph(const pipeline::ResourcePtr &resource);
  inline bool eliminate_forward() const { return eliminate_forward_; }
  inline void set_eliminate_forward(bool eliminate_forward) { eliminate_forward_ = eliminate_forward; }
  void Clear();

 private:
  void GradJitInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                    const FuncGraphPtr &primal_func_graph, const FuncGraphPtr &jit_grad_graph,
                    const CNodePtr &added_node, const ValuePtr &added_out_v);
  // Update device address of value node in grad graph by forward tensors.
  void RunReplace(const CNodePtr &added_node, const ValuePtrList &total_output_tensors) const;
  void ReplaceAddedCnodeActualOutput(const CNodePtr &added_node, const ValuePtrList &total_output_tensors) const;
  // Make CNode for jit forward graph.
  void GetInputArgsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                        AnfNodePtrList *input_nodes) const;
  void GetWeightsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                      const FuncGraphPtr &ms_func_graph, AnfNodePtrList *input_nodes) const;
  void MakeCNodeForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                       const FuncGraphPtr &ms_func_graph, CNodePtr *jit_cnode) const;
  // Make adjoint for jit fprop graph and connect it with previous op
  void MakeAdjointForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                         const FuncGraphPtr &jit_forward_graph, const FuncGraphPtr &jit_grad_graph,
                         bool has_added_v) const;
  void KPynativeWithFProp(const GradExecutor *grad_executor, const autograd::AutoGradPtr &auto_grad_cell_ptr,
                          const GradParamPtr &grad_param) const;
  void RecordForwardGraphForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                const FuncGraphPtr &ms_func_graph) const;
  void UpdateJitForwardTensorInfoInBpropGraph(const std::string &op_info, const ValuePtr &v, const size_t &stream_id);
  FuncGraphPtr GetJitForwardGraphCNodeInfo(const FuncGraphPtr &jit_forward_graph);
  void Reset();

  bool eliminate_forward_{true};
  // The graph phase is used to obtain backend graph that is complied by jit
  std::string graph_phase_;
  JitCompileInfo compile_info_;
  mindspore::HashMap<std::string, TensorReplaceInfo> graph_phase_with_replace_info_{};
  mindspore::HashMap<std::string, JitCompileInfo> jit_compile_info_{};
};
using JitPtr = std::shared_ptr<Jit>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_JIT_JIT_GRAD_H_
