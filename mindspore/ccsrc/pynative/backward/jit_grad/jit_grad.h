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
#include "pynative/utils/base.h"
#include "pynative/backward/top_cell.h"
#include "include/frontend/jit/ps/resource_interface.h"
#include "include/utils/visible.h"

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
  py::object GradJit(const py::args &args);
  bool GetJitGradGraph(const pipeline::ResourcePtr &resource, const std::string &phase);
  static void ClearAutoGradCache();

 private:
  void GradJitInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                    const FuncGraphPtr &jit_forward_graph, const FuncGraphPtr &jit_grad_graph);
  // Make CNode for jit forward graph.
  void GetInputArgsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                        AnfNodePtrList *input_nodes) const;
  void GetWeightsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                      const FuncGraphPtr &ms_func_graph, AnfNodePtrList *input_nodes) const;
  void MakeCNodeForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                       const FuncGraphPtr &ms_func_graph, CNodePtr *jit_cnode) const;
  // create grad param for jit fprop graph and connect it with previous op
  GradParamPtr CreateJitGradParam(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                  const FuncGraphPtr &jit_forward_graph, const FuncGraphPtr &jit_grad_graph);
  void RecordForwardGraphForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                const FuncGraphPtr &ms_func_graph) const;
  void Reset();

  // The graph phase is used to obtain backend graph that is complied by jit
  std::string graph_phase_;
  JitCompileInfo compile_info_;
  mindspore::HashMap<std::string, JitCompileInfo> jit_compile_info_{};
};
using JitPtr = std::shared_ptr<Jit>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_JIT_JIT_GRAD_H_
