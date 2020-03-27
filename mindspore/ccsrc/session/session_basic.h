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
#ifndef MINDSPORE_CCSRC_SESSION_SESSION_BASIC_H
#define MINDSPORE_CCSRC_SESSION_SESSION_BASIC_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <map>
#include "session/session_context.h"
#include "session/kernel_graph.h"
#include "ir/anf.h"
#include "ir/meta_tensor.h"
#include "utils/any.h"
#include "utils/base_ref.h"
#include "pynative/pynative_execute.h"
#include "device/kernel_info.h"

namespace mindspore {
using GraphId = uint32_t;
using GraphInfo = std::string;
namespace session {
using CallBackFunc = uint32_t (*)(uint32_t graph_id,
                                  const std::map<std::string, mindspore::tensor::TensorPtr> &params_list);
using AnyList = std::vector<Any>;
using AnyListPtr = std::shared_ptr<AnyList>;

using OpRunInfo = pynative::OpExecInfo;
using OpRunInfoPtr = std::shared_ptr<OpRunInfo>;

class SessionBasic {
 public:
  SessionBasic() : device_id_(0) {
    graphs_ = {};
    run_op_graphs_ = {};
    summary_callback_ = nullptr;
  }

  virtual void Init(uint32_t device_id) { device_id_ = device_id; }

  virtual ~SessionBasic() { summary_callback_ = nullptr; }

  virtual GraphId CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) = 0;
  // build graph ,used to handle mupltiple child graphs
  virtual void BuildGraph(GraphId) {}

  virtual void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) = 0;

  virtual void BuildOp(const OpRunInfo &, const GraphInfo &) {}

  virtual py::tuple RunOp(const OpRunInfo &, const GraphInfo &) { return py::tuple(); }

  virtual void RegisterSummaryCallBackFunc(const CallBackFunc &callback);

  std::shared_ptr<KernelGraph> ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs);

  CNodePtr CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph);

  // set parameters of final graph
  virtual GraphId SetFinalGraphInput(const std::vector<AnfNodePtr> &) { return kInvalidGraphId; }
  // set output of final graph
  virtual void SetFinalGraphOutput(const BaseRef &) {}
  // insert switch and set the relative acitve ops
  virtual void SwitchCompile(GraphId, GraphId, GraphId) {}
  // set args of child graph.the arg maybe come from a output of other child graphs,or from final graph's parameter
  virtual void SetChildGraphInput(GraphId, const VectorRef &) {}
  // get graph id in child graphs by ME front anf node pointer
  virtual GraphId GetGraphIdByNode(const AnfNodePtr &) const { return kInvalidGraphId; }
  virtual GraphId GetFinalRunGraph() const { return kInvalidGraphId; }
  virtual void SetActive(GraphId, GraphId) {}

 protected:
  virtual void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                             const std::vector<tensor::TensorPtr> &inputs_const) const;
  void UpdateOutputs(const std::shared_ptr<KernelGraph> &kernel_graph, VectorRef *const outputs,
                     const std::vector<tensor::TensorPtr> &input_tensors) const;
  void Reorder(std::vector<CNodePtr> *node_list);
  void Summary(KernelGraph *graph);
  // create graph output for RunOp
  void CreateOutputNode(const CNodePtr &cnode, const std::shared_ptr<KernelGraph> &graph);
  CNodePtr ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph);
  // create a single run op graph
  std::shared_ptr<KernelGraph> ConstructSingleOpGraph(const OpRunInfo &op_run_info);
  // get tensors from op inputs
  void ToTensorPtr(const OpRunInfo &op_run_info, std::vector<tensor::TensorPtr> *inputs,
                   std::vector<bool> *tensor_mask);
  // trans BaseRef list to py::tuple
  BaseRef TransformBaseRefListToTuple(const BaseRef &base_ref);

  std::unordered_map<GraphId, std::shared_ptr<KernelGraph>> graphs_;
  std::unordered_map<GraphInfo, std::shared_ptr<KernelGraph>> run_op_graphs_;
  std::shared_ptr<Context> context_;
  CallBackFunc summary_callback_;
  static GraphId graph_sum_;
  uint32_t device_id_;
};

using SessionPtr = std::shared_ptr<session::SessionBasic>;
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_SESSION_SESSION_BASIC_H
