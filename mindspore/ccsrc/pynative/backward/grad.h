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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_

#include <memory>
#include <string>
#include <utility>
#include <stack>
#include <set>
#include <vector>
#include <map>
#include <unordered_map>
#include "include/utils/pynative/grad_state.h"
#include "pynative/utils/base.h"
#include "pynative/backward/top_cell.h"
#include "pynative/backward/jit_grad/jit_grad.h"
#include "include/runtime/pipeline/pipeline.h"
#include "pynative/backward/op_grad/bprop_task.h"
#include "include/utils/pynative/variable.h"
#include "include/frontend/operator/composite/grad_operation.h"
#include "pynative/backward/hook/custom_function.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
class ForwardExecutor;
using ForwardExecutorPtr = std::shared_ptr<ForwardExecutor>;
using ForwardExecutorWeakPtr = std::weak_ptr<ForwardExecutor>;

class GradExecutor {
  // key: ready run cell id, value: all ready run top cell
  using TopCellIdWithTopCell = std::unordered_multimap<std::string, TopCellInfoPtr>;

 public:
  ~GradExecutor() = default;
  explicit GradExecutor(const ForwardExecutorPtr &forward_executor = nullptr)
      : forward_executor_(ForwardExecutorWeakPtr(forward_executor)), jit_(std::make_shared<Jit>()) {}

  void Init();
  std::function<void(const py::object &, const py::args &)> InitGraph = [this](auto &&PH1, auto &&PH2) {
    NewGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2));
  };
  std::function<void(const py::object &, const py::object &, const py::args &)> LinkGraph = [this](auto &&PH1,
                                                                                                   auto &&PH2,
                                                                                                   auto &&PH3) {
    EndGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3));
  };
  std::function<py::object(const prim::GradOperationPtr &, const py::object &, const py::object &, const py::object &,
                           const py::object &has_aux, const py::args &)>
    Run = [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4, auto &&PH5, auto &&PH6) {
      return RunGrad(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
                     std::forward<decltype(PH3)>(PH3), std::forward<decltype(PH4)>(PH4),
                     std::forward<decltype(PH5)>(PH5), std::forward<decltype(PH6)>(PH6));
    };
  std::function<py::object(const py::object &, const py::object &, const py::args &)> CallCustomBpropFunc =
    [this](auto &&PH1, auto &&PH2, auto &&PH3) {
      return CallCustomBprop(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
                             std::forward<decltype(PH3)>(PH3));
    };
  std::function<py::object(const py::args &)> GradJit = [this](auto &&PH1) {
    return jit()->GradJit(std::forward<decltype(PH1)>(PH1));
  };
  inline TopCellInfoPtr top_cell() const {
    MS_EXCEPTION_IF_NULL(top_cell_);
    return top_cell_;
  }
  inline TopCellInfoPtr TopCellNoCheck() const { return top_cell_; }
  inline JitPtr jit() const {
    MS_EXCEPTION_IF_NULL(jit_);
    return jit_;
  }
  inline void set_top_cell(TopCellInfoPtr top_cell) { top_cell_ = std::move(top_cell); }
  inline void set_is_run_recompute(bool is_run_recompute) { is_run_recompute_ = is_run_recompute; }
  py::object RunGrad(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                     const py::object &grad_position, const py::object &has_aux, const py::args &args);
  py::object RunGradFunc(const autograd::GradAttr &grad_attr, const std::vector<tensor::TensorPtr> &w_args,
                         const std::vector<size_t> &p_args, bool has_aux, bool collect_default_weights);
  CNodePtr ConstructForwardGraph(const OpGradInfoPtr &grad_info, const std::vector<std::string> &input_value_id) const;
  void RecordForwardGraph(const OpGradInfoPtr &grad_info) const;
  void RecordCustomBprop(const autograd::CustomContext &context) const;
  void RecordForwardGraphForInput(const ValuePtr &value, const string &input_id);
  py::object CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                             const py::object &grad_position, const py::args &args, const py::kwargs &kwargs);
  void GetTopCellWithInputArgsRespectTo(const prim::GradOperationPtr &grad, const py::object &obj,
                                        const py::args &args);
  void ProcessOpGradInfo(const OpGradInfoPtr &op_run_info) const;
  py::object CallCustomBprop(const py::object &obj, const py::object out, const py::args &args);
  AnfNodePtr GetInput(const ValuePtr &v, const string &obj_id) const;
  AnfNodePtr GetParamInput(const ValuePtr &v, const std::string &id) const;
  void ClearRes();
  void WorkerJoin();
  void WaitBpropTask() const;
  void SaveDynamicInputsCells(const py::object &obj, const py::args &args);

  inline bool forward_use_dynamic_shape_process() const { return forward_use_dynamic_shape_process_; }
  inline void set_forward_use_dynamic_shape_process(bool forward_use_dynamic_shape_process) {
    forward_use_dynamic_shape_process_ = forward_use_dynamic_shape_process;
  }

  std::string GetReadyRunCellId(const std::string &obj_id, const std::string &input_args_id) const;

  inline bool is_high_order_top_cell() const { return top_cell_ != nullptr && top_cell_->is_high_order_top_cell(); }
  void ChildAfterFork();
  void DispatchGradQueueTask(std::function<void(void)> &&task) const;
  inline bool IsHighOrderTopCell() const {
    return !input_args_info_stack_.empty() && IsNestedGrad() && top_cell()->grad_order() != grad_order_;
  }
  void QueueFinalCallback(std::function<void()> callback) const;

 private:
  ForwardExecutorPtr forward() const;
  inline FuncGraphPtr curr_g() const {
    MS_EXCEPTION_IF_NULL(top_cell()->fg());
    return top_cell()->fg();
  }
  void ClearForwardGraph() { top_cell()->ClearForwardGraph(); }
  inline void PushTopCellStack(const TopCellInfoPtr &top_cell) {
    top_cell_stack_.push(top_cell);
    MS_LOG(DEBUG) << "Push top cell " << top_cell << " on top cell stack";
  }
  bool NeedIncreaseGradOrder(const std::string &obj_id);
  void SaveOutputNodeMap(const std::string &obj_id, const OpGradInfoPtr &grad_info, const CNodePtr &cnode,
                         const std::vector<std::string> &input_value_id) const;
  void DoOpGrad(const OpGradInfoPtr &grad_info) const;
  void SetBpropGraphJitLevel(const py::object &obj) const;
  void ClearGlobalRes() const;
  void ClearGradRes();

  // Higher derivative
  inline bool IsNestedGrad() const { return grad_order_ > 1; }
  inline void IncreaseGradOrder() {
    ++grad_order_;
    MS_LOG(DEBUG) << "Increase grad order, current grad_order is " << grad_order_;
  }
  inline void DecreaseGradOrder() {
    if (grad_order_ > 0) {
      --grad_order_;
    }
    MS_LOG(DEBUG) << "Decrease grad order, current grad_order is " << grad_order_;
  }
  TopCellInfoPtr GetTopCell(const std::string &already_run_cell_id, const std::string &input_args_id);
  TopCellInfoPtr PopTopCellStack();
  void PushInputArgsInfoStack(const InputArgsInfoPtr &input_args_info);
  void PopInputArgsInfoStack();
  void HandleInputArgsForTopCell(const InputArgsInfoPtr &input_args_info);
  void InitResourceAndDfBuilder(const InputArgsInfoPtr &cell_info);
  void MakeNewTopCell(const InputArgsInfoPtr &input_args_info);

  // Manage resource when run grad process.
  void NewGraphInner(const py::object &obj, const py::args &args);
  InputArgsInfoPtr GetInputArgsInfo(const py::object &obj, const py::args &args);
  void EndGraphInner(const py::object &obj, const py::object &out, const py::args &args);
  void EndGraphImpl(const InputArgsInfoPtr &input_args_info);
  void SetForwardLastNodeInfo(const InputArgsInfoPtr &input_args_info) const;
  std::vector<tensor::TensorPtr> GetWeightsArgs(const py::object &weights, bool *weight_param_is_tuple,
                                                bool *collect_default_weights) const;
  std::vector<size_t> GetGradPositionArgs(const py::object &grad_position, bool get_by_position) const;
  // Manage resource for construct forward graph.
  AnfNodePtr GetOutputNodeAsInput(const std::string &obj_id) const;
  AnfNodePtr GetValueSequenceInput(const ValuePtr &v) const;
  AnfNodePtr CreateTupleGetItemNode(const std::string &obj_id,
                                    const std::pair<AnfNodePtr, std::vector<int64_t>> &out) const;
  std::string SizeofContainer() const;

  bool init_{false};
  bool is_run_recompute_{false};
  bool save_graphs_{false};
  bool forward_use_dynamic_shape_process_{false};

  // If grad_order=1, indicate first derivative; grad_order=2, indicate second derivative; ...
  size_t grad_order_{0};
  // if call grad not set_grad first, grad first is true.
  bool call_grad_api_first_{false};

  // Used for auto grad map reserve
  size_t op_num_in_bprop_graph_{kDefaultContainerSize};

  TopCellInfoPtr top_cell_{nullptr};
  InputArgsInfoPtr top_input_args_info_{nullptr};

  // Records every cell info for share, regardless of whether you need construct grad graph
  std::stack<InputArgsInfoPtr> input_args_info_stack_;

  // For top cell nested top cell, import for high-order grad
  std::stack<TopCellInfoPtr> top_cell_stack_;

  // Used for set grad scenario. If top cell set in CheckAlreadyRun, no need find again in RunGrad;
  TopCellInfoPtr finded_top_cell_;
  // Record all top cells that have been run
  TopCellIdWithTopCell ready_run_top_cell_;
  // Record pipeline top cells.

  std::set<std::string> dynamic_inputs_cells_;

  ForwardExecutorWeakPtr forward_executor_;
  JitPtr jit_;
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_
