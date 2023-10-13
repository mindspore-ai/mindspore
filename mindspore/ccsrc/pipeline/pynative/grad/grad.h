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
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/grad/jit/jit_grad.h"
#include "runtime/pynative/async/async_queue.h"
#include "runtime/pynative/async/async_hqueue.h"
#include "pipeline/pynative/grad/bprop_task.h"
#include "pipeline/pynative/grad/dynamic_shape.h"
#include "pipeline/jit/ps/resource.h"
namespace mindspore {
namespace pynative {
namespace py = pybind11;
class ForwardExecutor;
using ForwardExecutorPtr = std::shared_ptr<ForwardExecutor>;
using ForwardExecutorWeakPtr = std::weak_ptr<ForwardExecutor>;

class GradExecutor {
 public:
  GradExecutor() = default;
  ~GradExecutor() = default;
  explicit GradExecutor(const ForwardExecutorPtr &forward_executor = nullptr)
      : forward_executor_(ForwardExecutorWeakPtr(forward_executor)),
        jit_(std::make_shared<Jit>()),
        dynamic_shape_(std::make_shared<DynamicShape>()),
        bprop_queue_(std::make_shared<AsyncHqueue>("bprop_queue")),
        assist_queue_(std::make_shared<AsyncHqueue>("assist_queue")) {}

  void Init();
  std::function<void(const py::object &, const py::args &)> InitGraph = [this](auto &&PH1, auto &&PH2) {
    NewGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2));
  };
  std::function<void(const py::object &, const py::object &, const py::args &)> LinkGraph = [this](auto &&PH1,
                                                                                                   auto &&PH2,
                                                                                                   auto &&PH3) {
    EndGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3));
  };
  std::function<void(const prim::GradOperationPtr &, const py::object &, const py::object &, const py::object &,
                     const py::args &)>
    GradGraph = [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4, auto &&PH5) {
      GradNetInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3),
                   std::forward<decltype(PH4)>(PH4), std::forward<decltype(PH5)>(PH5));
    };
  std::function<py::object(void)> RunGraph = [this]() { return RunGradGraph(); };
  inline TopCellInfoPtr top_cell() const {
    MS_EXCEPTION_IF_NULL(top_cell_);
    return top_cell_;
  }
  inline DynamicShapePtr dynamic_shape() const {
    MS_EXCEPTION_IF_NULL(dynamic_shape_);
    return dynamic_shape_;
  }
  inline JitPtr jit() const {
    MS_EXCEPTION_IF_NULL(jit_);
    return jit_;
  }
  inline InputArgsInfoPtr top_input_args_info() const {
    MS_EXCEPTION_IF_NULL(top_input_args_info_);
    return top_input_args_info_;
  }

  inline bool TopCellHasNotBeenCreate() const { return top_cell_ == nullptr; }
  inline void set_top_cell(TopCellInfoPtr top_cell) { top_cell_ = std::move(top_cell); }
  inline bool grad_flag() const { return grad_flag_; }
  inline void set_grad_flag(bool flag) { grad_flag_ = flag; }
  inline bool enable_grad() const { return enable_grad_; }
  inline void set_enable_grad(bool enable_grad) { enable_grad_ = enable_grad; }
  inline bool RequiresGrad() const { return enable_grad() && grad_flag(); }
  // Construct grad graph for jit
  inline size_t custom_bprop_cell_count() const { return custom_bprop_cell_count_; }
  inline AsyncHqueuePtr bprop_queue() const { return bprop_queue_; }
  mindspore::OrderedMap<std::string, TopCellInfoPtr> &already_run_top_cell() { return already_run_top_cell_; }
  void SetHookChanged(const py::object &cell) const;
  void GradNetInner(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                    const py::object &grad_position, const py::args &args);
  py::object RunGradGraph();
  CNodePtr ConstructForwardGraph(const FrontendOpRunInfoPtr &op_run_info) const;
  void RecordForwardGraph(const FrontendOpRunInfoPtr &op_run_info) const;
  void RecordForwardGraphForInput(const ValuePtr &value, const string &input_id,
                                  const abstract::AbstractBasePtr &param_abs);
  void RecordNestedGraph(const FuncGraphPtr &first_grad_fg, const GraphInfoPtr &inner_graph_info,
                         const std::vector<ValuePtr> &forward_args, const ValuePtr &out);
  void SaveForwardGraph(const ValuePtr &value, const string &value_id) const;
  void BackupInputTensorGradInfo(const ValuePtr &value);
  void SaveInputTensorGradInfo(const InputArgsInfoPtr &input_args_info);
  void ClearParamGradInfo(const TopCellInfoPtr &top_cell) const;
  void ResumeParamGradInfo(const TopCellInfoPtr &top_cell) const;
  py::object CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                             const py::object &grad_hash_id, const py::args &args);
  TopCellInfoPtr GetAlreadyRunTopCell(const std::string &already_run_cell_id) const;
  void GetPreRunTopCell(const prim::GradOperationPtr &grad, const py::object &obj, const py::args &args);
  void ProcessOpGradInfo(const FrontendOpRunInfoPtr &op_run_info) const;
  AnfNodePtr GetInput(const ValuePtr &v, const string &obj_id) const;
  AnfNodePtr GetParamInput(const ValuePtr &v, const std::string &id) const;
  void UpdateTopCellForwardTensorInfoInBpropGraph(const string &op_info, const ValuePtr &v) const;
  void ClearRes();
  void AsyncClearTopCell();
  void AsyncClearAutoGradCell(const TopCellInfoPtr &top_cell);
  void WorkerJoin();
  void WaitBpropTask() const;
  void SaveDynamicInputsCells(const py::object &obj, const py::args &args);
  void SetTopCellDynamicAttr(const py::object &cell);
  bool use_dynamic_shape_process() const {
    if (top_cell_ == nullptr) {
      return false;
    }
    return top_cell()->use_dynamic_shape_process();
  }

  void set_use_dynamic_shape_process(bool use_dynamic_shape_process) {
    if (top_cell_ == nullptr) {
      return;
    }
    return top_cell()->set_use_dynamic_shape_process(use_dynamic_shape_process);
  }

  inline bool forward_use_dynamic_shape_process() const { return forward_use_dynamic_shape_process_; }
  inline void set_forward_use_dynamic_shape_process(bool forward_use_dynamic_shape_process) {
    forward_use_dynamic_shape_process_ = forward_use_dynamic_shape_process;
  }
  inline bool is_cell_has_dynamic_inputs(const std::string &obj_id) const {
    return dynamic_inputs_cells_.count(obj_id) > 0;
  }
  std::string GetAlreadyRunCellId(const std::string &obj_id) const;
  void DispatchAssistQueueTask(std::function<void(void)> task) const;

  inline bool is_high_order_top_cell() const { return top_cell_ != nullptr && top_cell_->is_high_order_top_cell(); }

 private:
  ForwardExecutorPtr forward() const;
  inline FuncGraphPtr curr_g() const { return top_cell()->fg(); }
  inline void PushHighOrderGraphStack(const TopCellInfoPtr &top_cell) { high_order_stack_.push(top_cell); }
  void SetGradOrder(const std::string &obj_id);
  void SaveOutputNodeMap(const std::string &obj_id, const FrontendOpRunInfoPtr &op_run_info,
                         const CNodePtr &cnode) const;
  void DoOpGrad(const FrontendOpRunInfoPtr &op_run_info) const;
  void DoGraphGrad(const FrontendOpRunInfoPtr &op_run_info) const;
  AnfNodePtr GetRealInputNodeBySkipHook(const AnfNodePtr &input_node) const;
  void SetBpropGraphJitLevel(const py::object &obj) const;
  void ClearGlobalRes() const;
  void ClearGradRes();

  // Higher derivative
  inline bool IsNestedGrad() const { return grad_order_ > 1; }
  inline void IncreaseGradOrder() { ++grad_order_; }
  inline void DecreaseGradOrder() {
    if (grad_order_ > 0) {
      --grad_order_;
    }
  }
  inline bool GetIsHighOrderTopCellFlag() const {
    return !input_args_info_stack_.empty() && IsNestedGrad() && top_cell()->grad_order() != grad_order_;
  }
  uint32_t kernel_graph_id_for_control_flow() { return --kernel_graph_id_for_control_flow_; }
  void ClearPreTopCell(const TopCellInfoPtr &new_top_cell, bool is_need_clear_device_mem);
  bool GetTopCellDynamicFlag(const InputArgsInfoPtr &input_args_info, const std::string &obj_id_with_grad_order);
  void SwitchTopCell();
  TopCellInfoPtr GetTopCell(const std::string &already_run_cell_id);
  void DoParameterReplace(const FuncGraphPtr &first_grad_fg, const GraphInfoPtr &inner_graph_info,
                          const std::vector<ValuePtr> &forward_args, AnfNodePtrList *inputs);
  void MakeNestedCnode(bool has_custom_bprop, const std::vector<ValuePtr> &forward_args,
                       const FuncGraphPtr &cur_run_bprop_graph, const BaseRef &out);
  TopCellInfoPtr PopHighOrderGraphStack();
  void PushInputArgsInfoStack(const InputArgsInfoPtr &input_args_info);
  void PopInputArgsInfoStack();
  void HandleInputArgsForTopCell(const InputArgsInfoPtr &input_args_info, bool is_bprop_top);
  void InitResourceAndDfBuilder(const InputArgsInfoPtr &cell_info);
  void MakeNewTopGraph(const InputArgsInfoPtr &input_args_info);

  // Manage resource when run grad process.
  bool IsBpropGraph(const std::string &cell_id) const;
  void NewGraphInner(const py::object &obj, const py::args &args);
  InputArgsInfoPtr GetInputArgsInfo(const py::object &obj, const py::args &args);
  void EndGraphInner(const py::object &obj, const py::object &out, const py::args &args);
  void EndGraphImpl(const InputArgsInfoPtr &input_args_info);
  void SetForwardLastNodeInfo(const ValuePtr &v) const;
  void GetCustomBpropPrim(const py::object &obj, const py::args &args, const InputArgsInfoPtr &input_args_info);
  void DoGradForCustomBprop(const InputArgsInfoPtr &input_args_info, const std::string &out_id) const;
  void CheckNeedCompileGraph(const InputArgsInfoPtr &input_args_info);
  void GetGradGraph(const autograd::GradAttr &grad_attr, const std::vector<tensor::TensorPtr> &w_args,
                    const std::vector<size_t> &p_args);
  FuncGraphPtr GetBpropGraph(const autograd::GradAttr &grad_attr, const std::vector<tensor::TensorPtr> &w_args,
                             const std::vector<size_t> &p_args);
  std::vector<tensor::TensorPtr> GetWeightsArgs(const py::object &weights, bool *weight_param_is_tuple) const;
  std::vector<tensor::TensorPtr> GetDefaultWeights() const;
  void CheckParamShapeAndType(const ParameterPtr &param_node, const abstract::AbstractBasePtr &input_abs,
                              const abstract::AbstractBasePtr &ir_abs) const;
  void UpdateParamAbsByArgs(const std::vector<ValuePtr> &input_args, const FuncGraphPtr &bprop_graph) const;
  std::vector<size_t> GetGradPositionArgs(const py::object &grad_position, bool get_by_position) const;
  // Manage resource for construct forward graph.
  AnfNodePtr GetOutputNodeAsInput(const std::string &obj_id) const;
  AnfNodePtr GetValueSequenceInput(const ValuePtr &v) const;
  AnfNodePtr CreateTupleGetItemNode(const std::string &obj_id,
                                    const std::pair<AnfNodePtr, std::vector<int64_t>> &out) const;
  void DispatchGradQueueTask(std::function<void(void)> &&task) const;
  void ClearBpropTask() const;
  bool init_{false};
  bool grad_flag_{false};
  bool enable_grad_{true};
  bool grad_is_running_{false};
  bool save_graphs_{false};
  uint32_t kernel_graph_id_for_control_flow_{UINT32_MAX};
  size_t custom_bprop_cell_count_{0};
  size_t obj_order_{0};
  // If grad_order=1, indicate first derivative; grad_order=2, indicate second derivative; ...
  size_t grad_order_{0};
  // Used for auto grad map reserve
  size_t op_num_in_bprop_graph_{kDefaultContainerSize};
  std::string grad_operation_;
  TopCellInfoPtr top_cell_{nullptr};
  InputArgsInfoPtr top_input_args_info_{nullptr};
  // Records every cell info for share, regardless of whether need construct grad graph
  std::stack<InputArgsInfoPtr> input_args_info_stack_;
  // For high grad of bprop
  std::stack<std::pair<std::string, bool>> bprop_grad_stack_;
  std::vector<std::string> bprop_cell_list_;
  // For high grad order
  std::stack<TopCellInfoPtr> high_order_stack_;
  // Record all top cell which has been ran
  mindspore::OrderedMap<std::string, TopCellInfoPtr> already_run_top_cell_;
  ForwardExecutorWeakPtr forward_executor_;
  JitPtr jit_;
  DynamicShapePtr dynamic_shape_{nullptr};
  AsyncHqueuePtr bprop_queue_;
  AsyncHqueuePtr assist_queue_;
  std::set<std::string> dynamic_inputs_cells_;
  std::vector<TopCellInfoPtr> need_gc_top_cell_list_;
  bool forward_use_dynamic_shape_process_{false};
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_
