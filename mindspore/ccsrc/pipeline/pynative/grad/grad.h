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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_

#include <memory>
#include <string>
#include <utility>
#include <stack>
#include <vector>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/dynamic_shape.h"
#include "pipeline/pynative/grad/ms_function_grad.h"

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
  explicit GradExecutor(const ForwardExecutorPtr &forward_executor = nullptr,
                        const DynamicShapePtr &dynamic_shape = nullptr)
      : forward_executor_(ForwardExecutorWeakPtr(forward_executor)),
        dynamic_shape_(DynamicShapeWeakPtr(dynamic_shape)),
        ms_function_(std::make_shared<MsFunction>()) {}

  std::function<void(const py::object *, const py::object &, const py::args &)> InitGraph = [this](auto &&PH1,
                                                                                                   auto &&PH2,
                                                                                                   auto &&PH3) {
    NewGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3));
  };
  std::function<void(const py::object *, const py::object &, const py::object &, const py::args &)> LinkGraph =
    [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
      EndGraphInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
                    std::forward<decltype(PH3)>(PH3), std::forward<decltype(PH4)>(PH4));
    };
  std::function<void(const py::object *, const prim::GradOperationPtr &, const py::object &, const py::object &,
                     const py::object &, const py::args &)>
    GradGraph = [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4, auto &&PH5, auto &&PH6) {
      GradNetInner(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3),
                   std::forward<decltype(PH4)>(PH4), std::forward<decltype(PH5)>(PH5),
                   std::forward<decltype(PH6)>(PH6));
    };
  std::function<void(py::object *, const py::object &, const py::object &sens_param, const py::tuple &)> RunGraph =
    [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
      RunGradGraph(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2), std::forward<decltype(PH3)>(PH3),
                   std::forward<decltype(PH4)>(PH4));
    };

  inline TopCellInfoPtr top_cell() const {
    MS_EXCEPTION_IF_NULL(top_cell_);
    return top_cell_;
  }
  inline MsFunctionPtr ms_function() const {
    MS_EXCEPTION_IF_NULL(ms_function_);
    return ms_function_;
  }
  DynamicShapePtr dynamic_shape() const;
  bool TopCellIsNull() const { return top_cell_ == nullptr; }
  bool need_renormalize() const { return need_renormalize_; }
  bool enable_op_cache() const { return enable_op_cache_; }
  void set_top_cell(TopCellInfoPtr top_cell) { top_cell_ = std::move(top_cell); }
  bool grad_flag() const { return grad_flag_; }
  size_t grad_order() const { return grad_order_; }
  void set_grad_flag(bool flag) { grad_flag_ = flag; }
  bool need_construct_graph() const { return !cell_stack_.empty() && grad_flag_; }
  // Construct grad graph for ms_function
  bool eliminate_forward() const { return eliminate_forward_; }
  void set_eliminate_forward(bool eliminate_forward) { eliminate_forward_ = eliminate_forward; }
  std::vector<TopCellInfoPtr> &top_cell_list() { return top_cell_list_; }
  void SetHookChanged(const py::object &cell) const;
  // Update forward tensors info
  void UpdateTensorInfo(const tensor::TensorPtr &new_tensor, const std::vector<tensor::TensorPtr> &pre_tensors) const;
  void UpdateForwardTensorInfoInBpropGraph(const string &op_info, const ValuePtr &op_out) const;
  void CheckGraph(const py::object &cell, const py::args &args);
  void RunGradGraph(py::object *ret, const py::object &cell, const py::object &sens_param, const py::tuple &args);
  CNodePtr ConstructForwardGraph(const FrontendOpRunInfoPtr &op_run_info) const;
  py::object CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &cell,
                             const py::object &grad_position, const py::args &args);
  std::string GetAlreadyRunCellId(const std::string &cell_id) const;
  void ProcessOpGradInfo(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v) const;
  AnfNodePtr GetInput(const ValuePtr &v) const;
  void ClearGrad(const py::object &cell, const py::args &args);
  void ClearRes();
  void ClearCellRes(const py::object &cell);

 private:
  ForwardExecutorPtr forward() const;
  FuncGraphPtr curr_g() const;
  void CheckNeedCompileGraph();
  void PushHighOrderGraphStack(const TopCellInfoPtr &top_cell);
  size_t GetHighOrderStackSize() const { return high_order_stack_.size(); }
  TopCellInfoPtr GetTopCell(const string &already_run_cell_id);
  std::string GetCurCellOrder() const;
  void EnableOpGraphCache(bool is_enable);
  std::string GetCellId(const py::object &cell, const py::args &args) const;
  void SaveOutputNodeMap(const std::string &obj_id, const ValuePtr &v, const CNodePtr &cnode) const;
  void DoOpGrad(const FrontendOpRunInfoPtr &op_run_info, const CNodePtr &cnode, const ValuePtr &op_out) const;
  void SaveForwardTensorInfoInBpropGraph(const pipeline::ResourcePtr &resource) const;
  AnfNodePtr GetRealInputNodeBySkipHook(const AnfNodePtr &input_node) const;
  void EraseTopCellFromTopCellList(const TopCellInfoPtr &top_cell);
  void SetBpropGraphJitLevel(const py::object &cell) const;

  // grad graph id to identify grad graph cache
  std::string grad_position_;
  std::string grad_weights_id_;

  // Higher derivative
  inline bool IsNestedGrad() const;
  void SwitchTopcell();
  void DoParameterReplace(const FuncGraphPtr &first_grad_fg, const py::tuple &forward_args,
                          std::vector<AnfNodePtr> *inputs, ValuePtrList *weights_args);
  void MakeNestedCnode(const py::object &cell, const py::tuple &forward_args, const pipeline::ResourcePtr &resource,
                       const py::object &out);
  void PushCellStack(const std::string &cell_id);
  void PopCellStack();
  TopCellInfoPtr PopHighOrderGraphStack();
  void HandleInputArgsForTopCell(const py::args &args, bool is_bprop_top) const;
  void InitResourceAndDfBuilder(const std::string &cell_id, const py::object &cell, const py::args &args);
  void MakeNewTopGraph(const string &cell_id, const py::object &cell, const py::args &args, bool is_topest);
  void UpdateTopCellInfo(bool forward_already_run, bool need_compile_graph, bool vm_compiled) const;
  // Manage resource when run grad process.
  bool IsBpropGraph(const std::string &cell_id) const;
  bool IsCellObjIdEq(const std::string &l_cell_id, const std::string &r_cell_id) const;
  void DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph) const;
  void NewGraphInner(const py::object *ret, const py::object &cell, const py::args &args);
  void EndGraphInner(const py::object *ret, const py::object &cell, const py::object &out, const py::args &args);
  void SetForwardLastNodeInfo(const ValuePtr &v, const std::string &obj_id) const;
  void DoGradForCustomBprop(const py::object &cell, const py::args &args, const ValuePtr &out,
                            const std::string &out_id);
  std::string GetGradCellId(bool has_sens, const py::object &cell, const py::args &args) const;
  void GradNetInner(const py::object *ret, const prim::GradOperationPtr &grad, const py::object &cell,
                    const py::object &weights, const py::object &grad_position, const py::args &args);
  FuncGraphPtr GetBpropGraph(const prim::GradOperationPtr &grad, const py::object &cell,
                             const std::vector<AnfNodePtr> &weights, const std::vector<size_t> &grad_position,
                             size_t arg_size, const py::args &args);
  std::vector<AnfNodePtr> GetWeightsArgs(const py::object &weights, const FuncGraphPtr &df_builder) const;
  void CheckParamShapeAndType(const AnfNodePtr &param, const ParameterPtr &param_node,
                              const abstract::AbstractBasePtr &input_abs,
                              const abstract::AbstractBasePtr &param_tensor_abs, const std::string &input_shape);
  void UpdateParamAbsByArgs(const py::list &args, const FuncGraphPtr &bprop_graph);
  std::vector<size_t> GetGradPositionArgs(const py::object &grad_position, const bool get_by_position) const;
  void ShallowCopySensValue(const py::tuple &input_args, bool has_sens, VectorRef *run_args) const;
  // Manage resource for construct forward graph.
  AnfNodePtr GetObjNode(const ValuePtr &v, const std::string &obj_id) const;
  AnfNodePtr MakeValueNode(const ValuePtr &v, const std::string &obj_id) const;
  AnfNodePtr CreateMakeTupleGradNode(const ValuePtr &v, const std::string &obj_id) const;
  AnfNodePtr CreateTupleGetItemNode(const std::string &obj_id) const;
  void RecordGradNodeToGraphInfoMap(const FuncGraphPtr &fg, const CNodePtr &cnode, const ValuePtr &v,
                                    const std::string &obj_id, const ValuePtrList &input_args) const;
  bool ConvertTupleAndScalarIntoTensor(const FrontendOpRunInfoPtr &op_run_info, ValuePtrList *input_args, size_t idx,
                                       const ValuePtr &default_value) const;
  bool grad_flag_{false};
  bool enable_op_cache_{true};
  bool grad_is_running_{false};
  bool need_renormalize_{false};
  bool eliminate_forward_{true};
  bool enable_tuple_to_tensor_{false};
  int custom_bprop_cell_count_{0};
  size_t cell_order_{0};
  size_t grad_order_{0};
  size_t top_cell_switch_counts_{0};

  // The cell run check graph which will be top cell
  std::string check_graph_cell_id_;
  std::string grad_operation_;
  TopCellInfoPtr top_cell_{nullptr};
  // Records forwrad cell, the bottom is top cell
  std::stack<std::string> cell_stack_;
  // For high grad of bprop
  std::stack<std::pair<std::string, bool>> bprop_grad_stack_;
  std::vector<std::string> bprop_cell_list_;
  // For high grad order
  std::stack<TopCellInfoPtr> high_order_stack_;
  // Use vector for keep order
  std::vector<TopCellInfoPtr> top_cell_list_;
  // Record all top cell which has been ran
  mindspore::HashMap<std::string, TopCellInfoPtr> already_run_top_cell_;
  ForwardExecutorWeakPtr forward_executor_;
  DynamicShapeWeakPtr dynamic_shape_;
  MsFunctionPtr ms_function_;
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_H_
