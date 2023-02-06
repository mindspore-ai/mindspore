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

#include "pipeline/pynative/grad/grad.h"
#include <algorithm>
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/forward/forward.h"
#include "pipeline/jit/pipeline.h"
#include "ir/cell.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/parse/parse_dynamic.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/debug/trace.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "backend/common/optimizer/helper.h"
#include "include/common/utils/convert_utils_py.h"
#include "frontend/optimizer/ad/grad.h"
#include "pipeline/jit/pass.h"

namespace mindspore {
namespace pynative {
const size_t MAX_TOP_CELL_COUNTS = 40;
const char kGrad[] = "grad";

ForwardExecutorPtr GradExecutor::forward() const {
  auto forward_executor = forward_executor_.lock();
  MS_EXCEPTION_IF_NULL(forward_executor);
  return forward_executor;
}

DynamicShapePtr GradExecutor::dynamic_shape() const {
  auto dynamic_shape = dynamic_shape_.lock();
  MS_EXCEPTION_IF_NULL(dynamic_shape);
  return dynamic_shape;
}

FuncGraphPtr GradExecutor::curr_g() const { return top_cell()->fg(); }

void GradExecutor::PushCellStack(const std::string &cell_id) {
  cell_stack_.push(cell_id);
  ++cell_order_;
}

void GradExecutor::PopCellStack() {
  if (cell_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack cell_stack_ is empty";
  }
  cell_stack_.pop();
}

std::string GradExecutor::GetCurCellOrder() const {
  if (cell_stack_.empty()) {
    MS_LOG(EXCEPTION) << "The cell_stack_ is empty!";
  }
  return cell_stack_.top() + "_" + std::to_string(cell_order_);
}

void GradExecutor::PushHighOrderGraphStack(const TopCellInfoPtr &top_cell) { high_order_stack_.push(top_cell); }

TopCellInfoPtr GradExecutor::PopHighOrderGraphStack() {
  if (high_order_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack high_order_stack_ is empty";
  }
  high_order_stack_.pop();
  TopCellInfoPtr top_cell = nullptr;
  if (!high_order_stack_.empty()) {
    top_cell = high_order_stack_.top();
  }
  return top_cell;
}

std::string GradExecutor::GetFnInfoByPyObj(const py::object &obj) const {
  std::string fn_info = obj.attr("__module__").cast<std::string>();
  fn_info += "_" + obj.attr("__name__").cast<std::string>();
  fn_info += "_" + obj.attr("__code__").attr("co_filename").cast<std::string>();
  fn_info += "_" + py::str(obj.attr("__code__").attr("co_firstlineno")).cast<std::string>();
  if (py::hasattr(obj, "__warpped__")) {
    auto warpped_obj = obj.attr("__warpped__");
    fn_info += "_" + warpped_obj.attr("__name__").cast<std::string>();
    fn_info += "_" + warpped_obj.attr("__code__").attr("co_filename").cast<std::string>();
    fn_info += "_" + py::str(warpped_obj.attr("__code__").attr("co_firstlineno")).cast<std::string>();
  }
  return fn_info;
}

std::string GradExecutor::GetCellId(const py::object &cell, const py::args &args) const {
  std::string cell_id;
  if (!py::isinstance<Cell>(cell)) {
    cell_id = GetFnInfoByPyObj(cell);
  } else {
    cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  }
  auto fn = [&cell_id](const abstract::AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->BuildShape();
    auto type = abs->BuildType();
    cell_id += "_" + shape->ToString();
    cell_id += type->ToString();
  };

  for (size_t i = 0; i < args.size(); i++) {
    const auto &arg_id = PyNativeAlgo::PyParser::GetIdByPyObj(args[i]);
    // Get dynamic input, like data sink
    const auto item = dynamic_shape()->id_with_dynamic_abs().find(arg_id);
    if (item != dynamic_shape()->id_with_dynamic_abs().end()) {
      MS_LOG(DEBUG) << "Input " << i << " get dynamic input";
      fn(item->second);
      continue;
    }

    // Find in step process
    const auto &node_abs_map = forward()->NodeAbsMap();
    auto it = node_abs_map.find(arg_id);
    if (it != node_abs_map.end()) {
      fn(it->second);
    } else {
      auto value = PyNativeAlgo::DataConvert::PyObjToValue(args[i]);
      MS_EXCEPTION_IF_NULL(value);
      auto abs = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(abs);
      if (abs->isa<abstract::AbstractTensor>()) {
        abs->set_value(kAnyValue);
      }
      forward()->SetNodeAbsMapById(arg_id, abs);
      fn(abs);
    }
  }
  return cell_id;
}

void GradExecutor::DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph) const {
#ifdef ENABLE_DUMP_IR
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    DumpIR(filename, graph);
  }
#endif
}

inline bool GradExecutor::IsNestedGrad() const {
  MS_LOG(DEBUG) << "Grad nested order is " << grad_order_;
  return grad_order_ > 1;
}

bool GradExecutor::IsCellObjIdEq(const std::string &l_cell_id, const std::string &r_cell_id) const {
  // just compare obj_id, ignore args id
  auto l_index = l_cell_id.find('_');
  auto r_index = r_cell_id.find('_');
  return l_cell_id.substr(0, l_index) == r_cell_id.substr(0, r_index);
}

bool GradExecutor::IsBpropGraph(const std::string &cell_id) const {
  if (top_cell_ == nullptr) {
    return false;
  }
  return std::any_of(bprop_cell_list_.begin(), bprop_cell_list_.end(),
                     [&cell_id](const std::string &value) { return cell_id.find(value) != std::string::npos; });
}

void GradExecutor::UpdateTopCellInfo(bool forward_already_run, bool need_compile_graph, bool vm_compiled) const {
  top_cell()->set_vm_compiled(vm_compiled);
  top_cell()->set_need_compile_graph(need_compile_graph);
  top_cell()->set_forward_already_run(forward_already_run);
}

void GradExecutor::ClearCellRes(const py::object &cell) {
  static bool clear_all_cell_res = false;
  // Grad clean
  if (py::isinstance<py::none>(cell)) {
    MS_LOG(DEBUG) << "Clear all cell resources";
    clear_all_cell_res = true;
    for (const auto &iter : top_cell_list_) {
      MS_EXCEPTION_IF_NULL(iter);
      iter->Clear();
    }
    top_cell_list_.clear();
    already_run_top_cell_.clear();
    clear_all_cell_res = false;
    return;
  }
  if (clear_all_cell_res) {
    MS_LOG(DEBUG) << "In process of clearing all cell resources, so no need to clear single cell resource again";
    return;
  }
  const auto &cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  MS_LOG(DEBUG) << "Clear cell res, cell id " << cell_id;
  // clear when cell destruction
  for (auto it = top_cell_list_.begin(); it != top_cell_list_.end();) {
    MS_EXCEPTION_IF_NULL(*it);
    const auto &top_cell_id = (*it)->cell_id();
    const auto &already_run_cell_id = (*it)->already_run_cell_id();
    if (IsCellObjIdEq(cell_id, top_cell_id)) {
      MS_LOG(DEBUG) << "Clear top cell resource. Top cell id " << top_cell_id;
      (*it)->Clear();
      (void)already_run_top_cell_.erase(already_run_cell_id);
      it = top_cell_list_.erase(it);
      continue;
    }
    ++it;
  }
}

void GradExecutor::HandleInputArgsForTopCell(const py::args &args, bool is_bprop_top) const {
  if (is_bprop_top) {
    // Convert input args to parameters for top cell graph in bprop.
    for (size_t i = 0; i < args.size(); ++i) {
      auto param = args[i];
      auto new_param = curr_g()->add_parameter();
      const auto &param_id = PyNativeAlgo::PyParser::GetIdByPyObj(param);
      top_cell()->SetTupleArgsToGraphInfoMap(curr_g(), PyNativeAlgo::DataConvert::PyObjToValue(param), new_param, true);
      top_cell()->SetNodeMapInGraphInfoMap(curr_g(), param_id, new_param);
      top_cell()->SetParamNodeMapInGraphInfoMap(curr_g(), param_id, new_param);
    }
    return;
  }
  // Convert input args to parameters for top cell graph in construct.
  std::vector<ValuePtr> input_param_values;
  const auto &only_tensors = PyNativeAlgo::PyParser::FilterTensorArgs(args);
  for (size_t i = 0; i < only_tensors.size(); ++i) {
    auto new_param = curr_g()->add_parameter();
    auto param_i = only_tensors[i];
    const auto &param_i_value = PyNativeAlgo::DataConvert::PyObjToValue(param_i);
    (void)input_param_values.emplace_back(param_i_value);
    const auto &param_i_id = PyNativeAlgo::PyParser::GetIdByPyObj(param_i);
    abstract::AbstractBasePtr param_i_abs = nullptr;
    auto item = dynamic_shape()->id_with_dynamic_abs().find(param_i_id);
    if (item != dynamic_shape()->id_with_dynamic_abs().end()) {
      MS_LOG(DEBUG) << "Param " << i << " is dynamic input";
      param_i_abs = item->second;
    } else {
      param_i_abs = param_i_value->ToAbstract();
      MS_EXCEPTION_IF_NULL(param_i_abs);
      param_i_abs = param_i_abs->Broaden();
    }
    MS_EXCEPTION_IF_NULL(param_i_abs);
    new_param->set_abstract(param_i_abs);
    top_cell()->SetTupleArgsToGraphInfoMap(curr_g(), param_i_value, new_param, true);
    top_cell()->SetNodeMapInGraphInfoMap(curr_g(), param_i_id, new_param);
    top_cell()->SetParamNodeMapInGraphInfoMap(curr_g(), param_i_id, new_param);
    top_cell()->SetParamNodeMapInGraphInfoMap(top_cell_->df_builder(), param_i_id, new_param);
  }
  top_cell()->set_k_pynative_cell_ptr(ad::GradPynativeCellBegin(curr_g()->parameters(), input_param_values));
}

void GradExecutor::InitResourceAndDfBuilder(const std::string &cell_id, const py::object &cell, const py::args &args) {
  if (cell_stack_.empty() || IsNestedGrad()) {
    if (cell_stack_.empty() && !grad_is_running_) {
      MS_LOG(DEBUG) << "Make new topest graph";
      MakeNewTopGraph(cell_id, cell, args, true);
    } else if (grad_is_running_ && IsBpropGraph(cell_id)) {
      MS_LOG(DEBUG) << "Run bprop cell";
      auto fg = std::make_shared<FuncGraph>();
      top_cell()->set_fg(fg);
      auto graph_info_cg = std::make_shared<GraphInfo>(cell_id);
      top_cell()->SetGraphInfoMap(fg, graph_info_cg);
      HandleInputArgsForTopCell(args, true);
      bprop_grad_stack_.push(std::make_pair(cell_id, false));
    } else if (grad_is_running_ && top_cell()->grad_order() != grad_order_) {
      MS_LOG(DEBUG) << "Nested grad graph existed in bprop";
      MakeNewTopGraph(cell_id, cell, args, false);
      bprop_grad_stack_.push(std::make_pair(cell_id, true));
    } else if (!cell_stack_.empty() && IsNestedGrad() && top_cell()->grad_order() != grad_order_) {
      MS_LOG(DEBUG) << "Nested grad graph existed in construct";
      auto cur_top_is_dynamic = top_cell()->is_dynamic_structure();
      MakeNewTopGraph(cell_id, cell, args, false);
      top_cell()->set_dynamic_structure(cur_top_is_dynamic);
    }
  }

  PushCellStack(cell_id);
  // Init kPynativeCellPtr with input parameters of top cell
  if (!top_cell()->is_init_kpynative()) {
    auto graph_info_cg = std::make_shared<GraphInfo>(cell_id);
    top_cell()->SetGraphInfoMap(curr_g(), graph_info_cg);
    auto graph_info_df = std::make_shared<GraphInfo>(cell_id);
    top_cell()->SetGraphInfoMap(top_cell_->df_builder(), graph_info_df);
    HandleInputArgsForTopCell(args, false);
    top_cell()->set_need_compile_graph(true);
    top_cell()->set_init_kpynative(true);
  } else {
    // Non-top cell
    top_cell()->SetSubCellList(cell_id);
  }
}

void GradExecutor::NewGraphInner(const py::object *ret, const py::object &cell, const py::args &args) {
  MS_EXCEPTION_IF_NULL(ret);
  const auto &cell_id = GetCellId(cell, args);
  MS_LOG(DEBUG) << "NewGraphInner start " << args.size() << " " << cell_id;
  bool pre_top_cell_is_dynamic_structure = false;
  if (top_cell_ != nullptr && cell_stack_.empty()) {
    // Already run top cell need distinguish high order; high order add "0" otherwise "1"
    const auto &already_run_cell_id = GetAlreadyRunCellId(cell_id);
    auto top_it = already_run_top_cell_.find(already_run_cell_id);
    if (top_it != already_run_top_cell_.end()) {
      // Top cell forward run.
      const auto &pre_top_cell = top_it->second;
      MS_EXCEPTION_IF_NULL(pre_top_cell);
      pre_top_cell_is_dynamic_structure = pre_top_cell->is_dynamic_structure();
      MS_LOG(DEBUG) << "Pre top cell, hook_changed " << pre_top_cell->hook_changed() << ", is_dynamic_structure "
                    << pre_top_cell_is_dynamic_structure;
      if (pre_top_cell->hook_changed()) {
        (void)already_run_top_cell_.erase(top_it);
        EraseTopCellFromTopCellList(pre_top_cell);
      } else if (!pre_top_cell->is_dynamic_structure()) {
        MS_LOG(DEBUG) << "Top cell " << cell_id << " is not dynamic structure, no need to run NewGraphInner again";
        pre_top_cell->ResetTopCellInfo(args);
        PushHighOrderGraphStack(pre_top_cell);
        set_top_cell(pre_top_cell);
        grad_order_ = pre_top_cell->grad_order();
        return;
      }
    } else if ((top_cell()->IsSubCell(cell_id) || GetHighOrderStackSize() >= 1) &&
               !IsCellObjIdEq(cell_id, check_graph_cell_id_)) {
      // Sub cell ( or may be a temporary cell, but must be non top) forward run in cache process.
      MS_LOG(DEBUG) << "Sub cell no need to run NewGraphInner again";
      return;
    }
  }
  // When the cell has custom bprop, in_custom_bprop_cell is lager than 0
  if (py::hasattr(cell, parse::CUSTOM_BPROP_NAME)) {
    custom_bprop_cell_count_ += 1;
  }
  // Make top graph and init resource for resource and df_builder
  InitResourceAndDfBuilder(cell_id, cell, args);
  // Check whether cell has dynamic construct
  if (pre_top_cell_is_dynamic_structure) {
    top_cell()->set_dynamic_structure(true);
  }
  if (!top_cell()->is_dynamic_structure()) {
    bool is_dynamic_structure = parse::DynamicParser::IsDynamicCell(cell);
    MS_LOG(DEBUG) << "Current cell dynamic " << is_dynamic_structure;
    if (is_dynamic_structure) {
      top_cell()->set_dynamic_structure(is_dynamic_structure);
    }
  }
}

void GradExecutor::MakeNewTopGraph(const string &cell_id, const py::object &cell, const py::args &args,
                                   bool is_topest) {
  pipeline::CheckArgsValid(cell, args);
  // Record input args info
  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) + "_";
  }
  // Run forward first need plus 1
  if (grad_order_ == 0) {
    ++grad_order_;
  }
  // The number of top cell exceeds MAX_TOP_CELL_COUNTS, delete the last one to keep the maximum length of the list,
  // disable backend cache
  if (top_cell_list_.size() >= MAX_TOP_CELL_COUNTS) {
    EnableOpGraphCache(false);
    // Delete top cell from begin
    auto delete_first_top_cell = top_cell_list_.front();
    MS_EXCEPTION_IF_NULL(delete_first_top_cell);
    delete_first_top_cell->Clear();
    (void)already_run_top_cell_.erase(delete_first_top_cell->already_run_cell_id());
    (void)top_cell_list_.erase(top_cell_list_.begin());
    MS_LOG(WARNING) << "Too many top cell has been built, please check if the cell " << cell.cast<CellPtr>()->ToString()
                    << " is repeatedly defined in each step/epoch, or the net input shape changes frequently.";
  }
  // Create top cell
  auto fg = std::make_shared<FuncGraph>();
  auto df_builder = std::make_shared<FuncGraph>();
  auto resource = std::make_shared<pipeline::Resource>();
  const auto &already_run_cell_id = GetAlreadyRunCellId(cell_id);
  auto top_cell =
    std::make_shared<TopCellInfo>(is_topest, grad_order_, resource, fg, df_builder, cell_id, already_run_cell_id);
  top_cell->set_forward_already_run(true);
  top_cell->set_input_args_id(input_args_id);
  TopCellInfoPtr top_cell_with_dynamic_shape = dynamic_shape()->GetTopCellWithDynamicShape(cell, args, true);
  if (top_cell_with_dynamic_shape != nullptr) {
    top_cell->set_cell_id(top_cell_with_dynamic_shape->cell_id());
    top_cell->set_already_run_cell_id(top_cell_with_dynamic_shape->already_run_cell_id());
    top_cell->set_cell_self_info(top_cell_with_dynamic_shape->cell_self_info());
    EraseTopCellFromTopCellList(top_cell_with_dynamic_shape);
    MS_LOG(DEBUG) << "Pre top cell and current top cell merged to one top cell with dynamic shape";
  } else {
    top_cell->SetCellSelfInfoForTopCell(cell, args);
  }
  (void)top_cell_list_.emplace_back(top_cell);
  PushHighOrderGraphStack(top_cell);
  set_top_cell(top_cell);
  MS_LOG(DEBUG) << "New top graph, fg ptr " << fg.get() << " resource ptr " << resource.get();
}

void GradExecutor::SetForwardLastNodeInfo(const ValuePtr &v, const std::string &obj_id) const {
  MS_EXCEPTION_IF_NULL(v);
  auto output_node = GetObjNode(v, obj_id);
  if (v->isa<tensor::CSRTensor>()) {
    auto csr_tensorptr = v->cast<tensor::CSRTensorPtr>();
    auto value_ptr = csr_tensorptr->GetValues();
    output_node = GetObjNode(value_ptr, PyNativeAlgo::Common::GetIdByValue(value_ptr));
  } else if (v->isa<tensor::COOTensor>()) {
    auto coo_tensorptr = v->cast<tensor::COOTensorPtr>();
    auto value_ptr = coo_tensorptr->GetValues();
    output_node = GetObjNode(value_ptr, PyNativeAlgo::Common::GetIdByValue(value_ptr));
  }
  MS_EXCEPTION_IF_NULL(output_node);
  if (top_cell()->dynamic_shape()) {
    abstract::AbstractBasePtr last_node_abs = nullptr;
    if (output_node->abstract() == nullptr) {
      last_node_abs = v->ToAbstract()->Broaden();
    } else {
      last_node_abs = output_node->abstract();
    }
    MS_EXCEPTION_IF_NULL(last_node_abs);
    // Set last output abstract and will be used for sens
    top_cell()->set_last_output_abs(last_node_abs);
  }
  // Set last node and sens for build adjoint
  const auto &sens_value = dynamic_shape()->GetSensValueForDynamicShapeOutput(top_cell(), v, output_node);
  auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
  MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
  k_pynative_cell_ptr->UpdateOutputNodeOfTopCell(output_node, sens_value);
}

void GradExecutor::EndGraphInner(const py::object *ret, const py::object &cell, const py::object &out,
                                 const py::args &args) {
  MS_EXCEPTION_IF_NULL(ret);
  constexpr size_t high_order_stack_size = 2;
  const auto &cell_id = GetCellId(cell, args);
  MS_LOG(DEBUG) << "EndGraphInner start " << args.size() << " " << cell_id;
  if (cell_stack_.empty()) {
    if (cell_id == top_cell()->cell_id()) {
      if (top_cell()->is_topest()) {
        set_grad_flag(false);
      }
      if (GetHighOrderStackSize() < high_order_stack_size) {
        auto outer_top_cell = PopHighOrderGraphStack();
        if (outer_top_cell != nullptr) {
          set_top_cell(outer_top_cell);
        }
      }
      // Top cell update sens
      dynamic_shape()->UpdateSensValueForDynamicShapeOutput(top_cell(), PyNativeAlgo::DataConvert::PyObjToValue(out));
    }
    MS_LOG(DEBUG) << "Current cell " << cell_id << " no need to run EndGraphInner again";
    return;
  }
  const auto &out_id = PyNativeAlgo::PyParser::GetIdByPyObj(out);
  const auto &out_value = PyNativeAlgo::DataConvert::PyObjToValue(out);
  DoGradForCustomBprop(cell, args, out_value, out_id);
  PopCellStack();
  if (grad_is_running_ && !bprop_grad_stack_.empty()) {
    if (!bprop_grad_stack_.top().second) {
      curr_g()->set_output(GetObjNode(out_value, out_id));
      bprop_grad_stack_.pop();
      return;
    } else if (bprop_grad_stack_.top().first == cell_id) {
      bprop_grad_stack_.pop();
    }
  }
  // Just only dump the last forward graph
  bool is_top_cell_end = cell_id == top_cell()->cell_id();
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && is_top_cell_end) {
    curr_g()->set_output(GetObjNode(out_value, out_id));
#ifdef ENABLE_DUMP_IR
    DumpIR("fg.ir", curr_g());
#endif
  }
  // Reset grad flag and update output node of the outermost cell
  if (cell_stack_.empty() && is_top_cell_end) {
    MS_LOG(DEBUG) << "Cur top last cell " << cell_id;
    (void)PopHighOrderGraphStack();
    SetForwardLastNodeInfo(out_value, out_id);
    top_cell()->ClearCellHookOp();
    cell_order_ = 0;
    set_grad_flag(false);
  }
  // Checkout whether need to compile graph when each top cell has ran finished
  if (is_top_cell_end) {
    // In high grad cases, the output of the internal graph may be a tuple, and node needs to be created in the getobj
    if (!cell_stack_.empty()) {
      SetForwardLastNodeInfo(out_value, out_id);
    }
    top_cell()->CheckSubCellHookChanged();
    CheckNeedCompileGraph();
  }
}

void GradExecutor::DoGradForCustomBprop(const py::object &cell, const py::args &args, const ValuePtr &out,
                                        const std::string &out_id) {
  MS_EXCEPTION_IF_NULL(out);
  if (!py::hasattr(cell, parse::CUSTOM_BPROP_NAME)) {
    return;
  }
  custom_bprop_cell_count_ -= 1;
  if (custom_bprop_cell_count_ != 0) {
    return;
  }
  MS_LOG(DEBUG) << "Do grad for custom bprop";
  py::function bprop_func = py::getattr(cell, parse::CUSTOM_BPROP_NAME);
  py::object code_obj = py::getattr(bprop_func, "__code__");
  // When the co_names is empty, we will still get a tuple which is empty.
  auto co_names = py::getattr(code_obj, "co_names").cast<py::tuple>();
  for (auto name : co_names) {
    if (!py::hasattr(cell, name)) {
      continue;
    }
    auto var = py::getattr(cell, name);
    if (py::hasattr(var, "__parameter__") && py::isinstance<tensor::MetaTensor>(var)) {
      MS_LOG(EXCEPTION) << "The user defined 'bprop' function does not support using Parameter.";
    }
  }

  auto bprop_func_cellid = GetFnInfoByPyObj(bprop_func);
  (void)bprop_cell_list_.emplace_back(bprop_func_cellid);
  auto fake_prim = std::make_shared<PrimitivePy>(prim::kPrimHookBackward->name());
  if (py::isinstance<Cell>(cell)) {
    auto cell_ptr = py::cast<CellPtr>(cell);
    fake_prim->set_bprop_cls_name(cell_ptr->name());
  }
  fake_prim->AddBackwardHookFn(0, bprop_func);

  const auto &cell_id = GetCellId(cell, args);
  (void)fake_prim->AddAttr("cell_id", MakeValue(cell_id));
  (void)fake_prim->AddAttr(parse::CUSTOM_BPROP_NAME, MakeValue(true));

  py::object co_name = py::getattr(code_obj, "co_name");
  if (std::string(py::str(co_name)) == "staging_specialize") {
    MS_LOG(EXCEPTION) << "Decorating bprop with '@ms_function' is not supported.";
  }
  // Three parameters self, out and dout need to be excluded
  const size_t inputs_num = static_cast<size_t>(py::cast<int64_t>(py::getattr(code_obj, "co_argcount")) - 3);
  if (inputs_num != args.size()) {
    MS_EXCEPTION(TypeError) << "Size of bprop func inputs[" << inputs_num
                            << "] is not equal to the size of cell inputs[" << args.size() << "]";
  }

  py::list cell_inputs;
  for (size_t i = 0; i < inputs_num; i += 1) {
    cell_inputs.append(args[i]);
  }
  FrontendOpRunInfoPtr op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->base_op_run_info.op_name = fake_prim->name();
  op_run_info->op_prim = fake_prim;
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, cell_inputs);
  auto cnode = ConstructForwardGraph(op_run_info);
  DoOpGrad(op_run_info, cnode, out);
  SaveOutputNodeMap(out_id, out, cnode);
}

std::string GradExecutor::GetAlreadyRunCellId(const std::string &cell_id) const {
  std::string already_run_cell_id(cell_id);
  already_run_cell_id += std::to_string(grad_order_ == 0 ? 1 : grad_order_);
  already_run_cell_id += "_" + grad_operation_;
  MS_LOG(DEBUG) << "Get already run top cell id " << already_run_cell_id;
  return already_run_cell_id;
}

std::string GradExecutor::GetGradCellId(bool has_sens, const py::object &cell, const py::args &args) const {
  size_t forward_args_size = args.size();
  py::args tmp = args;
  if (has_sens) {
    forward_args_size--;
    py::tuple f_args(forward_args_size);
    for (size_t i = 0; i < forward_args_size; ++i) {
      f_args[i] = args[i];
    }
    tmp = f_args;
  }
  const auto &cell_id = GetCellId(cell, tmp);
  return cell_id;
}

void GradExecutor::GradNetInner(const py::object *ret, const prim::GradOperationPtr &grad, const py::object &cell,
                                const py::object &weights, const py::object &grad_position, const py::args &args) {
  MS_EXCEPTION_IF_NULL(ret);
  MS_EXCEPTION_IF_NULL(grad);
  auto size = args.size();
  const auto &cell_id = GetGradCellId(grad->sens_param(), cell, args);
  MS_LOG(DEBUG) << "GradNet start " << size << " " << cell_id;
  if (!top_cell()->need_compile_graph()) {
    MS_LOG(DEBUG) << "No need compile graph";
    UpdateTopCellInfo(false, false, !cell_stack_.empty());
    return;
  }
  top_cell()->set_grad_operation(grad_operation_);
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  auto df_builder = top_cell()->df_builder();
  MS_EXCEPTION_IF_NULL(df_builder);
  MS_LOG(DEBUG) << "fg ptr " << curr_g().get() << " resource ptr " << resource.get();

  // Get params(weights) require derivative
  auto w_args = GetWeightsArgs(weights, df_builder);
  auto p_args = GetGradPositionArgs(grad_position, grad->get_by_position_);
  bool weight_param_is_tuple = true;
  if (w_args.empty()) {
    MS_LOG(DEBUG) << "Add weights params to w_args";
    if (py::isinstance<py::tuple>(weights) || py::isinstance<py::list>(weights)) {
      auto weights_list = py::cast<py::list>(weights);
      for (size_t i = 0; i < weights_list.size(); ++i) {
        (void)w_args.emplace_back(GetInput(PyNativeAlgo::DataConvert::PyObjToValue(weights_list[i])));
      }
    } else if (!py::isinstance<py::none>(weights)) {
      // Single input
      (void)w_args.emplace_back(GetInput(PyNativeAlgo::DataConvert::PyObjToValue(weights)));
      weight_param_is_tuple = false;
    } else {
      (void)w_args.insert(w_args.end(), df_builder->parameters().cbegin(), df_builder->parameters().cend());
    }
  }
  // Get bprop graph of top cell
  ad::GradAttr grad_attr(grad->get_all_, grad->get_by_list_, grad->sens_param_, grad->get_by_position_,
                         weight_param_is_tuple);
  auto bprop_graph = GetBpropGraph(grad_attr, cell, w_args, p_args, size, args);
  MS_EXCEPTION_IF_NULL(bprop_graph);
  bprop_graph->set_flag(kFlagIsPynativeBpropGraph, true);
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph, true);
  DumpGraphIR("launch_bprop_graph.ir", bprop_graph);
  // Launch bprop graph to backend
  SaveForwardTensorInfoInBpropGraph(resource);
  compile::SetMindRTEnable();
  resource->SetBackendAsync([]() { return compile::CreateBackend(); });
  MS_LOG(DEBUG) << "Start task emit action";
  (void)TaskEmitAction(resource);
  MS_LOG(DEBUG) << "Start execute action";
  (void)ExecuteAction(resource);
  MS_LOG(DEBUG) << "Start update top cell info when run finish";
  UpdateTopCellInfo(false, false, true);
  resource->Clean();
  abstract::AnalysisContext::ClearContext();
  // Clean cache used for parse. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  trace::ClearTraceStack();
}

std::vector<AnfNodePtr> GradExecutor::GetWeightsArgs(const py::object &weights, const FuncGraphPtr &df_builder) const {
  MS_EXCEPTION_IF_NULL(df_builder);
  if (!py::hasattr(weights, "__parameter_tuple__")) {
    MS_LOG(DEBUG) << "No parameter tuple get";
    return {};
  }

  const auto &tuple = weights.cast<py::tuple>();
  MS_LOG(DEBUG) << "Get weights tuple size " << tuple.size();
  std::vector<AnfNodePtr> w_args;
  for (size_t it = 0; it < tuple.size(); ++it) {
    auto param = tuple[it];
    auto param_id = PyNativeAlgo::PyParser::GetIdByPyObj(param);
    auto &graph_info_map = top_cell()->graph_info_map();
    if (graph_info_map.find(df_builder) == graph_info_map.end()) {
      MS_LOG(EXCEPTION) << "Can not find df_builder " << df_builder.get() << " Top cell " << top_cell().get()
                        << " cell id " << top_cell()->cell_id();
    }
    const auto &graph_info = graph_info_map.at(df_builder);
    MS_EXCEPTION_IF_NULL(graph_info);
    AnfNodePtr para_node = nullptr;
    if (graph_info->params.find(param_id) != graph_info->params.end()) {
      para_node = graph_info->params.at(param_id);
      (void)w_args.emplace_back(para_node);
      continue;
    }
    const auto &name_attr = python_adapter::GetPyObjAttr(param, "name");
    if (py::isinstance<py::none>(name_attr)) {
      MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
    }
    const auto &param_name = py::cast<std::string>(name_attr);
    MS_LOG(DEBUG) << "The input " << it << " parameter weight name " << param_name;
    if (graph_info->params.find(param_name) != graph_info->params.end()) {
      para_node = graph_info->params.at(param_name);
    } else {
      MS_LOG(DEBUG) << "Can not find input param in graph info map, make a new parameter";
      auto free_param = df_builder->add_parameter();
      free_param->set_name(param_name);
      auto value = py::cast<tensor::TensorPtr>(param);
      free_param->set_default_param(value);
      free_param->debug_info()->set_name(param_name);
      para_node = free_param;
    }
    (void)w_args.emplace_back(para_node);
  }
  return w_args;
}

std::vector<size_t> GradExecutor::GetGradPositionArgs(const py::object &grad_position,
                                                      const bool get_by_position) const {
  std::vector<size_t> pos_args;
  if (!get_by_position) {
    return pos_args;
  }
  if (py::isinstance<py::tuple>(grad_position)) {
    const auto &tuple = grad_position.cast<py::tuple>();
    (void)std::transform(tuple.begin(), tuple.end(), std::back_inserter(pos_args),
                         [](const py::handle &elem) { return py::cast<int64_t>(elem); });
    return pos_args;
  }
  MS_LOG(EXCEPTION) << "Grad position only support tuple when grad_by_position is set True.";
}

void GradExecutor::ShallowCopySensValue(const py::tuple &input_args, bool has_sens, VectorRef *run_args) const {
  if (!has_sens) {
    return;
  }
  // Get index and number of sens args.
  size_t sens_index = input_args.size() - 1;
  size_t sens_num = 1;
  if (py::isinstance<py::tuple>(input_args[sens_index])) {
    py::tuple tuple_sens = py::cast<py::tuple>(input_args[sens_index]);
    sens_num = PyNativeAlgo::DataConvert::ConvertArgs(tuple_sens).size();
  }
  // Shallow copy sens args to new sens args.
  MS_EXCEPTION_IF_NULL(run_args);
  for (size_t i = sens_index; i < sens_index + sens_num; ++i) {
    const auto &original_sens = (*run_args)[i];
    if (utils::isa<ValuePtr>(original_sens)) {
      auto sens_value = utils::cast<ValuePtr>(original_sens);
      MS_EXCEPTION_IF_NULL(sens_value);
      auto new_sens_value = ShallowCopyTensorValue(sens_value);
      MS_EXCEPTION_IF_NULL(new_sens_value);
      MS_LOG(DEBUG) << "sens args [" << sens_value->ToString() << "] has been shallow copied to ["
                    << new_sens_value->ToString() << "].";
      (*run_args)[i] = new_sens_value;
    }
  }
}

void GradExecutor::CheckParamShapeAndType(const AnfNodePtr &param, const ParameterPtr &param_node,
                                          const abstract::AbstractBasePtr &input_abs,
                                          const abstract::AbstractBasePtr &param_tensor_abs,
                                          const std::string &input_shape) {
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(param_tensor_abs);
  auto ir_base_shape = param_tensor_abs->BuildShape();
  MS_EXCEPTION_IF_NULL(ir_base_shape);
  auto ir_shape = ir_base_shape->ToString();
  if (input_shape != "()" && ir_shape != "()") {
    if (input_shape != ir_shape) {
      // Sens shape in ir graph is determined by graph output, so it can be dynamic shape; But input shape is
      // determined by user input, which could not be dynamic shape.
      if (param_node->debug_info()->name() != "sens" || !ir_base_shape->IsDynamic()) {
        MS_EXCEPTION(ValueError) << "The shape should be " << ir_shape << ", but got " << input_shape << ", "
                                 << param->DebugString();
      }
    }
    auto ir_dtype = param_tensor_abs->BuildType()->ToString();
    MS_EXCEPTION_IF_NULL(input_abs);
    auto input_dtype = input_abs->BuildType()->ToString();
    if (input_dtype != ir_dtype) {
      MS_EXCEPTION(TypeError) << "The dtype should be " << ir_dtype << ", but got " << input_dtype << ", "
                              << param->DebugString();
    }
  }
  if (param_node->debug_info()->name() == "sens" && ir_shape != input_shape) {
    need_renormalize_ = true;
  }
}

void GradExecutor::UpdateParamAbsByArgs(const py::list &args, const FuncGraphPtr &bprop_graph) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  const auto &bprop_params = bprop_graph->parameters();
  // bprop_params include inputs, parameters, more than size(inputs)
  if (bprop_params.size() < args.size()) {
    MS_LOG(EXCEPTION) << "Df parameters size " << bprop_params.size() << " less than " << args.size();
  }
  size_t index = 0;
  for (const auto &param : bprop_params) {
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      // update abstract info for weights
      ValuePtr value = param_node->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto ptr = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(ptr);
      param_node->set_abstract(ptr->Broaden());
    } else {
      // update abstract info for input params
      abstract::AbstractBasePtr input_abs;
      auto it = dynamic_shape()->id_with_dynamic_abs().find(PyNativeAlgo::PyParser::GetIdByPyObj(args[index]));
      if (it != dynamic_shape()->id_with_dynamic_abs().end()) {
        input_abs = it->second;
      } else {
        input_abs = abstract::FromValue(PyNativeAlgo::DataConvert::PyObjToValue(args[index]), true);
      }
      MS_EXCEPTION_IF_NULL(input_abs);
      if (param_node->abstract() != nullptr) {
        auto input_shape = input_abs->BuildShape()->ToString();
        auto param_tensor_abs = param_node->abstract();
        if (param_tensor_abs->isa<abstract::AbstractRefTensor>()) {
          param_tensor_abs = param_tensor_abs->cast<abstract::AbstractRefPtr>()->CloneAsTensor();
        }
        CheckParamShapeAndType(param, param_node, input_abs, param_tensor_abs, input_shape);
      }
      param_node->set_abstract(input_abs->Broaden());
      index++;
    }
  }
}

FuncGraphPtr GradExecutor::GetBpropGraph(const ad::GradAttr &grad_attr, const py::object &cell,
                                         const std::vector<AnfNodePtr> &weights,
                                         const std::vector<size_t> &grad_position, size_t arg_size,
                                         const py::args &args) {
  bool build_formal_param = false;
  if (!py::hasattr(cell, parse::CUSTOM_BPROP_NAME) && !cell_stack_.empty() && IsNestedGrad()) {
    build_formal_param = true;
    need_renormalize_ = true;
  }
  if (top_cell()->ms_function_flag()) {
    need_renormalize_ = true;
  }

  auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
  MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
  FuncGraphPtr bprop_graph =
    ad::GradPynativeCellEnd(k_pynative_cell_ptr, weights, grad_position, grad_attr, build_formal_param);
  MS_EXCEPTION_IF_NULL(bprop_graph);

  MS_LOG(DEBUG) << "Top graph input params size " << arg_size;
  std::ostringstream ss;
  ss << "grad{" << arg_size << "}";
  bprop_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop_graph->debug_info()->set_name(ss.str());
  // Get the parameters items and add the value to args_spec
  if (top_cell()->dynamic_shape() && grad_attr.has_sens) {
    MS_EXCEPTION_IF_NULL(top_cell()->last_output_abs());
    auto shape = top_cell()->last_output_abs()->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->IsDynamic()) {
      const auto &sens_id = PyNativeAlgo::PyParser::GetIdByPyObj(args[arg_size - 1]);
      dynamic_shape()->SetIdWithDynamicAbs(sens_id, top_cell()->last_output_abs());
    }
  }
  UpdateParamAbsByArgs(PyNativeAlgo::PyParser::FilterTensorArgs(args, grad_attr.has_sens), bprop_graph);
  // Dynamic shape graph need add some other pass
  if (top_cell()->dynamic_shape()) {
    bprop_graph->set_flag(FUNC_GRAPH_FLAG_DYNAMIC_SHAPE, true);
  }
  if (top_cell()->is_real_dynamic_structure()) {
    bprop_graph->set_flag(kFlagIsDynamicStructure, true);
  }
  SetBpropGraphJitLevel(cell);
  // Do opt for final bprop graph
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph);
  auto optimized_bg = ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().BpropGraphFinalOpt(resource);

  if (cell_stack_.empty()) {
    need_renormalize_ = false;
  }
  DumpGraphIR("after_final_opt.ir", optimized_bg);
  return optimized_bg;
}

void GradExecutor::CheckGraph(const py::object &cell, const py::args &args) {
  check_graph_cell_id_ = GetCellId(cell, args);
  if (!(top_cell_ != nullptr && check_graph_cell_id_.find(top_cell_->cell_id()) != std::string::npos &&
        grad_order_ >= 1)) {
    ++grad_order_;
  }
  if (!grad_is_running_) {
    MS_LOG(DEBUG) << "Grad not running yet";
    return;
  }
  MS_LOG(DEBUG) << "Key is " << check_graph_cell_id_;
  if (top_cell_ != nullptr) {
    for (auto it = top_cell_->sub_cell_list().begin(); it != top_cell_->sub_cell_list().end(); ++it) {
      MS_LOG(DEBUG) << "Cur cell id " << *it;
      if (!IsCellObjIdEq(*it, check_graph_cell_id_)) {
        continue;
      }
      MS_LOG(DEBUG) << "Delete cellid from cell graph list, top cell is " << top_cell_;
      top_cell_->EraseFromSubCellList(*it);
      break;
    }
  }
}

py::object GradExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &cell,
                                         const py::object &grad_hash_id, const py::args &args) {
  // Check current cell grad order and erase it if in current top cell list
  CheckGraph(cell, args);

  bool forward_run = false;
  // Get cell id and input args info
  const auto &cell_id = GetCellId(cell, args);
  std::string grad_hash_id_str;
  if (!py::isinstance<py::none>(grad_hash_id)) {
    grad_hash_id_str = std::string(py::str(grad_hash_id));
  }
  grad_operation_ = std::to_string(static_cast<int>(grad->get_all_)) +
                    std::to_string(static_cast<int>(grad->get_by_list_)) + grad_hash_id_str;

  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) + "_";
  }
  // Under the condition that the stack is empty (forward process completed or no forward process),
  // check whether need to run forward process
  if (cell_stack_.empty() && top_cell_ != nullptr) {
    const auto &check_already_run_cell_id = GetAlreadyRunCellId(cell_id);
    auto find_top_cell = GetTopCell(check_already_run_cell_id);
    if (find_top_cell != nullptr) {
      MS_LOG(DEBUG) << "Find already run top cell";
      forward_run = find_top_cell->forward_already_run();
      const auto &curr_top_cell = top_cell();
      set_top_cell(find_top_cell);
      bool input_args_changed =
        !find_top_cell->input_args_id().empty() && find_top_cell->input_args_id() != input_args_id;
      if (forward_run && input_args_changed && find_top_cell->is_dynamic_structure()) {
        MS_LOG(WARNING) << "The construct of running cell is dynamic and the input info of this cell has changed, "
                           "forward process will run again";
        forward_run = false;
      }
      if (forward_run && GetHighOrderStackSize() >= 1) {
        PushHighOrderGraphStack(curr_top_cell);
      }
    }
  }
  MS_LOG(DEBUG) << "Graph have already ran " << forward_run << " top cell id " << cell_id;
  return BaseRefToPyData(forward_run);
}

void GradExecutor::CheckNeedCompileGraph() {
  const auto &new_top_cell = top_cell();
  const auto &already_top_cell_id = new_top_cell->already_run_cell_id();
  // Update top cell by current cell op info
  if (already_run_top_cell_.find(already_top_cell_id) == already_run_top_cell_.end()) {
    MS_LOG(DEBUG) << "Top cell " << new_top_cell->cell_id() << " has never been ran, need compile graph";
    already_run_top_cell_[already_top_cell_id] = new_top_cell;
    return;
  }

  MS_LOG(DEBUG) << "Top cell " << new_top_cell->cell_id() << " has been ran";
  auto pre_top_cell = already_run_top_cell_.at(already_top_cell_id);
  MS_EXCEPTION_IF_NULL(pre_top_cell);
  const auto &pre_all_op_info = pre_top_cell->all_op_info();
  const auto &new_all_op_info = new_top_cell->all_op_info();
  MS_LOG(DEBUG) << "Pre all op info : " << pre_all_op_info;
  MS_LOG(DEBUG) << "New all op info : " << new_all_op_info;
  if (pre_all_op_info != new_all_op_info) {
    MS_LOG(DEBUG) << "The op info has been changed, need to compile graph again";
    // The top cell switches exceeds MAX_TOP_CELL_COUNTS under the control flow, disable backend cache
    if (top_cell_switch_counts_ >= MAX_TOP_CELL_COUNTS) {
      EnableOpGraphCache(false);
    } else {
      // Increase top cell switches counts
      ++top_cell_switch_counts_;
    }

    auto has_higher_order = std::any_of(top_cell_list_.begin(), top_cell_list_.end(),
                                        [](const TopCellInfoPtr &value) { return !value->is_topest(); });
    EraseTopCellFromTopCellList(pre_top_cell);
    if (pre_top_cell->is_topest() && !has_higher_order) {
      pre_top_cell->ClearDeviceMemory();
    }
    pre_top_cell->Clear();
    already_run_top_cell_[already_top_cell_id] = new_top_cell;
    top_cell()->set_is_real_dynamic_structure(true);
  } else {
    MS_LOG(DEBUG) << "The op info has not been changed, no need to compile graph again";
    pre_top_cell->set_input_args_id(new_top_cell->input_args_id());
    // In high order situations, the internal top cell remains unchanged, but the external top cell has changed. Then
    // the graph info of the internal top cell needs to be updated so that the external top cell can perceive it.
    if (!cell_stack_.empty()) {
      pre_top_cell->SetGraphInfoMap(pre_top_cell->df_builder(),
                                    new_top_cell->graph_info_map().at(new_top_cell->df_builder()));
    }
    EraseTopCellFromTopCellList(new_top_cell);
    new_top_cell->Clear();
    pre_top_cell->set_forward_already_run(true);
    set_top_cell(pre_top_cell);
  }
}

void GradExecutor::RunGradGraph(py::object *ret, const py::object &cell, const py::object &sens_param,
                                const py::tuple &args) {
  MS_EXCEPTION_IF_NULL(ret);
  bool has_sens = sens_param.cast<bool>();
  const auto &cell_id = GetGradCellId(has_sens, cell, args);
  MS_LOG(DEBUG) << "Run has sens " << has_sens << " cell id " << cell_id;
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Run resource ptr " << resource.get();

  VectorRef arg_list;
  auto filter_args = PyNativeAlgo::PyParser::FilterTensorArgs(args, has_sens);
  py::tuple converted_args = PyNativeAlgo::DataConvert::ConvertArgs(filter_args);
  pipeline::ProcessVmArgInner(converted_args, resource, &arg_list);
  ShallowCopySensValue(filter_args, has_sens, &arg_list);
  MS_LOG(DEBUG) << "Convert args size " << converted_args.size() << ", graph param size " << arg_list.size();
  compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
  MS_EXCEPTION_IF_NULL(run);

  const auto &backend = MsContext::GetInstance()->backend_policy();
  MS_LOG(DEBUG) << "Eval run " << backend;
  grad_is_running_ = true;
  top_cell()->set_k_pynative_cell_ptr(nullptr);
  BaseRef value = (*run)(arg_list);
  grad_is_running_ = false;
  FuncGraphPtr fg = resource->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto output_abs = fg->output()->abstract();
  MS_LOG(DEBUG) << "Eval run end " << value.ToString();
  *ret = BaseRefToPyData(value, output_abs);
  // Clear device memory resource of top cell when it has been ran.
  auto has_higher_order = std::any_of(top_cell_list_.begin(), top_cell_list_.end(),
                                      [](const TopCellInfoPtr &value) { return !value->is_topest(); });
  if (top_cell()->is_topest() && !has_higher_order) {
    top_cell()->ClearDeviceMemory();
    if (!py::isinstance<Cell>(cell)) {
      ClearCellRes(cell);
    }
  }
  // High order
  constexpr size_t high_order_size = 2;
  if (top_cell()->vm_compiled()) {
    MakeNestedCnode(cell, converted_args, resource, *ret);
  } else if (GetHighOrderStackSize() >= high_order_size) {
    SwitchTopcell();
  }
}

void GradExecutor::SwitchTopcell() {
  const auto &inner_top_cell_all_op_info = top_cell()->all_op_info();
  bool inner_top_cell_is_dynamic = top_cell()->is_dynamic_structure();

  // Get outer top cell
  auto outer_top_cell = PopHighOrderGraphStack();
  MS_EXCEPTION_IF_NULL(outer_top_cell);
  const auto &outer_top_cell_all_op_info = outer_top_cell->all_op_info();
  outer_top_cell->set_all_op_info(outer_top_cell_all_op_info + inner_top_cell_all_op_info);
  // If inner is dynamic, outer set dynamic too
  if (inner_top_cell_is_dynamic) {
    outer_top_cell->set_dynamic_structure(inner_top_cell_is_dynamic);
  }
  set_top_cell(outer_top_cell);
}

void GradExecutor::DoParameterReplace(const FuncGraphPtr &first_grad_fg, const py::tuple &forward_args,
                                      std::vector<AnfNodePtr> *inputs, ValuePtrList *weights_args) {
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(weights_args);

  auto first_df_builder = top_cell()->df_builder();
  MS_EXCEPTION_IF_NULL(first_df_builder);
  auto first_graph_info = top_cell()->graph_info_map().at(first_df_builder);
  MS_EXCEPTION_IF_NULL(first_graph_info);
  SwitchTopcell();
  auto second_df_builder = top_cell()->df_builder();
  MS_EXCEPTION_IF_NULL(second_df_builder);
  auto second_graph_info = top_cell()->graph_info_map().at(second_df_builder);
  MS_EXCEPTION_IF_NULL(second_graph_info);

  mindspore::HashSet<std::string> params_weights_set;
  mindspore::HashSet<std::string> params_inputs_set;
  for (const auto &sec : second_graph_info->params) {
    if (sec.second->has_default()) {
      (void)params_weights_set.emplace(sec.first);
    } else {
      (void)params_inputs_set.insert(sec.first);
    }
  }
  auto manager = Manage({first_grad_fg}, false);
  // Replace inputs param
  for (size_t i = 0; i < forward_args.size(); ++i) {
    const auto &id = PyNativeAlgo::PyParser::GetIdByPyObj(forward_args[i]);
    if (params_inputs_set.count(id) != 0) {
      // Can find in second graph
      const auto &input_param_second = second_graph_info->params.at(id);
      (void)manager->Replace(first_graph_info->params.at(id), input_param_second);
      (void)inputs->emplace_back(input_param_second);
    } else {
      (void)inputs->emplace_back(GetInput(PyNativeAlgo::DataConvert::PyObjToValue(forward_args[i])));
    }
  }

  // Replace weights param
  for (const auto &fir : first_graph_info->params) {
    if (!fir.second->has_default()) {
      continue;
    }
    // Second graph no this weight param, need add to second graph
    if (params_weights_set.count(fir.first) == 0) {
      MS_LOG(DEBUG) << "Can't find " << fir.first << " in outer graph, add it";
      second_df_builder->add_parameter(fir.second);
      top_cell()->SetParamNodeMapInGraphInfoMap(second_df_builder, fir.first, fir.second);
      (void)inputs->emplace_back(fir.second);
      (void)weights_args->emplace_back(fir.second->default_param());
    } else {
      // Need replace
      MS_LOG(DEBUG) << "Param name " << fir.first << " ptr " << fir.second.get();
      auto it = std::find_if(second_graph_info->params.begin(), second_graph_info->params.end(),
                             [&fir](const std::pair<std::string, ParameterPtr> &sec) {
                               return sec.second->has_default() && fir.second->name() == sec.second->name();
                             });
      if (it != second_graph_info->params.end()) {
        (void)manager->Replace(fir.second, it->second);
        (void)inputs->emplace_back(it->second);
        (void)weights_args->emplace_back(it->second->default_param());
      }
    }
  }
}

void GradExecutor::MakeNestedCnode(const py::object &cell, const py::tuple &forward_args,
                                   const pipeline::ResourcePtr &resource, const py::object &out) {
  if (cell_stack_.empty()) {
    MS_LOG(DEBUG) << "No nested grad find";
    return;
  }
  FuncGraphPtr first_grad_fg = nullptr;
  if (py::hasattr(cell, parse::CUSTOM_BPROP_NAME)) {
    first_grad_fg = curr_g();
    MS_LOG(DEBUG) << "Bprop nested";
  } else {
    first_grad_fg = resource->func_graph();
  }
  MS_EXCEPTION_IF_NULL(first_grad_fg);
  DumpGraphIR("first_grad_fg.ir", first_grad_fg);

  std::vector<AnfNodePtr> inputs{NewValueNode(first_grad_fg)};
  ValuePtrList weights_args;
  DoParameterReplace(first_grad_fg, forward_args, &inputs, &weights_args);

  pipeline::ResourcePtr r = std::make_shared<pipeline::Resource>();
  r->manager()->AddFuncGraph(first_grad_fg);
  set_eliminate_forward(false);
  (void)first_grad_fg->transforms().erase(kGrad);
  FuncGraphPtr second_grad_fg = ad::Grad(first_grad_fg, opt::Optimizer::MakeEmptyOptimizer(r));
  set_eliminate_forward(true);
  DumpGraphIR("second_grad_fg.ir", second_grad_fg);
  r->Clean();

  MS_LOG(DEBUG) << "Get pre graph ptr " << curr_g().get();
  auto cnode = curr_g()->NewCNode(inputs);
  auto out_id = PyNativeAlgo::PyParser::GetIdByPyObj(out);
  top_cell()->SetTupleArgsToGraphInfoMap(curr_g(), PyNativeAlgo::DataConvert::PyObjToValue(out), cnode);
  top_cell()->SetNodeMapInGraphInfoMap(curr_g(), out_id, cnode);
  MS_LOG(DEBUG) << "Nested make cnode is " << cnode->DebugString();

  // Get input values
  ValuePtrList input_args;
  for (size_t i = 0; i < forward_args.size(); ++i) {
    const auto &arg = PyNativeAlgo::DataConvert::PyObjToValue(forward_args[i]);
    (void)input_args.emplace_back(arg);
  }
  (void)input_args.insert(input_args.end(), weights_args.cbegin(), weights_args.cend());
  // Get output values
  py::object new_out;
  if (py::hasattr(cell, parse::CUSTOM_BPROP_NAME) && !py::isinstance<py::tuple>(out)) {
    new_out = py::make_tuple(out);
  } else {
    new_out = out;
  }
  const auto &out_value = PyNativeAlgo::DataConvert::PyObjToValue(new_out);
  if (!top_cell()->k_pynative_cell_ptr()->KPynativeWithFProp(cnode, input_args, out_value, second_grad_fg)) {
    MS_LOG(EXCEPTION) << "Failed to run ad grad for second grad graph " << cnode->ToString();
  }
  need_renormalize_ = true;
}

void GradExecutor::EraseTopCellFromTopCellList(const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(top_cell);
  auto iter = std::find_if(top_cell_list_.begin(), top_cell_list_.end(),
                           [&](const TopCellInfoPtr &elem) { return elem.get() == top_cell.get(); });
  if (iter == top_cell_list_.end()) {
    MS_LOG(WARNING) << "Can not find top cell " << top_cell.get() << " cell id " << top_cell->cell_id()
                    << " from top cell list";
  } else {
    (void)top_cell_list_.erase(iter);
  }
}

void GradExecutor::ClearGrad(const py::object &cell, const py::args &args) {
  MS_LOG(DEBUG) << "Clear top cell grad resource " << GetCellId(cell, args);
  if (grad_order_ > 0) {
    --grad_order_;
  }
  check_graph_cell_id_.clear();
  grad_operation_.clear();
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
}

void GradExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear grad res";
  grad_flag_ = false;
  enable_op_cache_ = true;
  grad_is_running_ = false;
  need_renormalize_ = false;
  eliminate_forward_ = true;
  custom_bprop_cell_count_ = 0;
  grad_order_ = 0;
  top_cell_switch_counts_ = 0;

  check_graph_cell_id_.clear();
  grad_operation_.clear();
  top_cell_ = nullptr;
  bprop_cell_list_.clear();
  already_run_top_cell_.clear();
  ClearCellRes(py::none());
  std::stack<std::pair<std::string, bool>>().swap(bprop_grad_stack_);
  std::stack<std::string>().swap(cell_stack_);
  std::stack<TopCellInfoPtr>().swap(high_order_stack_);
}

AnfNodePtr GradExecutor::GetInput(const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  AnfNodePtr node = nullptr;
  const auto &obj_id = PyNativeAlgo::Common::GetIdByValue(v);
  const auto &fg = top_cell()->fg();

  if (v->isa<tensor::Tensor>() && v->cast<tensor::TensorPtr>()->is_parameter()) {
    MS_LOG(DEBUG) << "Cell parameters(weights)";
    // get the parameter name from parameter object
    const auto &tensor = v->cast<tensor::TensorPtr>();
    const auto &param_info = tensor->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    const auto &param_name = param_info->name();
    const auto &df_builder = top_cell()->df_builder();
    MS_EXCEPTION_IF_NULL(df_builder);

    auto graph_info = top_cell()->graph_info_map().at(df_builder);
    MS_EXCEPTION_IF_NULL(graph_info);
    if (graph_info->params.find(obj_id) == graph_info->params.end()) {
      auto free_param = df_builder->add_parameter();
      free_param->set_name(param_name);
      free_param->debug_info()->set_name(param_name);
      free_param->set_default_param(tensor);
      MS_LOG(DEBUG) << "Top graph set free parameter " << obj_id;
      top_cell()->SetParamNodeMapInGraphInfoMap(df_builder, obj_id, free_param);
      top_cell()->SetParamNodeMapInGraphInfoMap(fg, obj_id, free_param);
      top_cell()->SetNodeMapInGraphInfoMap(df_builder, obj_id, free_param);
      top_cell()->SetNodeMapInGraphInfoMap(fg, obj_id, free_param);
      return free_param;
    }
    node = graph_info->params.at(obj_id);
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Get input param node " << node->ToString() << ", v id " << obj_id;
    return node;
  }

  auto curr_graph_info = top_cell()->graph_info_map().at(fg);
  MS_EXCEPTION_IF_NULL(curr_graph_info);
  if (curr_graph_info->node_map.find(obj_id) != curr_graph_info->node_map.end()) {
    // op(x, y)
    // out = op(op1(x, y))
    // out = op(cell1(x, y))
    // out = op(cell1(x, y)[0])
    node = GetObjNode(v, obj_id);
  } else if (v->isa<ValueSequence>()) {
    // out = op((x, y))
    // out = cell((x, y))
    auto tuple = v->cast<ValueSequencePtr>();
    // cell((1,2)): support not mix (scalar, tensor)
    if (tuple->size() != 0 && !tuple->value()[0]->isa<tensor::Tensor>()) {
      return MakeValueNode(v, obj_id);
    }
    std::vector<AnfNodePtr> args;
    (void)args.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto tuple_size = tuple->size();
    for (size_t i = 0; i < tuple_size; i++) {
      (void)args.emplace_back(GetInput(tuple->value()[i]));
    }
    auto cnode = fg->NewCNode(args);
    top_cell()->SetNodeMapInGraphInfoMap(fg, obj_id, cnode);
    node = cnode;
  } else {
    node = MakeValueNode(v, obj_id);
  }
  node == nullptr ? MS_LOG(DEBUG) << "Get node is nullptr"
                  : MS_LOG(DEBUG) << "Get input node " << node->ToString() << ", id " << obj_id;
  return node;
}

AnfNodePtr GradExecutor::GetObjNode(const ValuePtr &v, const std::string &obj_id) const {
  MS_EXCEPTION_IF_NULL(v);
  const auto &fg = top_cell()->fg();
  auto graph_info = top_cell()->graph_info_map().at(fg);
  MS_EXCEPTION_IF_NULL(graph_info);
  if (graph_info->node_map.find(obj_id) == graph_info->node_map.end()) {
    // A tuple returns in this case: x = op1, y = op2, return (x, y)
    // or a constant returns in this case
    auto make_tuple = CreateMakeTupleGradNode(v, obj_id);
    if (make_tuple == nullptr) {
      MS_LOG(DEBUG) << "Create value node for obj id: " << obj_id;
      return MakeValueNode(v, obj_id);
    }
    return make_tuple;
  }
  // single output CNode
  const auto &out = graph_info->node_map.at(obj_id);
  if (out.second.size() == 1 && out.second[0] == -1) {
    return out.first;
  }
  // Params node
  if (graph_info->params.find(obj_id) != graph_info->params.end()) {
    auto para_node = out.first;
    for (auto &item : out.second) {
      std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), para_node,
                                                    NewValueNode(item)};
      para_node = fg->NewCNode(tuple_get_item_inputs);
    }
    return para_node;
  }
  // Create tuple get item node for multiple output CNode
  return CreateTupleGetItemNode(obj_id);
}

AnfNodePtr GradExecutor::MakeValueNode(const ValuePtr &v, const std::string &obj_id) const {
  MS_EXCEPTION_IF_NULL(v);
  auto node = NewValueNode(v);
  top_cell()->SetNodeMapInGraphInfoMap(top_cell()->fg(), obj_id, node);
  return node;
}

void GradExecutor::RecordGradNodeToGraphInfoMap(const FuncGraphPtr &fg, const CNodePtr &cnode, const ValuePtr &v,
                                                const std::string &obj_id, const ValuePtrList &input_args) const {
  top_cell()->SetTupleArgsToGraphInfoMap(fg, v, cnode);
  top_cell()->SetNodeMapInGraphInfoMap(fg, obj_id, cnode);
  // run ad for make tuple node
  if (grad_is_running_ && !bprop_grad_stack_.empty() && !bprop_grad_stack_.top().second) {
    MS_LOG(DEBUG) << "Running custom bprop, no need to do GradPynativeOp.";
  } else {
    (void)ad::GradPynativeOp(top_cell()->k_pynative_cell_ptr(), cnode, input_args,
                             std::make_shared<ValueTuple>(input_args));
  }
}

AnfNodePtr GradExecutor::CreateMakeTupleGradNode(const ValuePtr &v, const std::string &obj_id) const {
  const auto &fg = top_cell()->fg();
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(v);
  ValuePtrList input_args;
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
  if (!v->isa<ValueSequence>()) {
    MS_LOG(DEBUG) << "The input obj is not a tuple or list.";
    return nullptr;
  }
  const auto &obj_tuple = v->cast<ValueSequencePtr>();
  const auto &v_list = obj_tuple->value();
  for (size_t i = 0; i < obj_tuple->size(); ++i) {
    const auto &v_arg = v_list[i];
    // Graph have no define for grad
    if (v_arg->isa<FuncGraph>()) {
      continue;
    }
    (void)input_args.emplace_back(v_arg);
    (void)inputs.emplace_back(GetInput(v_arg));
    (void)CreateMakeTupleGradNode(v_arg, PyNativeAlgo::Common::GetIdByValue(v_arg));
  }
  // Create make tuple node and record to graph info map.
  auto cnode = fg->NewCNode(inputs);
  MS_LOG(DEBUG) << "Create make tuple node: " << cnode->DebugString();
  RecordGradNodeToGraphInfoMap(fg, cnode, v, obj_id, input_args);
  return cnode;
}

AnfNodePtr GradExecutor::CreateTupleGetItemNode(const std::string &obj_id) const {
  const auto &fg = top_cell()->fg();
  // obj_id is obtained by calling the 'PyParser::GetIdByPyObj()'
  auto graph_info = top_cell()->graph_info_map().at(fg);
  MS_EXCEPTION_IF_NULL(graph_info);
  if (graph_info->node_map.find(obj_id) == graph_info->node_map.end()) {
    MS_LOG(DEBUG) << "Can not find CNode for obj id: " << obj_id;
    return nullptr;
  }
  const auto &out = graph_info->node_map.at(obj_id);
  MS_LOG(DEBUG) << "Output size: " << out.second.size();
  auto c_node = out.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_node);
  auto abs = c_node->abstract();
  // Create tuple get item node
  for (const auto &idx : out.second) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), c_node, NewValueNode(idx)};
    c_node = fg->NewCNode(tuple_get_item_inputs);
    if (abs != nullptr && abs->isa<abstract::AbstractTuple>()) {
      auto abs_tuple = dyn_cast<abstract::AbstractTuple>(abs);
      MS_EXCEPTION_IF_NULL(abs_tuple);
      const auto &elements = abs_tuple->elements();
      if (static_cast<size_t>(idx) >= elements.size()) {
        MS_LOG(EXCEPTION) << "Index exceeds the size of elements. Index " << idx << ", element size "
                          << elements.size();
      }
      auto prim_abs = elements[static_cast<size_t>(idx)];
      MS_EXCEPTION_IF_NULL(prim_abs);
      MS_LOG(DEBUG) << "Set tuple getitem abs " << prim_abs->ToString();
      c_node->set_abstract(prim_abs);
    }
  }
  MS_LOG(DEBUG) << "Create tuple get item node: " << c_node->DebugString();
  return c_node;
}

TopCellInfoPtr GradExecutor::GetTopCell(const std::string &already_run_cell_id) {
  TopCellInfoPtr find_top_cell = nullptr;
  for (const auto &top_cell : top_cell_list_) {
    MS_EXCEPTION_IF_NULL(top_cell);
    // Complete match, means run grad operation first
    if (top_cell->already_run_cell_id() == already_run_cell_id) {
      return top_cell;
    }
    // Partial match, means run forward first
    if (already_run_cell_id.find(top_cell->already_run_cell_id()) != std::string::npos &&
        top_cell->already_run_cell_id().back() == '_') {
      find_top_cell = top_cell;
      break;
    }
  }
  // Same topcell info, but grad operation is not the same, construct backward graph again
  if (find_top_cell != nullptr) {
    if (!find_top_cell->grad_operation().empty() && find_top_cell->grad_operation() != grad_operation_) {
      MS_LOG(DEBUG) << "Already exist grad operation " << find_top_cell->grad_operation() << " is different with new "
                    << grad_operation_;
      EraseTopCellFromTopCellList(find_top_cell);
      (void)already_run_top_cell_.erase(find_top_cell->already_run_cell_id());
      return nullptr;
    } else {
      return find_top_cell;
    }
  }
  return nullptr;
}

void GradExecutor::EnableOpGraphCache(bool is_enable) {
  MS_LOG(DEBUG) << "Op cache is enable: " << is_enable;
  enable_op_cache_ = is_enable;
  const auto inst = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  inst->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE, is_enable);
}

void GradExecutor::SetHookChanged(const py::object &cell) const {
  auto cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  for (const auto &top_cell : top_cell_list_) {
    MS_EXCEPTION_IF_NULL(top_cell);
    if (top_cell->cell_id().find(cell_id) != std::string::npos) {
      top_cell->set_hook_changed(true);
    }
    const auto &sub_cells = top_cell->sub_cell_list();
    for (const auto &sub_cell_id : sub_cells) {
      if (sub_cell_id.find(cell_id) != std::string::npos) {
        top_cell->set_hook_changed(true);
      }
    }
  }
  if (need_construct_graph() && top_cell_ != nullptr) {
    top_cell_->set_sub_cell_hook_changed(cell_id);
  }
}

void GradExecutor::ProcessOpGradInfo(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  // Get output value
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  // Record op info for judge whether the construct of cell has been changed
  MS_EXCEPTION_IF_NULL(op_run_info);
  top_cell()->RecordGradOpInfo(op_run_info);
  // Const value no need do op grad
  if (op_run_info->output_get_by_infer_value) {
    return;
  }
  // Do op grad and save node info
  if (need_construct_graph() && custom_bprop_cell_count_ <= 0) {
    const auto &cnode = ConstructForwardGraph(op_run_info);
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &obj_id = PyNativeAlgo::Common::GetIdByValue(v);
    cnode->set_abstract(op_run_info->base_op_run_info.abstract);
    SaveOutputNodeMap(obj_id, v, cnode);
    // Dynamic shape should update to top cell
    if (PyNativeAlgo::Common::IsDynamicShape(op_run_info)) {
      top_cell()->set_dynamic_shape(true);
    }
    DoOpGrad(op_run_info, cnode, v);
  }
  forward()->SetNodeAbsMapByValue(v, op_run_info->base_op_run_info.abstract);
  UpdateForwardTensorInfoInBpropGraph(op_run_info->op_info, v);
}

void GradExecutor::SaveOutputNodeMap(const std::string &obj_id, const ValuePtr &v, const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(v);
  if (cell_stack_.empty()) {
    MS_LOG(DEBUG) << "No need save output";
    return;
  }
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Cnode is " << cnode->DebugString() << ", out value id " << obj_id;
  if (v->isa<ValueSequence>()) {
    auto value = v->cast<ValueSequencePtr>();
    auto size = static_cast<int64_t>(value->size());
    if (size > 1) {
      for (int64_t i = 0; i < size; ++i) {
        auto value_id = PyNativeAlgo::Common::GetIdByValue(value->value()[static_cast<size_t>(i)]);
        top_cell()->SetNodeMapInGraphInfoMap(curr_g(), value_id, cnode, i);
      }
    }
  }
  top_cell()->SetNodeMapInGraphInfoMap(curr_g(), obj_id, cnode);
}

bool GradExecutor::ConvertTupleAndScalarIntoTensor(const FrontendOpRunInfoPtr &op_run_info, ValuePtrList *input_args,
                                                   size_t idx, const ValuePtr &default_value) const {
  if (top_cell()->dynamic_shape() &&
      kDynamicInputOpMap.find(op_run_info->base_op_run_info.op_name) != kDynamicInputOpMap.end()) {
    const auto &input_vec = kDynamicInputOpMap[op_run_info->base_op_run_info.op_name];
    bool marked = std::any_of(input_vec.begin(), input_vec.end(), [&idx](size_t i) { return idx == i; });
    if (marked) {
      if (default_value->isa<ValueSequence>()) {
        MS_LOG(DEBUG) << "Ready to convert tulpe into tensor, op name:" << op_run_info->base_op_run_info.op_name
                      << ", index:" << idx;
        ValueSequencePtr value_seq = default_value->cast<ValueSequencePtr>();
        ValueTuplePtr value_tuple;
        if (value_seq->isa<ValueList>()) {
          value_tuple = std::make_shared<ValueTuple>(value_seq->value());
        } else {
          value_tuple = value_seq->cast<ValueTuplePtr>();
        }
        auto tensor_ptr = opt::CreateTupleTensor(value_tuple);
        (*input_args)[idx] = tensor_ptr;
        return true;
      } else if (default_value->isa<Scalar>()) {
        MS_LOG(DEBUG) << "Ready to convert scalar into tensor, op name:" << op_run_info->base_op_run_info.op_name
                      << ", index:" << idx;
        auto scalar_tensor = ScalarToTensor(default_value->cast<ScalarPtr>());
        (*input_args)[idx] = scalar_tensor;
        return true;
      }
    }
  }
  return false;
}

// Run ad grad for curr op and connect grad graph with previous op
void GradExecutor::DoOpGrad(const FrontendOpRunInfoPtr &op_run_info, const CNodePtr &cnode,
                            const ValuePtr &op_out) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_out);
  if (grad_is_running_ && !bprop_grad_stack_.top().second) {
    MS_LOG(DEBUG) << "Custom bprop, no need do op grad";
    return;
  }

  ValuePtrList input_args;
  input_args.resize(op_run_info->input_value.size(), nullptr);
  // Run in Vm, inputs not convert to tensor object, so need do transform it
  if (op_run_info->run_in_vm) {
    input_args = op_run_info->input_value;
  } else {
    for (size_t i = 0; i < op_run_info->input_value.size(); ++i) {
      if (enable_tuple_to_tensor_ &&
          ConvertTupleAndScalarIntoTensor(op_run_info, &input_args, i, op_run_info->input_value[i])) {
        continue;
      }
      input_args[i] = op_run_info->input_value[i];
    }
  }
  if (op_run_info->base_op_run_info.has_dynamic_output) {
    dynamic_shape()->UpdateValueToDynamicShape(op_out);
  }

  if (!ad::GradPynativeOp(top_cell()->k_pynative_cell_ptr(), cnode, input_args, op_out)) {
    MS_LOG(EXCEPTION) << "Failed to run ad grad for op " << op_run_info->base_op_run_info.op_name;
  }
}

void GradExecutor::UpdateTensorInfo(const tensor::TensorPtr &new_tensor,
                                    const std::vector<tensor::TensorPtr> &pre_tensors) const {
  MS_EXCEPTION_IF_NULL(new_tensor);
  if (pre_tensors.empty() || new_tensor->device_address() == nullptr) {
    MS_LOG(DEBUG) << "The number of pre tensors is zero or the device address of new tensor is nullptr.";
    return;
  }
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  for (auto &pre_tensor : pre_tensors) {
    MS_EXCEPTION_IF_NULL(pre_tensor);
    MS_LOG(DEBUG) << "Replace Old tensor id " << pre_tensor->id() << " device_address: " << pre_tensor->device_address()
                  << " shape and type " << pre_tensor->GetShapeAndDataTypeInfo() << " with New tensor id "
                  << new_tensor->id() << " device_address " << new_tensor->device_address() << " shape and dtype "
                  << new_tensor->GetShapeAndDataTypeInfo();
    (void)pre_tensor->set_shape(new_tensor->shape());
    (void)pre_tensor->set_data_type(new_tensor->data_type());
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(new_tensor->device_address());
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_target != kCPUDevice && device_address->GetDeviceType() != device::DeviceType::kCPU) {
      pre_tensor->set_device_address(new_tensor->device_address());
      continue;
    }
    for (const auto &item : forward()->mindrt_backend()) {
      MS_EXCEPTION_IF_NULL(item.second);
      item.second->WaitTaskFinish();
    }
    // Replace data in device address when run in CPU device.
    if (pre_tensor->device_address() != nullptr) {
      // If tensor is dynamic shape, Just replace device address.
      if (PyNativeAlgo::Common::ValueHasDynamicShape(pre_tensor)) {
        pre_tensor->set_device_address(new_tensor->device_address());
        continue;
      }
      auto old_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(pre_tensor->device_address());
      MS_EXCEPTION_IF_NULL(old_device_address);
      auto new_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(new_tensor->device_address());
      MS_EXCEPTION_IF_NULL(new_device_address);

      // CPU host tensor data_c is different from device address if the address is from mem_pool.
      if (new_device_address->from_mem_pool()) {
        pre_tensor->set_device_address(new_device_address);
        continue;
      }

      auto old_ptr = old_device_address->GetMutablePtr();
      MS_EXCEPTION_IF_NULL(old_ptr);
      auto new_ptr = new_device_address->GetPtr();
      MS_EXCEPTION_IF_NULL(new_ptr);
      MS_EXCEPTION_IF_CHECK_FAIL(old_device_address->GetSize() == new_device_address->GetSize(), "Size not equal");
      if (old_device_address->GetSize() < SECUREC_MEM_MAX_LEN) {
        auto ret_code = memcpy_s(old_ptr, old_device_address->GetSize(), new_ptr, new_device_address->GetSize());
        MS_EXCEPTION_IF_CHECK_FAIL(ret_code == EOK, "Memory copy failed, ret code: " + std::to_string(ret_code));
      } else {
        auto ret_code = std::memcpy(old_ptr, new_ptr, old_device_address->GetSize());
        MS_EXCEPTION_IF_CHECK_FAIL(ret_code == old_ptr, "Memory copy failed");
      }
    } else {
      pre_tensor->set_device_address(device_address);
      pre_tensor->data_sync();
      pre_tensor->set_device_address(nullptr);
      pre_tensor->set_sync_status(kNeedSyncHostToDevice);
    }
  }
}

void GradExecutor::UpdateForwardTensorInfoInBpropGraph(const string &op_info, const ValuePtr &op_out) const {
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "The grad flag is false, no need to update forward op info in bprop graph";
    return;
  }
  MS_EXCEPTION_IF_NULL(op_out);
  MS_LOG(DEBUG) << "Current op info: " << op_info;
  std::vector<tensor::TensorPtr> all_op_tensors;
  // Get output tensors
  TensorValueToTensor(op_out, &all_op_tensors);
  // Save all tensors info of current op
  if (need_construct_graph()) {
    top_cell()->SaveOpInfo(op_info, all_op_tensors);
  }

  // First run top cell
  if (already_run_top_cell_.find(top_cell_->already_run_cell_id()) == already_run_top_cell_.end()) {
    MS_LOG(DEBUG) << "Top cell " << top_cell_->cell_id() << " run firstly";
    if (!need_construct_graph()) {
      MS_LOG(EXCEPTION) << "The cell stack is empty when running a new top cell " << top_cell_->cell_id();
    }
    return;
  }
  // Non-first run
  const auto &pre_top_cell = already_run_top_cell_.at(top_cell_->already_run_cell_id());
  MS_EXCEPTION_IF_NULL(pre_top_cell);
  if (pre_top_cell->op_info_with_tensor_id().find(op_info) == pre_top_cell->op_info_with_tensor_id().end()) {
    MS_LOG(DEBUG) << "Can not find op info " << op_info << " in op info with tensor id map. Top cell "
                  << top_cell_->cell_id();
    return;
  }

  // Update new output tensor info in bprop graph
  const auto &pre_op_tensor_id = pre_top_cell->op_info_with_tensor_id().at(op_info);
  if (pre_op_tensor_id.size() != all_op_tensors.size()) {
    MS_LOG(EXCEPTION) << "The size of pre op tensor id: " << pre_op_tensor_id.size()
                      << " is not equal to the size of all tensors of current op " << all_op_tensors.size();
  }
  const auto &pre_tensor_id_with_tensor_object = pre_top_cell->tensor_id_with_tensor_object();
  for (size_t i = 0; i < pre_op_tensor_id.size(); ++i) {
    auto pre_id = pre_op_tensor_id[i];
    if (pre_tensor_id_with_tensor_object.find(pre_id) == pre_tensor_id_with_tensor_object.end()) {
      continue;
    }
    const auto &new_tensor = all_op_tensors[i];
    const auto &pre_tensor_object = pre_tensor_id_with_tensor_object.at(pre_id);
    UpdateTensorInfo(new_tensor, pre_tensor_object);
  }
}

void GradExecutor::SaveForwardTensorInfoInBpropGraph(const pipeline::ResourcePtr &resource) const {
  MS_EXCEPTION_IF_NULL(resource);
  // Get all tensors id of forward op
  mindspore::HashSet<std::string> forward_op_tensor_id;
  const auto &op_info_with_tensor_id = top_cell()->op_info_with_tensor_id();
  for (const auto &record : op_info_with_tensor_id) {
    (void)std::for_each(
      record.second.begin(), record.second.end(),
      [&forward_op_tensor_id](const std::string &tensor_id) { (void)forward_op_tensor_id.emplace(tensor_id); });
  }
  // Get all tensors obj in value node of bprop graph
  const auto &bprop_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(bprop_graph);
  const auto &value_node_list = bprop_graph->value_nodes();
  std::vector<tensor::TensorPtr> tensors_in_bprop_graph;
  for (const auto &elem : value_node_list) {
    auto value_node = elem.first->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    TensorValueToTensor(value_node->value(), &tensors_in_bprop_graph);
  }

  // Save tensor in value node of bprop graph
  for (const auto &tensor : tensors_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (forward_op_tensor_id.find(tensor->id()) == forward_op_tensor_id.end() || tensor->device_address() == nullptr) {
      continue;
    }
    tensor->set_is_forward_output(true);
    top_cell()->SetTensorIdWithTensorObject(tensor->id(), tensor);
    MS_LOG(DEBUG) << "Save forward tensor " << tensor.get() << " id " << tensor->id()
                  << " device address: " << tensor->device_address() << " shape and dtype "
                  << tensor->GetShapeAndDataTypeInfo();
  }
}

AnfNodePtr GradExecutor::GetRealInputNodeBySkipHook(const AnfNodePtr &input_node) const {
  if (input_node == nullptr) {
    MS_LOG(DEBUG) << "The input node is nullptr.";
    return input_node;
  }
  const auto &cell_backward_hook_op = top_cell()->cell_backward_hook_op();
  for (const auto &elem : cell_backward_hook_op) {
    constexpr size_t cell_backward_hook_num = 2;
    if (elem.second.size() < cell_backward_hook_num) {  // In cell own scope, no need to skip backward hook op.
      continue;
    }
    // The input node is the first backward hook op of another cell, skip the backward hook op.
    if (IsPrimitiveCNode(input_node, prim::kPrimCellBackwardHook) && input_node == elem.second[0]) {
      // Single input.
      auto backward_hook_op = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(backward_hook_op);
      return backward_hook_op->input(1);
    } else if (IsPrimitiveCNode(input_node, prim::kPrimTupleGetItem)) {
      // Multi inputs.
      auto tuple_get_item = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_get_item);
      auto inp_in_tuple = tuple_get_item->input(1);
      MS_EXCEPTION_IF_NULL(inp_in_tuple);
      if (IsPrimitiveCNode(inp_in_tuple, prim::kPrimCellBackwardHook) && inp_in_tuple == elem.second[0]) {
        constexpr size_t idx = 2;
        auto idx_node = tuple_get_item->input(idx);
        MS_EXCEPTION_IF_NULL(idx_node);
        auto value_node = idx_node->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        auto out_idx = GetValue<int64_t>(value_node->value());
        auto backward_hook_op = inp_in_tuple->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(backward_hook_op);
        return backward_hook_op->input(1 + LongToSize(out_idx));
      }
    }
  }
  return input_node;
}

CNodePtr GradExecutor::ConstructForwardGraph(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  size_t input_size = op_run_info->input_value.size();
  std::vector<AnfNodePtr> inputs;
  (void)inputs.emplace_back(NewValueNode(op_run_info->op_prim));
  for (size_t i = 0; i < input_size; i++) {
    AnfNodePtr input_node = nullptr;
    const auto node = GetInput(op_run_info->input_value[i]);
    input_node = GetRealInputNodeBySkipHook(node);
    // update abstract
    if (input_node != nullptr) {
      (void)inputs.emplace_back(input_node);
    }
  }
  const auto &cnode = top_cell()->fg()->NewCNodeInOrder(inputs);
  if (IsPrimitiveCNode(cnode, prim::kPrimCellBackwardHook)) {
    top_cell()->RecordCellBackwardHookOp(GetCurCellOrder(), cnode);
  }
  MS_LOG(DEBUG) << "Make CNode for " << op_run_info->base_op_run_info.op_name << ", new cnode is "
                << cnode->DebugString();
  return cnode;
}

void GradExecutor::SetBpropGraphJitLevel(const py::object &cell) const {
  if (!py::hasattr(cell, kAttrCellJitConfigDict)) {
    return;
  }

  auto jit_config = py::getattr(cell, kAttrCellJitConfigDict);
  if (!py::isinstance<py::dict>(jit_config)) {
    MS_LOG(EXCEPTION) << "JitConfig only support dict!";
  }
  auto jit_config_dict = jit_config.cast<py::dict>();
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  graph_executor->SetJitConfig(jit_config_dict);
}
}  // namespace pynative
}  // namespace mindspore
