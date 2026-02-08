/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "pynative/backward/grad.h"

#include <algorithm>
#include <unordered_set>

#include "ir/cell.h"
#include "ir/func_graph_cloner.h"
#include "ir/value.h"
#include "ir/tensor_new.h"
#include "include/frontend/jit/ps/pass_interface.h"
#include "include/frontend/jit/ps/resource_interface.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"
#include "include/frontend/jit/ps/debug/trace_interface.h"
#include "include/frontend/optimizer/ad/grad_interface.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/convert_utils_py.h"
#include "include/utils/tensor_py.h"
#include "include/utils/pynative/common_utils.h"
#include "include/utils/pipeline/pipeline_utils.h"
#include "include/frontend/jit/ps/parse/py_data_convert.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pynative/backward/hook/custom_function.h"
#include "pynative/backward/top_cell.h"
#include "pynative/backward/op_grad/func_grad.h"
#include "pynative/backward/grad_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "utils/log_adapter.h"
#include "include/pynative/utils/pyboost/grad_functions/pyboost_grad_functions.h"
#include "include/pynative/utils/pyboost/grad_functions/value_converter.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "primitive/conv_pool_op_name.h"
#include "primitive/nn_op_name.h"
#include "primitive/math_op_name.h"
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace pynative {
namespace {
const mindspore::HashSet<std::string> kHookOp = {"HookBackward", "CellBackwardHook"};
constexpr char kGrad[] = "grad";
constexpr auto kNeedRecompute = "is_cell_recompute";
constexpr auto kInternalParams = "internal_params";
constexpr auto kUsedBpropInputs = "used_bprop_inputs";
constexpr auto kCreateGraph = "create_graph";
constexpr size_t kContainerRatio = 2;

void ParsePyArgsToInputArgsInfo(const InputArgsInfoPtr &input_args_info, const py::object &obj, const py::args &args) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  input_args_info->has_custom_bprop = py::hasattr(obj, parse::CUSTOM_BPROP_NAME);
  MS_LOG(DEBUG) << "Cell has custom bprop " << input_args_info->has_custom_bprop;
  bool is_top_cell = input_args_info->is_grad_topest_cell || input_args_info->is_high_order_top_cell ||
                     input_args_info->is_inner_grad_topest_cell;
  // Only the top cell or custom bprop cell requires value conversion
  if (is_top_cell) {
    pipeline::CheckArgsValid(obj, args);
    input_args_info->input_size = args.size();
    for (size_t i = 0; i < input_args_info->input_size; ++i) {
      const auto &id = PyNativeAlgo::PyParser::GetIdByPyObj(args[i]);
      (void)input_args_info->input_arg_id_vec.emplace_back(id);
    }
    const auto &forward = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor();
    for (size_t i = 0; i < input_args_info->input_size; ++i) {
      input_args_info->input_args_id += input_args_info->input_arg_id_vec[i] + "_";
      // Get arg value
      if (py::isinstance<py::list>(args[i])) {
        (void)input_args_info->input_arg_value_vec.emplace_back(
          parse::data_converter::PyObjToValue(py::cast<py::tuple>(args[i])));
      } else {
        (void)input_args_info->input_arg_value_vec.emplace_back(parse::data_converter::PyObjToValue(args[i]));
      }

      // Get arg abstract
      auto abs = forward->GetNodeAbsById(input_args_info->input_arg_id_vec[i]);
      if (abs == nullptr) {
        abs = input_args_info->input_arg_value_vec.back()->ToAbstract();
      }
      (void)input_args_info->input_arg_base_shape_vec.emplace_back(abs->BuildShape());
    }
    input_args_info->cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
    MS_LOG(DEBUG) << "Cell_id is " << input_args_info->cell_id << ", is grad topest cell "
                  << input_args_info->is_grad_topest_cell << ", is high order top cell "
                  << input_args_info->is_high_order_top_cell << ", is bprop need get forward graph ";
  }
}

AnfNodePtr GetNonTensorInput(const ValuePtr &v, const std::string &obj_id) {
  MS_EXCEPTION_IF_NULL(v);
  bool is_value_seq = v->isa<ValueSequence>();
  bool is_single_non_tensor = !is_value_seq && !PyNativeAlgo::Common::IsTensor(v);
  bool mixed_tensor = true;
  if (is_value_seq) {
    const auto &v_seq = v->cast<ValueSequencePtr>();
    mixed_tensor = std::any_of(v_seq->value().begin(), v_seq->value().end(),
                               [](const ValuePtr &e) { return PyNativeAlgo::Common::IsTensor(e, true); });
  }
  if (is_single_non_tensor || !mixed_tensor) {
    auto v_node = PyNativeAlgo::Common::CreateValueNodeByValue(v);
    MS_LOG(DEBUG) << "Get input value node " << v_node->ToString() << ", id " << obj_id;
    return v_node;
  }
  return nullptr;
}

ValuePtr ConvertOutputValueToTensor(const ValuePtr &v, bool dict_convert_to_tuple) {
  MS_EXCEPTION_IF_NULL(v);
  if (PyNativeAlgo::Common::IsTensor(v, true)) {
    return v;
  }
  if (v->isa<ValueSequence>()) {
    auto v_seq = v->cast<ValueSequencePtr>();
    if (v_seq->size() == 0) {
      MS_LOG(EXCEPTION) << "Get empty value seq";
    }
    // All value are tensor
    if (std::all_of(v_seq->value().begin(), v_seq->value().end(),
                    [](const ValuePtr &e) { return PyNativeAlgo::Common::IsTensor(e, true); })) {
      MS_LOG(DEBUG) << "All output value is tensor";
      return v;
    }
    MS_LOG(DEBUG) << "Output is value sequence, but have tensor and other type mixed. Its value is " << v->ToString();
    return PyNativeAlgo::Common::FilterSensValues(v, dict_convert_to_tuple);
  }
  if (v->isa<FloatImm>()) {
    double input_value = v->cast<FP32ImmPtr>()->value();
    return tensor::from_scalar(input_value, kFloat32);
  }
  if (v->isa<BoolImm>()) {
    return tensor::from_scalar(v->cast<BoolImmPtr>()->value(), kBool);
  }
  if (v->isa<IntegerImm>()) {
    int64_t input = v->cast<Int64ImmPtr>()->value();
    return tensor::from_scalar(input, kInt64);
  }
  if (v->isa<ValueDictionary>() && dict_convert_to_tuple) {
    MS_LOG(DEBUG) << "Get dict value";
    return PyNativeAlgo::DataConvert::ConvertValueDictToValueTuple(v);
  }
  MS_LOG(DEBUG) << "Output is " << v->ToString() << ", abstract "
                << CommonUtils::SetAbstractValueToAnyValue(v->ToAbstract());
  return v;
}

void SetSensValue(const prim::GradOperationPtr &grad, const InputArgsInfoPtr &input_args_info, const py::args &args,
                  bool dict_convert_to_tuple) {
  MS_EXCEPTION_IF_NULL(grad);
  if (!grad->sens_param()) {
    return;
  }
  size_t forward_args_size = args.size() - 1;
  auto sens_v = parse::data_converter::PyObjToValue(args[forward_args_size]);
  MS_LOG(DEBUG) << "Get sens param " << sens_v->ToString();
  const auto &sens_tensor = ConvertOutputValueToTensor(sens_v, dict_convert_to_tuple);
  if (sens_tensor == nullptr) {
    MS_LOG(EXCEPTION) << "sens convert tensor is nullptr";
  }
  // Sens have already existed, which may be need update
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (input_args_info->input_arg_value_vec.size() == args.size()) {
    input_args_info->input_arg_value_vec.pop_back();
  }
  (void)input_args_info->input_arg_value_vec.emplace_back(sens_tensor);
  if (sens_tensor->isa<ValueSequence>()) {
    input_args_info->sens_type = SensType::kTuple;
  } else if (!dict_convert_to_tuple) {
    input_args_info->sens_type = SensType::kDict;
  }
}

GradParamPtr CreateOpGradParam(const OpGradInfoPtr &grad_info) {
  auto grad_param = std::make_shared<GradParam>(grad_info);
  BpropExpander::FreeUselessValues(BpropCallback(grad_info->op_prim, &grad_info->input_value, &grad_info->out_value));
  return grad_param;
}

std::string GetInputArgsId(const py::args &args) {
  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) + "_";
  }
  return input_args_id;
}

void SetCustomBpropInputs(const py::object &obj, autograd::CustomContext *context) {
  if (py::hasattr(obj, kUsedBpropInputs)) {
    py::object object = py::getattr(obj, kUsedBpropInputs);
    if (!py::isinstance<py::tuple>(object) && !py::isinstance<py::list>(object)) {
      MS_LOG(EXCEPTION) << "For cell bprop, used bprop inputs sholud be tuple or list";
    }
    auto used_bprop_inputs = py::cast<py::tuple>(object);
    std::unordered_set<int64_t> used_inputs;
    for (size_t i = 0; i < used_bprop_inputs.size(); ++i) {
      if (!py::isinstance<py::int_>(used_bprop_inputs[i])) {
        MS_LOG(EXCEPTION) << "For cell bprop, element of used bprop inputs should be int type!";
      }
      int64_t used_index = py::cast<int64_t>(used_bprop_inputs[i]);
      (void)used_inputs.insert(used_index);
    }
    const size_t input_size = context->inputs.size();
    for (size_t i = 0; i < input_size; ++i) {
      const auto &input_value = context->inputs[i];
      if (used_inputs.find(static_cast<int64_t>(i)) == used_inputs.end()) {
        auto fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(input_value);
        context->inputs[i] = fake_value;
        MS_LOG(DEBUG) << "Clear input value" << i << "device address";
      }
    }
    if (used_inputs.find(static_cast<int64_t>(input_size)) == used_inputs.end()) {
      const auto &fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(context->output);
      context->output = fake_value;
      MS_LOG(DEBUG) << "Clear output value device address";
    }

    context->used_inputs_set = std::move(used_inputs);
  }

  if (py::hasattr(obj, kInternalParams)) {
    py::object weights = py::getattr(obj, kInternalParams);
    if (py::isinstance<py::tuple>(weights) || py::isinstance<py::list>(weights)) {
      auto weights_tuple = py::cast<py::tuple>(weights);
      context->weight_size = weights_tuple.size();
      for (size_t i = 0; i < weights_tuple.size(); ++i) {
        if (!tensor::IsTensorPy(weights_tuple[i])) {
          MS_LOG(EXCEPTION) << "For cell bprop, element of internal params should be tensor type!";
        }
        auto tensor = tensor::ConvertToTensor(weights_tuple[i]);
        (void)context->inputs.emplace_back(tensor);
        (void)context->input_value_grad_type.emplace_back(AutoGradUtil::SetValueGradInfo(tensor, InputType::kConstant));
      }
    }
  }
}

std::string PrintPyObjInfo(const py::object &obj, const std::string &str) {
  std::ostringstream buf;
  if (py::isinstance<Cell>(obj)) {
    buf << str << " run " << obj.cast<CellPtr>()->ToString();
    return buf.str();
  }
  buf << str << " run python function " << py::getattr(obj, "__name__").cast<std::string>();
  return buf.str();
}

bool CheckBpropWithJit(const py::function &bprop_func) {
  py::object code_obj = py::getattr(bprop_func, "__code__");
  py::object co_name = py::getattr(code_obj, "co_name");
  if (std::string(py::str(co_name)) == "staging_specialize") {
    MS_LOG(EXCEPTION) << "Decorating bprop with '@jit' is not supported.";
  }
  return true;
}

FrontendOpRunInfoPtr CustomContext2OpRunInfo(const autograd::CustomContext &context) {
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->base_op_run_info.op_name = prim::kPrimCellBackwardHook->name();
  op_run_info->op_grad_info->op_prim = prim::kPrimCellBackwardHook;
  op_run_info->op_grad_info->input_value = context.inputs;
  op_run_info->op_grad_info->weight_size = context.weight_size;
  op_run_info->op_grad_info->is_need_recompute = context.is_recompute;
  op_run_info->input_size = context.inputs.size();
  op_run_info->real_out = context.output;
  op_run_info->base_op_run_info.abstract = CommonUtils::SetAbstractValueToAnyValue(context.output->ToAbstract());
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);
  op_run_info->op_grad_info->out_value = context.output;
  op_run_info->op_grad_info->out_abs = op_run_info->base_op_run_info.abstract;
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    const auto &value = context.inputs[i];
    (void)op_run_info->op_grad_info->input_abs.emplace_back(
      CommonUtils::SetAbstractValueToAnyValue(value->ToAbstract()));
  }
  op_run_info->op_grad_info->input_value_grad_type = context.input_value_grad_type;
  return op_run_info;
}

void AsyncClearEngine(const std::shared_ptr<autograd::AutoDiff> &engine) {
  const auto &pynative_executor = PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto &forward_executor = pynative_executor->forward_executor();
  if (forward_executor->enable_async()) {
    auto task = [engine]() { engine->Clear(); };
    runtime::Pipeline::Get().bprop_stage()->Push(std::make_shared<BpropTask>(std::move(task)));
  } else {
    engine->Clear();
  }
}

ValuePtr CheckAndUpdateOutput(const ValuePtr &out, const ValuePtrList &inputs, bool *is_return_self) {
  std::unordered_set<Value *> inputs_set;
  inputs_set.reserve(inputs.size());
  for (const auto &input : inputs) {
    (void)inputs_set.insert(input.get());
  }
  if (out->isa<tensor::Tensor>() && inputs_set.find(out.get()) != inputs_set.end()) {
    MS_LOG(DEBUG) << "Cell custom bprop return self";
    *is_return_self = true;
    return AutoGradUtil::ViewAsSelfWithNoGrad(out->cast<tensor::TensorPtr>());
  }
  if (out->isa<ValueSequence>()) {
    auto out_seq = out->cast<ValueSequencePtr>();
    std::vector<ValuePtr> res;
    res.reserve(out_seq->size());
    for (const auto &out_val : out_seq->value()) {
      if (out_val->isa<tensor::Tensor>() && inputs_set.find(out_val.get()) != inputs_set.end()) {
        *is_return_self = true;
        MS_LOG(DEBUG) << "Cell custom bprop multi output return self";
        (void)res.emplace_back(AutoGradUtil::ViewAsSelfWithNoGrad(out_val->cast<tensor::TensorPtr>()));
      } else {
        (void)res.emplace_back(out_val);
      }
    }
    return is_return_self ? std::make_shared<ValueTuple>(res) : out;
  }
  return out;
}
}  // namespace

ForwardExecutorPtr GradExecutor::forward() const {
  auto forward_executor = forward_executor_.lock();
  MS_EXCEPTION_IF_NULL(forward_executor);
  return forward_executor;
}

void GradExecutor::Init() {
  if (init_) {
    return;
  }
#ifdef _MSC_VER
  static WinBpropRegister reg;
  reg.DoNothing();
  MS_LOG(DEBUG) << "Do windows bprop expander register";
#endif
  init_ = true;
}

TopCellInfoPtr GradExecutor::PopTopCellStack() {
  if (top_cell_stack_.empty()) {
    MS_LOG(DEBUG) << "Stack top cell stack is empty";
    return nullptr;
  }
  MS_LOG(DEBUG) << "Pop top cell " << top_cell_stack_.top() << " on top cell stack";
  top_cell_stack_.pop();
  TopCellInfoPtr top_cell = nullptr;
  if (!top_cell_stack_.empty()) {
    top_cell = top_cell_stack_.top();
  }
  top_cell == nullptr ? MS_LOG(DEBUG) << "Top cell stack has no top cell"
                      : MS_LOG(DEBUG) << "After pop, top cell stack size " << top_cell_stack_.size();
  return top_cell;
}

void GradExecutor::PushInputArgsInfoStack(const InputArgsInfoPtr &input_args_info) {
  input_args_info_stack_.push(input_args_info);
}

void GradExecutor::PopInputArgsInfoStack() {
  if (input_args_info_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack input_args_info_stack_ is empty";
  }
  input_args_info_stack_.pop();
}

void GradExecutor::HandleInputArgsForTopCell(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  const auto &input_value = input_args_info->input_arg_value_vec;
  if (input_args_info->input_size != 0 && input_value.empty()) {
    MS_LOG(EXCEPTION) << "Input value is empty";
  }

  for (size_t i = 0; i < input_args_info->input_size; ++i) {
    const auto &v = input_value[i];
    auto tensor = PyNativeAlgo::Common::GetTensorFromSparseTensor(v);
    if (tensor != nullptr) {
      if (tensor->auto_grad_meta_data() != nullptr && autograd::impl::GetUnsafeGradNodeImpl(tensor) == nullptr) {
        tensor->auto_grad_meta_data()->set_input_type(InputType::kInput);
      }
      (void)AutoGradUtil::SetValueGradInfo(tensor, InputType::kInput);
    }
    RecordForwardGraphForInput(v, input_args_info->input_arg_id_vec[i]);
  }
}

void GradExecutor::InitResourceAndDfBuilder(const InputArgsInfoPtr &input_args_info) {
  MS_LOG(DEBUG) << "InitResourceAndDfBuilder";
  MS_EXCEPTION_IF_NULL(input_args_info);
  forward()->WaitForwardTask();
  // Because bprop task will not clear now, just not wait bprop task.
  if (input_args_info->is_grad_topest_cell) {
    MS_LOG(DEBUG) << "Make new topest graph";
    MakeNewTopCell(input_args_info);
  } else if (input_args_info->is_high_order_top_cell) {
    MS_LOG(DEBUG) << "Nested grad graph existed in construct";
    // High order need wait bprop, because back-up grad info may conflict with first grad.
    WaitBpropTask();
    MakeNewTopCell(input_args_info);
  } else if (input_args_info->is_inner_grad_topest_cell) {
    MS_LOG(DEBUG) << "Make new topest inner graph";
    MakeNewTopCell(input_args_info);
  }
  auto graph_info_cg = std::make_shared<PyNGraphInfo>();
  top_cell_->SetGraphInfoMap(curr_g(), graph_info_cg);
  HandleInputArgsForTopCell(input_args_info);
}

void GradExecutor::NewGraphInner(const py::object &obj, const py::args &args) {
  const auto input_args_info = GetInputArgsInfo(obj, args);
  PushInputArgsInfoStack(input_args_info);
  MS_LOG(DEBUG) << PrintPyObjInfo(obj, "Begin") << ", NewGraphInner start " << args.size() << ", cell_id "
                << PyNativeAlgo::PyParser::GetIdByPyObj(obj) << ", is custom bprop "
                << input_args_info->has_custom_bprop << ", input args info ptr " << input_args_info.get();
  // Make top graph and init resource
  if (input_args_info->is_grad_topest_cell || input_args_info->is_high_order_top_cell ||
      input_args_info->is_inner_grad_topest_cell) {
    InitResourceAndDfBuilder(input_args_info);
  }
}

InputArgsInfoPtr GradExecutor::GetInputArgsInfo(const py::object &obj, const py::args &args) {
  bool is_high_order = IsHighOrderTopCell();
  bool is_grad_topest_top_cell = input_args_info_stack_.empty();
  bool is_inner_grad_topest_top_cell = !is_grad_topest_top_cell && !is_high_order;
  const auto &input_args_info =
    std::make_shared<InputArgsInfo>(is_grad_topest_top_cell, is_inner_grad_topest_top_cell, is_high_order);
  ParsePyArgsToInputArgsInfo(input_args_info, obj, args);

  // CheckAlready run first, grad_order_ will increase 1(highorder scenario)
  // If NetA.set_grad(), so come here first, CheckAlready run later, so grad_order_ need increase 1
  if (input_args_info->is_grad_topest_cell || input_args_info->is_high_order_top_cell ||
      input_args_info->is_inner_grad_topest_cell) {
    if (grad_order_ == 0) {
      IncreaseGradOrder();
    }
    input_args_info->ready_run_cell_id = GetReadyRunCellId(input_args_info->cell_id, input_args_info->input_args_id);
    MS_LOG(DEBUG) << "Get already run top cell id " << input_args_info->ready_run_cell_id;
    // top_input_args_info_ indicate current running cell info
    top_input_args_info_ = input_args_info;
  }
  return input_args_info;
}

void GradExecutor::MakeNewTopCell(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);

  auto fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name("pynative_forward_graph");

  finded_top_cell_ = nullptr;
  top_cell_ = std::make_shared<TopCellInfo>(input_args_info->is_high_order_top_cell, grad_order_,
                                            input_args_info->cell_id, input_args_info->ready_run_cell_id, nullptr, fg,
                                            op_num_in_bprop_graph_ * kContainerRatio);
  top_cell_->set_input_args_id(input_args_info->input_args_id);
  top_cell_->set_input_args_info(top_input_args_info_);
  top_cell_->set_grad_first(call_grad_api_first_);
  call_grad_api_first_ = false;
  PushTopCellStack(top_cell_);
  MS_LOG(DEBUG) << "New top cell, top cell ptr " << top_cell_.get() << ", fg ptr " << fg.get()
                << ", with input args id " << top_cell_->input_args_id();
}

void GradExecutor::SetForwardLastNodeInfo(const InputArgsInfoPtr &input_args_info) const {
  auto v = input_args_info->out_value;
  MS_EXCEPTION_IF_NULL(v);
  auto value = v;
  if (v->isa<tensor::CSRTensor>()) {
    auto csr_tensorptr = v->cast<tensor::CSRTensorPtr>();
    value = csr_tensorptr->GetValues();
  } else if (v->isa<tensor::COOTensor>()) {
    auto coo_tensorptr = v->cast<tensor::COOTensorPtr>();
    value = coo_tensorptr->GetValues();
  }
  // Set last output abstract and will be used for sens
  (void)AutoGradUtil::SetValueGradInfo(value, InputType::kOpOutput);
  input_args_info->out_value = ShallowCopyTensorValue(value);
}

void GradExecutor::EndGraphInner(const py::object &obj, const py::object &out, const py::args &args) {
  if (input_args_info_stack_.empty()) {
    return;
  }
  const auto input_args_info = input_args_info_stack_.top();
  MS_EXCEPTION_IF_NULL(input_args_info);
  top_cell_ = top_cell_stack_.top();
  MS_EXCEPTION_IF_NULL(top_cell_);
  MS_LOG(DEBUG) << PrintPyObjInfo(obj, "End") << ", EndGraphInner start " << args.size() << ", cell_id "
                << PyNativeAlgo::PyParser::GetIdByPyObj(obj) << ", is custom bprop "
                << input_args_info->has_custom_bprop << ", input args info ptr " << input_args_info.get();
  if (input_args_info->is_grad_topest_cell) {
    GradState::Get().set_grad_flag(false);
  }
  // Get top cell endgraph
  if (input_args_info->cell_id == top_cell()->cell_id()) {
    {
      GilReleaseWithCheck no_gil;
      runtime::Pipeline::Get().frontend_stage()->Wait();
      runtime::Pipeline::Get().bprop_stage()->Wait();
    }
    if (input_args_info->out_value == nullptr) {
      input_args_info->out_value = parse::data_converter::PyObjToValue(out, false);
    }
    MS_LOG(DEBUG) << "Get cell output value " << input_args_info->out_value->ToString();
    EndGraphImpl(input_args_info);
    (void)PopTopCellStack();
  }
  PopInputArgsInfoStack();
}

void GradExecutor::EndGraphImpl(const InputArgsInfoPtr &input_args_info) {
  auto out_tensor = ConvertOutputValueToTensor(input_args_info->out_value, !top_cell()->jit_out_has_dict());
  if (out_tensor != nullptr) {
    input_args_info->out_value = out_tensor;
  }

  // Just only dump the last forward graph or bprop forward graph
  if (save_graphs_) {
    auto output_node =
      GetInput(input_args_info->out_value, PyNativeAlgo::Common::GetIdByValue(input_args_info->out_value));
    curr_g()->set_output(output_node);
    CommonUtils::DumpGraphIR("fg.ir", curr_g());
    MS_LOG(DEBUG) << "Save forward graph";
  }

  // Set sens value for grad
  SetForwardLastNodeInfo(input_args_info);

  // Checkout whether you need to compile graph when each top cell has run finished
  ready_run_top_cell_.insert({top_cell()->ready_run_cell_id(), top_cell_});

  if (!top_cell_->grad_first()) {
    DecreaseGradOrder();
  }
  top_input_args_info_ = input_args_info;
  forward()->ClearNodeAbsMap();
  ClearForwardGraph();
  MS_LOG(DEBUG) << "Cur top last cell " << input_args_info->cell_id;
}

py::object GradExecutor::RunGrad(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                                 const py::object &grad_position, const py::object &has_aux, const py::args &args) {
  // Wait forward task finish.
  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();
  GetTopCellWithInputArgsRespectTo(grad, obj, args);
  MS_EXCEPTION_IF_NULL(top_cell_);
  MS_LOG(DEBUG) << "Run top cell " << top_cell_;
  MS_LOG(DEBUG) << "Check size" << SizeofContainer();
  // Inputs args info must be update to current even no need compile graph again
  top_input_args_info_ = top_cell_->input_args_info();
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  // Set sens
  SetSensValue(grad, top_input_args_info_, args, !top_cell_->jit_out_has_dict());

  MS_LOG(DEBUG) << "RunGrad start " << args.size() << ", cell_id " << top_input_args_info_->cell_id
                << ", input args info ptr " << top_input_args_info_.get();
  op_num_in_bprop_graph_ = top_cell_->op_index();
  SetBpropGraphJitLevel(obj);
  bool weight_param_is_tuple = true;
  bool collect_default_param = false;
  auto w_args = GetWeightsArgs(weights, &weight_param_is_tuple, &collect_default_param);
  auto p_args = GetGradPositionArgs(grad_position, grad->get_by_position_);
  autograd::GradAttr grad_attr(grad->get_all_, grad->get_by_list_, grad->sens_param_, grad->get_by_position_,
                               weight_param_is_tuple);
  bool has_aux_val = py::cast<bool>(has_aux);
  auto ret = RunGradFunc(grad_attr, w_args, p_args, has_aux_val, collect_default_param);
  return ret;
}

std::string GradExecutor::GetReadyRunCellId(const std::string &obj_id, const std::string &input_args_id) const {
  std::string already_run_cell_id(obj_id);
  already_run_cell_id += "_" + input_args_id;
  already_run_cell_id += "_" + std::to_string(grad_order_ == 0 ? 1 : grad_order_);
  return already_run_cell_id;
}

void GradExecutor::GetTopCellWithInputArgsRespectTo(const prim::GradOperationPtr &grad, const py::object &obj,
                                                    const py::args &args) {
  if (finded_top_cell_ == nullptr) {
    MS_EXCEPTION_IF_NULL(grad);
    py::args args_without_sens;
    if (grad->sens_param_) {
      // If there is a sense, it will not hit the already run cache
      auto tuple_args_size = args.size() - 1;
      if (tuple_args_size < 0) {
        MS_LOG(EXCEPTION) << "args.size:" << args.size() << " tuple_args_size:" << tuple_args_size << " is invalid.";
      }
      py::tuple tuple_args(tuple_args_size);
      for (size_t i = 0; i < tuple_args_size; ++i) {
        tuple_args[i] = args[i];
      }
      args_without_sens = tuple_args;
    } else {
      args_without_sens = args;
    }
    const auto &input_args_id = GetInputArgsId(args_without_sens);
    const auto &cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
    const auto &ready_run_cell_id = GetReadyRunCellId(cell_id, input_args_id);

    MS_LOG(DEBUG) << "Get input cell id " << cell_id << " and already run cell id " << ready_run_cell_id
                  << ", input args id " << input_args_id;
    finded_top_cell_ = GetTopCell(ready_run_cell_id, input_args_id);
    MS_EXCEPTION_IF_CHECK_FAIL(finded_top_cell_ != nullptr,
                               "Can not find top cell for backward, please check your network whether set grad "
                               "or inputs of your network whether be inplace modified, this is forbidden!");
  }
  top_cell_ = finded_top_cell_;
  finded_top_cell_ = nullptr;
}

std::vector<tensor::TensorPtr> GradExecutor::GetWeightsArgs(const py::object &weights, bool *weight_param_is_tuple,
                                                            bool *collect_default_weights) const {
  std::vector<tensor::TensorPtr> w_args;
  if (py::hasattr(weights, "__parameter_tuple__")) {
    const auto &weights_tuple = weights.cast<py::tuple>();
    MS_LOG(DEBUG) << "Get weights tuple size " << weights_tuple.size();
    for (size_t i = 0; i < weights_tuple.size(); ++i) {
      const auto value = parse::data_converter::PyObjToValue(weights_tuple[i]);
      auto tensor = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      (void)w_args.emplace_back(tensor);
    }
  } else {
    if (py::isinstance<py::tuple>(weights) || py::isinstance<py::list>(weights)) {
      MS_LOG(DEBUG) << "Get weights params by input tuple/list weight";
      auto weights_tuple = py::cast<py::tuple>(weights);
      for (size_t i = 0; i < weights_tuple.size(); ++i) {
        const auto value = parse::data_converter::PyObjToValue(weights_tuple[i]);
        auto tensor = value->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        (void)w_args.emplace_back(tensor);
      }
    } else if (!py::isinstance<py::none>(weights)) {
      // Single input
      MS_LOG(DEBUG) << "Get weights params by single input weight";
      const auto value = parse::data_converter::PyObjToValue(weights);
      auto tensor = value->cast<tensor::TensorPtr>();
      (void)w_args.emplace_back(tensor);
      if (tensor == nullptr) {
        MS_EXCEPTION(TypeError) << "The weights you provided is expected to be a tensor, or a list or tuple. but got "
                                << value->ToString();
      }
      *weight_param_is_tuple = false;
    } else {
      MS_LOG(DEBUG) << "Need collect default weight from forward record";
      *collect_default_weights = true;
      return {};
    }
  }
  return w_args;
}

std::vector<size_t> GradExecutor::GetGradPositionArgs(const py::object &grad_position, bool get_by_position) const {
  std::vector<size_t> pos_args;
  if (!get_by_position) {
    return pos_args;
  }
  if (py::isinstance<py::tuple>(grad_position)) {
    const auto &tuple = grad_position.cast<py::tuple>();
    (void)std::transform(tuple.begin(), tuple.end(), std::back_inserter(pos_args),
                         [](const py::handle &elem) { return elem.cast<int64_t>(); });
    if (pos_args.empty()) {
      MS_LOG(EXCEPTION) << "grad_position should not be empty when grad by position!";
    }
    return pos_args;
  }
  MS_LOG(EXCEPTION) << "Grad position only support tuple when grad_by_position is set True.";
}

bool GradExecutor::NeedIncreaseGradOrder(const std::string &obj_id) {
  // top_cell_ == nullptr means call by grad first
  // top_cell_->obj_id_with_grad_order() include obj_id and grad_order
  // If top_cell_->obj_id_with_grad_order().find(obj_id) == std::string::npos, means current cell is not top cell,
  // another cell or function needs to get grad, so high-order comes up
  if (top_cell_ == nullptr || top_cell_->cell_id().find(obj_id + "_") == std::string::npos) {
    IncreaseGradOrder();
    return true;
  }
  return false;
}

py::object GradExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj,
                                         const py::object &weights, const py::object &grad_position,
                                         const py::args &args, const py::kwargs &kwargs) {
  const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);

  // The rule of grad order is:
  // scenarios 1. net.set_grad, net(input) calls first, increase 1 before MakeNewTopCell, and decrease 1 when running to
  // EndGraphImpl, indicating that a complete bprop graph construction is completed; Then call grad(net)(input) is won't
  // be affected and get forward_run is true.
  // scenarios 2. If grad(net)(input) calls first, then increase 1 before MakeNewTopCell and decrease 1 in Rungrad. The
  // reason for this design is that if grad(net)(input) calls first and decrease 1 in EndGraphImpl, it will cause
  // matching problems during RunGrad due to the presence of GradOperation information in already_run_cell_id is not the
  // same. GradOperation information includes grad order for distinguish high-order.
  // Use a flag: call_grad_api_first_ for distinguish these two scenarios. If scenarios 1 are taken,
  // call_grad_api_first_ will not take effect, otherwise, it works.
  bool high_order = kwargs.contains(kCreateGraph) ? (py::cast<bool>(kwargs[kCreateGraph])) : true;
  bool need_increase_grad_order = false;
  if (!high_order) {
    grad_order_ = 1;
  } else {
    need_increase_grad_order = NeedIncreaseGradOrder(obj_id);
  }
  auto input_args_id = GetInputArgsId(args);
  // Under the condition that the stack is empty (forward process completed or no forward process),
  // check whether need to run forward process
  bool forward_run = false;
  // To do
  if (input_args_info_stack_.empty()) {
    const auto &check_ready_run_cell_id = GetReadyRunCellId(obj_id, input_args_id);
    MS_LOG(DEBUG) << "Get check ready run top cell id " << check_ready_run_cell_id;
    auto find_top_cell = GetTopCell(check_ready_run_cell_id, input_args_id);
    if (find_top_cell != nullptr) {
      MS_LOG(DEBUG) << "Find already run top cell " << find_top_cell;
      // If need_increase_grad_order is true means grad order increased and prepare to do grad;
      // But forward run is true now, means no need do forward again, so grad order need be decrease.
      if (need_increase_grad_order) {
        DecreaseGradOrder();
      }
      finded_top_cell_ = find_top_cell;
      forward_run = true;
    }
  }
  if (!forward_run) {
    call_grad_api_first_ = true;
  }
  forward_run ? MS_LOG(DEBUG) << "Top cell have already ran with input args id " << input_args_id
              : MS_LOG(DEBUG) << "Top cell no run before with input args id " << input_args_id;
  return BaseRefToPyData(forward_run);
}

py::object GradExecutor::RunGradFunc(const autograd::GradAttr &grad_attr, const std::vector<tensor::TensorPtr> &w_args,
                                     const std::vector<size_t> &p_args, bool has_aux, bool collect_default_weights) {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  ValuePtr sens = nullptr;
  if (grad_attr.has_sens) {
    sens = top_input_args_info_->input_arg_value_vec.back();
    top_input_args_info_->input_arg_value_vec.pop_back();
  }
  MS_LOG(DEBUG) << "Eval run begin";
  MS_EXCEPTION_IF_NULL(top_cell_);
  auto cur_top_cell = top_cell_;
  auto engine = std::make_shared<autograd::AutoDiff>(top_input_args_info_->out_value, false,
                                                     cur_top_cell->is_high_order_top_cell(), is_run_recompute_);
  autograd::AutoDiffGuard auto_diff_guard(engine);
  top_cell_->set_grad_is_running(true);
  auto grads = engine->RunGradFunc(top_input_args_info_->input_arg_value_vec, w_args, p_args, grad_attr,
                                   collect_default_weights, has_aux, sens);
  engine->RunFinalCallback();
  top_cell_ = cur_top_cell;
  MS_EXCEPTION_IF_NULL(grads);
  MS_EXCEPTION_IF_NULL(cur_top_cell);
  cur_top_cell->set_grad_is_running(false);
  MS_LOG(DEBUG) << "Eval run end";
  cur_top_cell = nullptr;
  ClearGradRes();
  AsyncClearEngine(engine);
  return BaseRefToPyData(grads);
}

void GradExecutor::ClearGlobalRes() const {
  abstract::AnalysisContext::ClearContext();
  parse::data_converter::ClearObjectCache();
  pipeline::CleanParserResource();
  trace::ClearTraceStack();
  ad::ClearDFunctor();
  pipeline::ReclaimOptimizer();
}

void GradExecutor::ClearGradRes() {
  MS_LOG(DEBUG) << "Top cell run finish " << top_cell_;
  top_cell_->input_args_info()->Reset();
  if (top_cell_->grad_first()) {
    DecreaseGradOrder();
  }
  top_input_args_info_ = nullptr;
  ClearGlobalRes();
  MS_LOG(DEBUG) << "Current top cell stack size " << top_cell_stack_.size() << "already_run top cell size"
                << ready_run_top_cell_.size() << " size ";
  auto range = ready_run_top_cell_.equal_range(top_cell_->ready_run_cell_id());
  for (auto iter = range.first; iter != range.second; iter++) {
    if (iter->second.get() == top_cell_.get()) {
      MS_LOG(DEBUG) << "Erase top cell " << top_cell_;
      ready_run_top_cell_.erase(iter);
    }
    break;
  }
  top_cell_ = nullptr;
  // Nested grad, get outer top cell if exist
  // Run top cell with bprop, and bprop has grad, after running inner grad, top cell should be restore
  if (!top_cell_stack_.empty()) {
    top_cell_ = top_cell_stack_.top();
    MS_LOG(DEBUG) << "Get outer top cell " << top_cell_ << " as the currently running top cell";
  }
}

void GradExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear grad res";
  WaitBpropTask();
  if (ready_run_top_cell_.size() > kIndex0) {
    MS_LOG(INFO) << "Top cell did not be consumed, which may cause device memory leaks, if the program "
                    "exits normally, make sure your network's set_grad() flag set correctly!";
  }
  init_ = false;
  GradState::Get().set_grad_flag(false);
  GradState::Get().set_enable_grad(true);
  is_run_recompute_ = false;
  save_graphs_ = false;
  forward_use_dynamic_shape_process_ = false;
  grad_order_ = 0;
  op_num_in_bprop_graph_ = kDefaultContainerSize;

  top_cell_ = nullptr;
  top_input_args_info_ = nullptr;
  std::stack<InputArgsInfoPtr>().swap(input_args_info_stack_);
  std::stack<TopCellInfoPtr>().swap(top_cell_stack_);
  finded_top_cell_ = nullptr;
  ready_run_top_cell_.clear();
  dynamic_inputs_cells_.clear();
}

void GradExecutor::WorkerJoin() { runtime::Pipeline::Get().bprop_stage()->WorkerJoin(); }

AnfNodePtr GradExecutor::GetInput(const ValuePtr &v, const string &obj_id) const {
  // Is not a tensor
  AnfNodePtr node = GetNonTensorInput(v, obj_id);
  if (node != nullptr) {
    return node;
  }
  // Get param input
  node = GetParamInput(v, obj_id);
  if (node != nullptr) {
    return node;
  }
  // Get op output
  node = GetOutputNodeAsInput(obj_id);
  if (node != nullptr) {
    return node;
  }
  // A tuple returns in this case: x = op1, y = op2, return (x, y)
  // or a scalar or (scalar, tensor)
  node = GetValueSequenceInput(v);
  if (node != nullptr) {
    return node;
  }
  auto v_node = PyNativeAlgo::Common::CreateValueNodeByValue(v);
  MS_LOG(DEBUG) << "Get input value node " << v_node->ToString() << ", id " << obj_id;
  return v_node;
}

AnfNodePtr GradExecutor::GetParamInput(const ValuePtr &v, const std::string &id) const {
  const auto &graph_info = top_cell()->graph_info_map().at(curr_g());
  MS_EXCEPTION_IF_NULL(graph_info);
  // Get input param input
  const auto it = graph_info->input_params.find(id);
  if (it != graph_info->input_params.end()) {
    MS_LOG(DEBUG) << "Get input param " << id;
    return it->second;
  }

  // Get weight param input
  MS_EXCEPTION_IF_NULL(v);
  if (v->isa<tensor::Tensor>() && v->cast<tensor::TensorPtr>()->is_parameter()) {
    const auto item_by_id = graph_info->weight_params.find(id);
    if (item_by_id != graph_info->weight_params.end()) {
      MS_LOG(DEBUG) << "Get weight param " << id;
      return item_by_id->second;
    }
    MS_LOG(DEBUG) << "Add new weight param " << id;
    auto tensor = v->cast<tensor::TensorPtr>();
    const auto &param_info = tensor->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    const auto &param_name = param_info->name();
    // Add new weight param to graph info
    auto weight_param = curr_g()->add_parameter();
    weight_param->set_name(param_name);
    if (weight_param->debug_info() != nullptr) {
      weight_param->debug_info()->set_name(param_name);
    }
    weight_param->set_default_param(tensor);
    weight_param->set_abstract(CommonUtils::SetAbstractValueToAnyValue(tensor->ToAbstract()));
    top_cell()->SetParamNodeMapInGraphInfoMap(id, weight_param, true);
    return weight_param;
  }
  return nullptr;
}

AnfNodePtr GradExecutor::GetOutputNodeAsInput(const std::string &obj_id) const {
  const auto &graph_info = top_cell()->graph_info_map().at(curr_g());
  MS_EXCEPTION_IF_NULL(graph_info);
  const auto it = graph_info->node_map.find(obj_id);
  if (it == graph_info->node_map.end()) {
    return nullptr;
  }
  // Single output CNode
  if (it->second.second.size() == 1 && it->second.second[0] == -1) {
    MS_LOG(DEBUG) << "Get input node " << it->second.first->ToString() << ", id " << obj_id;
    return it->second.first;
  }
  // Create tuple get item node for multiple output CNode
  return CreateTupleGetItemNode(obj_id, it->second);
}

AnfNodePtr GradExecutor::GetValueSequenceInput(const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  if (!v->isa<ValueSequence>()) {
    return nullptr;
  }
  ValuePtrList input_args;
  abstract::AbstractBasePtrList abs_list;
  AnfNodePtrList inputs{NewValueNode(prim::kPrimMakeTuple)};
  const auto &obj_tuple = v->cast<ValueSequencePtr>();
  const auto &v_list = obj_tuple->value();
  for (size_t i = 0; i < obj_tuple->size(); ++i) {
    const auto &v_arg = v_list[i];
    // Graph have no define for grad
    if (v_arg->isa<FuncGraph>()) {
      continue;
    }
    (void)input_args.emplace_back(v_arg);
    const std::string &id = PyNativeAlgo::Common::GetIdByValue(v_arg);
    (void)inputs.emplace_back(GetInput(v_arg, id));
    (void)abs_list.emplace_back(CommonUtils::SetAbstractValueToAnyValue(v_arg->ToAbstract()));
    (void)GetValueSequenceInput(v_arg);
  }
  // Create make tuple node and record to graph info map.
  auto cnode = curr_g()->NewCNode(inputs);
  cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  MS_LOG(DEBUG) << "Create make tuple node: " << cnode->DebugString();
  return cnode;
}

AnfNodePtr GradExecutor::CreateTupleGetItemNode(const std::string &obj_id,
                                                const std::pair<AnfNodePtr, std::vector<int64_t>> &out) const {
  AnfNodePtr c_node = out.first->cast<CNodePtr>();
  bool param_is_sequence = false;
  if (c_node == nullptr) {
    // Input param is tuple or list
    if (GetParamInput(MakeValue(true), obj_id) != nullptr) {
      MS_LOG(EXCEPTION) << "Get wrong input node " << out.first->DebugString();
    }
    param_is_sequence = true;
    c_node = out.first;
  }
  MS_LOG(DEBUG) << "Sequence input node " << c_node->DebugString() << ", id " << obj_id << ", out second "
                << out.second;
  // Create tuple get item node
  auto abs = c_node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  for (auto idx : out.second) {
    AnfNodePtrList tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), c_node, NewValueNode(idx)};
    c_node = curr_g()->NewCNode(tuple_get_item_inputs);
    if (!abs->isa<abstract::AbstractSequence>()) {
      MS_LOG(EXCEPTION) << "Input node abs is not sequence " << abs->ToString();
    }
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    if (static_cast<size_t>(idx) >= abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Index exceeds the size of elements. Index " << idx << ", element size " << abs_seq->size();
    }
    abs = abs_seq->elements()[static_cast<size_t>(idx)];
    MS_EXCEPTION_IF_NULL(abs);
    c_node->set_abstract(abs);
    if (param_is_sequence) {
      c_node->set_user_data(kParamterIsSequence, MakeValue(param_is_sequence));
    }
  }
  MS_LOG(DEBUG) << "Create tuple getitem node " << c_node->DebugString() << ", abs " << c_node->abstract()->ToString();
  return c_node;
}

TopCellInfoPtr GradExecutor::GetTopCell(const std::string &already_run_cell_id, const std::string &input_args_id) {
  TopCellInfoPtr find_top_cell = nullptr;
  for (const auto &[cell_id, top_cell] : ready_run_top_cell_) {
    MS_EXCEPTION_IF_NULL(top_cell);
    MS_LOG(DEBUG) << "Top cell " << top_cell << " with ready run cell id " << cell_id << ", input args id "
                  << top_cell->input_args_id();
    // Complete match, means run grad operation first
    if (top_cell->ready_run_cell_id() == already_run_cell_id) {
      find_top_cell = top_cell;
      break;
    }
  }
  return find_top_cell;
}

void GradExecutor::ProcessOpGradInfo(const OpGradInfoPtr &grad_info) const {
  RecordForwardGraph(grad_info);
  DoOpGrad(grad_info);
}

py::object GradExecutor::CallCustomBprop(const py::object &obj, const py::object out, const py::args &args) {
  MS_LOG(DEBUG) << "Begin CallCustomBprop";
  autograd::CustomContext context;
  if (!py::isinstance<Cell>(obj)) {
    MS_LOG(EXCEPTION) << "For custom bprop, obj should be Cell";
  }
  const auto &cell_ptr = obj.cast<CellPtr>();
  context.cell_name = cell_ptr->name();
  context.is_recompute = cell_ptr->HasAttr(kNeedRecompute);
  context.bprop_fn = py::getattr(obj, parse::CUSTOM_BPROP_NAME);
  (void)CheckBpropWithJit(context.bprop_fn);
  context.inputs.reserve(args.size() + kSizeEight);
  context.input_value_grad_type.reserve(args.size() + kSizeEight);
  for (size_t i = 0; i < args.size(); ++i) {
    auto input = PyNativeAlgo::Common::StubNodeToValue(parse::data_converter::PyObjToValue(args[i], true));
    (void)context.input_value_grad_type.emplace_back(AutoGradUtil::SetValueGradInfo(input, InputType::kConstant));
    (void)context.inputs.emplace_back(std::move(input));
  }
  auto output = PyNativeAlgo::Common::StubNodeToValue(parse::data_converter::PyObjToValue(out, true));
  if (context.is_recompute) {
    output = ConvertOutputValueToTensor(output, !top_cell()->jit_out_has_dict());
  }
  bool is_return_self = false;
  output = CheckAndUpdateOutput(output, context.inputs, &is_return_self);
  (void)AutoGradUtil::SetValueGradInfo(output, InputType::kOpOutput);
  context.output = output;
  SetCustomBpropInputs(obj, &context);
  RecordCustomBprop(context);
  forward()->WaitForwardTask();
  if (forward()->enable_async()) {
    auto task = [new_context = std::move(context)]() { (void)autograd::CallCustomBprop(new_context); };
    DispatchGradQueueTask(std::move(task));
  } else {
    (void)autograd::CallCustomBprop(std::move(context));
  }
  MS_LOG(DEBUG) << "End CallCustomBprop";
  return is_return_self ? CValueToPybindObj(output) : out;
}

void GradExecutor::SaveOutputNodeMap(const std::string &obj_id, const OpGradInfoPtr &grad_info, const CNodePtr &cnode,
                                     const std::vector<std::string> &input_value_id) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Cnode is " << cnode->DebugString() << ", out value id " << obj_id;
  if (grad_info->run_in_vm && kHookOp.find(grad_info->op_prim->name()) != kHookOp.end()) {
    for (size_t i = 0; i < grad_info->input_value.size(); ++i) {
      top_cell()->DeleteParamNodeInfo(curr_g(), input_value_id[i]);
    }
  }
  top_cell()->SetNodeMapInGraphInfoMap(obj_id, cnode);
}

void GradExecutor::DoOpGrad(const OpGradInfoPtr &grad_info) const {
  auto &&grad_param = CreateOpGradParam(grad_info);
  if (forward()->enable_async()) {
    auto task = [grad_param]() { autograd::KPynativeOp(grad_param); };
    DispatchGradQueueTask(std::move(task));
  } else {
    autograd::KPynativeOp(grad_param);
  }
}

CNodePtr GradExecutor::ConstructForwardGraph(const OpGradInfoPtr &grad_info,
                                             const std::vector<std::string> &input_value_id) const {
  AnfNodePtrList inputs;
  (void)inputs.emplace_back(NewValueNode(grad_info->op_prim));
  for (size_t i = 0; i < grad_info->input_value.size(); i++) {
    (void)inputs.emplace_back(GetInput(grad_info->input_value[i], input_value_id[i]));
  }
  const auto &cnode = curr_g()->NewCNodeInOrder(inputs);
  MS_LOG(DEBUG) << "Make CNode for " << grad_info->op_prim->name() << ", new cnode is " << cnode->DebugString();
  return cnode;
}

void GradExecutor::RecordForwardGraph(const OpGradInfoPtr &grad_info) const {
  if (save_graphs_ && top_cell_ != nullptr && top_cell_->fg() != nullptr) {
    std::string out_value_id;
    // Hold tensorGradType
    std::vector<std::string> input_value_id;
    (void)std::transform(grad_info->input_value.begin(), grad_info->input_value.end(),
                         std::back_inserter(input_value_id),
                         [](const ValuePtr &value) { return PyNativeAlgo::Common::GetIdByValue(value); });

    out_value_id = PyNativeAlgo::Common::GetIdByValue(grad_info->out_value);
    const auto &cnode = ConstructForwardGraph(grad_info, input_value_id);
    MS_EXCEPTION_IF_NULL(cnode);
    // By simple infer, abstract is nullptr
    cnode->set_abstract(CommonUtils::SetAbstractValueToAnyValue(grad_info->out_value->ToAbstract()));

    SaveOutputNodeMap(out_value_id, grad_info, cnode, input_value_id);
  }
}

void GradExecutor::RecordCustomBprop(const autograd::CustomContext &context) const {
  if (save_graphs_ && top_cell_ != nullptr && top_cell_->fg() != nullptr) {
    auto op_run_info = CustomContext2OpRunInfo(context);
    RecordForwardGraph(op_run_info->op_grad_info);
  }
}

void GradExecutor::RecordForwardGraphForInput(const ValuePtr &value, const string &input_id) {
  save_graphs_ = MsContext::GetInstance()->CanDump(kIntroductory);
  if (save_graphs_ && top_cell_ != nullptr && top_cell_->fg() != nullptr) {
    auto param_abs = CommonUtils::SetAbstractValueToAnyValue(value->ToAbstract());
    auto new_param = curr_g()->add_parameter();
    new_param->set_abstract(param_abs);
    if (value->isa<ValueSequence>()) {
      top_cell()->SetNodeMapInGraphInfoMap(input_id, new_param, true);
    }
    top_cell()->SetParamNodeMapInGraphInfoMap(input_id, new_param);
  }
}

void GradExecutor::SetBpropGraphJitLevel(const py::object &obj) const {
  if (!py::hasattr(obj, kAttrCellJitConfigDict)) {
    return;
  }

  auto jit_config = py::getattr(obj, kAttrCellJitConfigDict);
  if (!py::isinstance<py::dict>(jit_config)) {
    MS_LOG(EXCEPTION) << "JitConfig only support dict!";
  }
  auto jit_config_dict = jit_config.cast<py::dict>();
  pipeline::ExecutorPyPtr graph_executor = pipeline::GetExecutor();
  MS_EXCEPTION_IF_NULL(graph_executor);
  graph_executor->SetJitConfig(jit_config_dict);
}

void GradExecutor::SaveDynamicInputsCells(const py::object &obj, const py::args &args) {
  const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
  MS_LOG(INFO) << "SaveDynamicInputsCells: "
               << (py::isinstance<Cell>(obj) ? obj_id + " " + obj.cast<CellPtr>()->ToString()
                                             : py::getattr(obj, "__name__").cast<std::string>());
  (void)dynamic_inputs_cells_.insert(obj_id);
}

void GradExecutor::DispatchGradQueueTask(std::function<void(void)> &&task) const {
  runtime::Pipeline::Get().bprop_stage()->Push(std::make_shared<BpropTask>(task));
}

void GradExecutor::QueueFinalCallback(std::function<void()> callback) const {
  auto cur_auto_diff_engine = autograd::impl::CurrentAutoDiffEngine();
  MS_EXCEPTION_IF_CHECK_FAIL(cur_auto_diff_engine != nullptr,
                             "Final backward callback can only be installed during backward pass");
  auto engine = std::dynamic_pointer_cast<autograd::AutoDiff>(cur_auto_diff_engine);
  MS_EXCEPTION_IF_NULL(engine);
  engine->AddFinalCallback(std::move(callback));
}

std::string GradExecutor::SizeofContainer() const {
  std::ostringstream buf;
  buf << "input_args_info_stack_ size: " << input_args_info_stack_.size();
  buf << " top_cell_stack_ size: " << top_cell_stack_.size();
  buf << " already_run_top_cell_ size: " << ready_run_top_cell_.size();
  buf << " dynamic_inputs_cells_ size: " << dynamic_inputs_cells_.size();
  return buf.str();
}

void GradExecutor::WaitBpropTask() const {
  const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
  if (bprop_queue != nullptr) {
    GilReleaseWithCheck gil_release;
    bprop_queue->Wait();
  }
}

void GradExecutor::ChildAfterFork() {
  MS_LOG(DEBUG) << "GradExecutor reinitialize after fork.";
  runtime::PyBoostOpExecute::GetInstance().ClearBackend();
  MS_LOG(DEBUG) << "GradExecutor reinitialize after fork done.";
}
}  // namespace pynative
}  // namespace mindspore
