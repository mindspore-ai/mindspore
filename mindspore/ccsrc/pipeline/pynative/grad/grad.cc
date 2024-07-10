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
#include "ops/conv_pool_op_name.h"
#include "ops/nn_op_name.h"
#include "ops/math_op_name.h"
#include "ops/sequence_ops.h"
#include "ops/framework_ops.h"
#include "pipeline/pynative/grad/top_cell.h"
#include "pipeline/pynative/grad/function/func_grad.h"
#include "pipeline/pynative/grad/ir/ir_grad.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/jit/ps/pipeline.h"
#include "ir/cell.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/convert_utils_py.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/environ_conversion.h"
#include "pipeline/jit/ps/pass.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "frontend/optimizer/fallback_rewriter.h"
#include "runtime/pynative/op_function/pyboost_grad_functions.h"
#include "runtime/pynative/op_executor.h"

namespace mindspore {
namespace pynative {
namespace {
const mindspore::HashSet<std::string> kHookOp = {"HookBackward", "CellBackwardHook"};
constexpr char kGrad[] = "grad";
constexpr auto kNeedRecompute = "is_cell_recompute";
constexpr auto kInternalParams = "internal_params";
constexpr auto kUsedBpropInputs = "used_bprop_inputs";
constexpr size_t kContainerRatio = 2;

void ParsePyArgsToInputArgsInfo(const InputArgsInfoPtr &input_args_info, const py::object &obj, const py::args &args,
                                bool is_bprop_need_get_forward_graph) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  input_args_info->has_custom_bprop = py::hasattr(obj, parse::CUSTOM_BPROP_NAME);
  MS_LOG(DEBUG) << "Cell has custom bprop " << input_args_info->has_custom_bprop;
  bool is_top_cell = input_args_info->is_grad_topest_cell || input_args_info->is_high_order_top_cell;
  if (is_top_cell) {
    pipeline::CheckArgsValid(obj, args);
  }
  // Only the top cell or custom bprop cell requires value conversion
  if (is_top_cell || input_args_info->has_custom_bprop || is_bprop_need_get_forward_graph) {
    input_args_info->obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
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
          PyNativeAlgo::DataConvert::PyObjToValue(py::cast<py::tuple>(args[i])));
      } else {
        (void)input_args_info->input_arg_value_vec.emplace_back(PyNativeAlgo::DataConvert::PyObjToValue(args[i]));
      }

      // Get arg abstract
      auto abs = forward->GetNodeAbsById(input_args_info->input_arg_id_vec[i]);
      if (abs == nullptr) {
        abs = input_args_info->input_arg_value_vec.back()->ToAbstract();
      }
      (void)input_args_info->input_arg_base_shape_vec.emplace_back(abs->BuildShape());
    }
    input_args_info->cell_id = PyNativeAlgo::Common::GetCellId(
      input_args_info->obj_id, input_args_info->input_arg_id_vec, input_args_info->input_arg_value_vec);
    MS_LOG(DEBUG) << "Cell_id is " << input_args_info->cell_id << ", is grad topest cell "
                  << input_args_info->is_grad_topest_cell << ", is high order top cell "
                  << input_args_info->is_high_order_top_cell << ", is bprop need get forward graph "
                  << is_bprop_need_get_forward_graph;
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
    return std::make_shared<tensor::Tensor>(input_value, kFloat32);
  }
  if (v->isa<BoolImm>()) {
    return std::make_shared<tensor::Tensor>(v->cast<BoolImmPtr>()->value(), kBool);
  }
  if (v->isa<IntegerImm>()) {
    int64_t input = v->cast<Int64ImmPtr>()->value();
    return std::make_shared<tensor::Tensor>(input, kInt64);
  }
  if (v->isa<ValueDictionary>() && dict_convert_to_tuple) {
    MS_LOG(DEBUG) << "Get dict value";
    return PyNativeAlgo::DataConvert::ConvertValueDictToValueTuple(v);
  }
  MS_LOG(DEBUG) << "Output is " << v->ToString() << ", abstract "
                << PyNativeAlgo::Common::SetAbstractValueToAnyValue(v->ToAbstract());
  return v;
}

FuncGraphPtr BpropGraphFinalOpt(const FuncGraphPtr &bprop_graph, bool has_control_flow) {
  MS_LOG(DEBUG) << "Do bprop graph final opt";
  MS_EXCEPTION_IF_NULL(bprop_graph);
  auto resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph);
  FuncGraphPtr after_opt_bg = nullptr;
  after_opt_bg = pipeline::FinalBpropGraphPass(resource, has_control_flow);
  PyNativeAlgo::Common::DumpGraphIR("after_final_opt.ir", after_opt_bg);
  return after_opt_bg;
}

void SetGraphInputArgs(const std::vector<ValuePtr> &input_vec, const pipeline::ResourcePtr &res,
                       size_t graph_param_size, SensType sens_type, VectorRef *const arg_list) {
  MS_EXCEPTION_IF_NULL(arg_list);
  MS_EXCEPTION_IF_NULL(res);
  auto graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  const auto &graph_params = graph->parameters();
  if (graph_params.size() < graph_param_size) {
    MS_LOG(EXCEPTION) << "Get initial bprop graph param size " << graph_param_size << " less than current param size "
                      << graph_params.size() << ". Graph parameters maybe update by kernel graph compile stage";
  }
  std::vector<ValuePtr> input_arg_list;
  if (sens_type == SensType::kNormal) {
    input_arg_list = input_vec;
  } else if (sens_type == SensType::kTuple) {
    PyNativeAlgo::DataConvert::FlattenArgs(input_vec, &input_arg_list, true);
  } else {
    input_arg_list.assign(input_vec.begin(), input_vec.end() - kIndex1);
    const auto &v_sens = input_vec.back();
    MS_EXCEPTION_IF_NULL(v_sens);
    if (!v_sens->isa<ValueDictionary>()) {
      MS_LOG(EXCEPTION) << "Get sens not dict " << v_sens->ToString();
    }
    const auto &v_dict = v_sens->cast<ValueDictionaryPtr>();
    ValuePtrList key_inputs;
    ValuePtrList value_inputs;
    for (const auto &elem : v_dict->value()) {
      (void)key_inputs.emplace_back(elem.first);
      (void)value_inputs.emplace_back(elem.second);
    }
    auto key = std::make_shared<ValueTuple>(key_inputs);
    auto value = std::make_shared<ValueTuple>(value_inputs);
    (void)input_arg_list.emplace_back(key);
    (void)input_arg_list.emplace_back(value);
  }
  (void)std::transform(input_arg_list.begin(), input_arg_list.end(), std::back_inserter(*arg_list),
                       [](const ValuePtr &v) { return v; });
  size_t arg_size = arg_list->size();
  if (arg_size != graph_param_size) {
    // Maybe have some default parameter for input
    MS_LOG(DEBUG) << "Get args size " << arg_size << ", graph param size " << graph_param_size;
    for (std::size_t i = arg_size; i < graph_param_size; ++i) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      auto param_ptr = (graph_params[i])->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->has_default()) {
        MS_LOG(EXCEPTION) << "Parameter[" << i << "] has no default param, " << param_ptr->DebugString();
      }
      if (!param_ptr->default_param()->isa<tensor::BaseTensor>()) {
        MS_LOG(EXCEPTION) << "Parameter[" << param_ptr->DebugString()
                          << "] is not initialized, need to call `.init_data()`";
      }
      arg_list->push_back(param_ptr->default_param());
    }
  }
}

void RestoreBpropGraphParameter(const FuncGraphPtr &graph, size_t graph_param_size) {
  auto parameters = graph->parameters();
  // Is ascend, kernel graph maybe adjust and insert some control parameters
  if (parameters.size() > graph_param_size) {
    (void)parameters.erase(parameters.begin() + graph_param_size, parameters.end());
    graph->set_parameters(std::move(parameters));
  }
}

void SetSensValue(const prim::GradOperationPtr &grad, const InputArgsInfoPtr &input_args_info, const py::args &args,
                  bool dict_convert_to_tuple) {
  MS_EXCEPTION_IF_NULL(grad);
  if (!grad->sens_param()) {
    return;
  }
  size_t forward_args_size = args.size() - 1;
  auto sens_v = PyNativeAlgo::DataConvert::PyObjToValue(args[forward_args_size]);
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

std::string GetWeightsObjIdsByWeights(const py::object &weights) {
  auto is_require_grad = [](const ValuePtr &value) {
    MS_EXCEPTION_IF_NULL(value);
    if (!value->isa<tensor::BaseTensor>()) {
      return false;
    }
    auto t = value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(t);
    if (t->is_parameter() && t->param_info() != nullptr && t->param_info()->requires_grad()) {
      return true;
    }
    return false;
  };

  std::string weights_obj_id;
  auto append_weights_info = [&weights_obj_id, &is_require_grad](const py::object &obj) {
    const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(obj);
    if (is_require_grad(v)) {
      (void)weights_obj_id.append("_").append(PyNativeAlgo::Common::GetIdByValue(v));
    }
  };

  if (py::isinstance<py::tuple>(weights)) {
    const auto &weights_tuple = weights.cast<py::tuple>();
    for (size_t i = 0; i < weights_tuple.size(); ++i) {
      append_weights_info(weights_tuple[i]);
    }
  } else if (py::isinstance<py::list>(weights)) {
    const auto &weights_list = weights.cast<py::list>();
    for (size_t i = 0; i < weights_list.size(); ++i) {
      append_weights_info(weights_list[i]);
    }
  } else if (!py::isinstance<py::none>(weights)) {
    append_weights_info(weights);
  }

  return weights_obj_id;
}

void FreeSpecialOpValue(const std::string &op_name, const FrontendOpRunInfoPtr &op_run_info, ValuePtr *const output) {
  // Special cases, manually free more inputs.
  static mindspore::HashSet<std::string> kMulOp{
    kMulOpName,
    kMatMulOpName,
    kConv2DOpName,
  };
  static mindspore::HashSet<std::string> kDivOp{
    kDivOpName,
    kRealDivOpName,
  };
  if (op_name == kBatchNormOpName) {
    // 1. BatchNorm is a multi-output node, it's out[0] and out[1] are not used.
    auto seq_v = (*output)->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_v);
    ValuePtrList new_v_list{seq_v->value()};
    new_v_list[kIndex0] = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(new_v_list[kIndex0]);
    new_v_list[kIndex1] = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(new_v_list[kIndex1]);
    *output = std::make_shared<ValueTuple>(new_v_list);
    MS_LOG(DEBUG) << "Clear device address for output[0, 1] of " << op_name;
  } else if (op_name == kLayerNormOpName) {
    // 2. LayerNorm is a multi-output node, it's out[0] and out[1] are not used.
    auto seq_v = (*output)->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_v);
    ValuePtrList new_v_list{seq_v->value()};
    new_v_list[kIndex0] = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(new_v_list[kIndex0]);
    *output = std::make_shared<ValueTuple>(new_v_list);
    MS_LOG(DEBUG) << "Clear device address for output[0] of " << op_name;
  } else if (kMulOp.find(op_name) != kMulOp.end()) {
    // 3. For operators like Mul, the dx ONLY rely on y, and dy ONLY rely on x.
    //    so if y is a valuenode, the dy is useless, we can free x in ahead.
    bool x_is_const_value = PyNativeAlgo::Common::IsConstant(op_run_info->op_grad_info->input_value_grad_type[kIndex0]);
    bool y_is_const_value = PyNativeAlgo::Common::IsConstant(op_run_info->op_grad_info->input_value_grad_type[kIndex1]);
    if (x_is_const_value && op_run_info->base_op_run_info.expanded_input_values[kIndex1]->isa<tensor::BaseTensor>()) {
      op_run_info->op_grad_info->input_value[kIndex1] = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(
        op_run_info->base_op_run_info.expanded_input_values[kIndex1]);
      MS_LOG(DEBUG) << "Clear device address for inputs[1] of " << op_name;
    }
    if (y_is_const_value && op_run_info->base_op_run_info.expanded_input_values[kIndex0]->isa<tensor::BaseTensor>()) {
      op_run_info->op_grad_info->input_value[kIndex0] = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(
        op_run_info->base_op_run_info.expanded_input_values[kIndex0]);
      MS_LOG(DEBUG) << "Clear device address for inputs[0] of " << op_name;
    }
  } else if (kDivOp.find(op_name) != kDivOp.end()) {
    // 3. For operators like Div, the dy does not rely on output node, so if y is a valuenode, we can free output.
    if (PyNativeAlgo::Common::IsConstant(op_run_info->op_grad_info->input_value_grad_type[kIndex1])) {
      *output = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(*output);
      MS_LOG(DEBUG) << "Clear device address for the output of " << op_name;
    }
  }
}

void FreeUselessValue(const FrontendOpRunInfoPtr &op_run_info, const GradParamPtr &grad_param,
                      const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_param);
  MS_EXCEPTION_IF_NULL(top_cell);
  if (top_cell->is_high_order_top_cell()) {
    return;
  }

  const auto &unused_inputs = BpropExpander::GetUnusedInputs(op_run_info->op_grad_info->op_prim->name());
  for (const auto i : unused_inputs) {
    if (i < op_run_info->input_size) {
      // Free bprop not used input
      op_run_info->op_grad_info->input_value[i] =
        PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(op_run_info->op_grad_info->input_value[i]);
    } else if (i == op_run_info->input_size) {
      // current op output tensor used in bprop graph, set used_in_bprop_graph which affect follow op free its inputs
      if (op_run_info->op_grad_info->used_in_bprop_graph) {
        PyNativeAlgo::Common::SetOutputUsedInBpropGraph(op_run_info->op_grad_info->out_value);
      } else {
        // Process output, free bprop not used output value
        op_run_info->op_grad_info->out_value =
          PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(op_run_info->op_grad_info->out_value);
        grad_param->out_used_in_bporp_graph = false;
      }
    }
  }

  // Free special op memory
  FreeSpecialOpValue(op_run_info->op_grad_info->op_prim->name(), op_run_info, &op_run_info->op_grad_info->out_value);
}

GradParamPtr CreateOpGradParam(const FrontendOpRunInfoPtr &op_run_info, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->op_grad_info->out_value = op_run_info->real_out;
  op_run_info->op_grad_info->out_abs = op_run_info->base_op_run_info.abstract;

  auto grad_param = std::make_shared<GradParam>(op_run_info->op_grad_info, top_cell->use_dynamic_shape_process());
  FreeUselessValue(op_run_info, grad_param, top_cell);
  return grad_param;
}

void CloneParameter(const AnfNodePtr &node, const KernelGraphPtr &new_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_graph);
  auto old_param = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(old_param);
  auto new_param = new_graph->add_parameter();
  new_param->set_name(old_param->name());
  if (auto t = PyNativeAlgo::Common::GetTensorFromParam(old_param); t != nullptr) {
    const auto &param_info = t->param_info();
    if (param_info != nullptr) {
      const auto &param_name = param_info->name();
      new_param->set_name(param_name);
      new_param->debug_info()->set_name(param_name);
    }
    new_param->set_default_param(t);
  }
  new_param->set_abstract(old_param->abstract());
  new_param->set_scope(old_param->scope());
}

KernelGraphPtr CloneKernelGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Begin clone kernel graph";
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto new_graph = std::make_shared<session::KernelGraph>();
  const auto &params = kernel_graph->parameters();
  for (auto &param : params) {
    CloneParameter(param, new_graph);
  }
  auto out = InlineClone(kernel_graph, new_graph, new_graph->parameters());
  new_graph->set_output(out);
  PyNativeAlgo::Common::FreeFuncGraphForwardNodes(func_graph);
  return new_graph;
}

std::string GetInputArgsId(const py::args &args) {
  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) + "_";
  }
  return input_args_id;
}

void SetCustomBpropInputs(const py::object &obj, const InputArgsInfoPtr &input_args_info) {
  if (py::hasattr(obj, kUsedBpropInputs)) {
    py::object object = py::getattr(obj, kUsedBpropInputs);
    if (!py::isinstance<py::tuple>(object) && !py::isinstance<py::list>(object)) {
      MS_LOG(EXCEPTION) << "For cell bprop, used bprop inputs sholud be tuple or list";
    }
    auto used_bprop_inputs = py::cast<py::tuple>(object);
    std::unordered_set<int64_t> used_inputs;
    for (size_t i = 0; i < used_bprop_inputs.size(); ++i) {
      const auto value = PyNativeAlgo::DataConvert::PyObjToValue(used_bprop_inputs[i]);
      MS_EXCEPTION_IF_NULL(value);
      int used_index = GetValue<int64_t>(value);
      (void)used_inputs.insert(used_index);
    }
    const size_t input_size = input_args_info->input_arg_value_vec.size();
    for (size_t i = 0; i < input_size; ++i) {
      const auto &input_value = input_args_info->input_arg_value_vec[i];
      if (used_inputs.find(i) == used_inputs.end()) {
        auto fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(input_value);
        input_args_info->input_arg_value_vec[i] = fake_value;
      }
    }
    if (used_inputs.find(input_size) == used_inputs.end()) {
      auto fake_value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(input_args_info->out_value);
      input_args_info->out_value = fake_value;
    }
  }

  if (py::hasattr(obj, kInternalParams)) {
    py::object weights = py::getattr(obj, kInternalParams);
    if (py::isinstance<py::tuple>(weights) || py::isinstance<py::list>(weights)) {
      auto weights_tuple = py::cast<py::tuple>(weights);
      for (size_t i = 0; i < weights_tuple.size(); ++i) {
        const auto value = PyNativeAlgo::DataConvert::PyObjToValue(weights_tuple[i]);
        auto tensor = value->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        (void)input_args_info->input_arg_value_vec.emplace_back(tensor);
        (void)input_args_info->input_arg_id_vec.emplace_back(tensor->id());
      }
    }
  }
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
    MS_LOG(EXCEPTION) << "Stack top cell stack is empty";
  }
  MS_LOG(DEBUG) << "Pop top cell " << top_cell_stack_.top() << " on top cell stack";
  top_cell_stack_.pop();
  TopCellInfoPtr top_cell = nullptr;
  if (!top_cell_stack_.empty()) {
    top_cell = top_cell_stack_.top();
  }
  top_cell == nullptr ? MS_LOG(DEBUG) << "Top cell stack has no top cell"
                      : MS_LOG(DEBUG) << "Top cell stack size " << top_cell_stack_.size();
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
  // Convert input args to parameters for top cell graph in construct.
  std::vector<ValuePtr> input_param_values;
  const auto &input_value = input_args_info->input_arg_value_vec;
  if (input_args_info->input_size != 0 && input_value.empty()) {
    MS_LOG(EXCEPTION) << "Input value is empty";
  }

  AbstractBasePtrList abs_list;
  for (size_t i = 0; i < input_args_info->input_size; ++i) {
    const auto &v = input_value[i];
    auto param_i_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(v->ToAbstract());
    if (!top_cell()->is_bprop_need_get_forward_graph()) {
      (void)PyNativeAlgo::Common::SetValueGradInfo(v, top_cell(), InputType::kInput);
      (void)input_param_values.emplace_back(v);
      (void)abs_list.emplace_back(param_i_abs);
    }
    RecordForwardGraphForInput(v, input_args_info->input_arg_id_vec[i], param_i_abs);
  }
  if (top_cell_->is_bprop_need_get_forward_graph()) {
    MS_LOG(DEBUG) << "Run bprop function, no need do prepare for grad";
    return;
  }
  // If New cellid come up, bprop graph use cnode for reusing
  if (IsCreateIrGrad()) {
    top_cell_->set_is_ir_grad(true);
  }
  if (top_cell_->is_ir_grad()) {
    top_cell_->set_auto_grad_cell_ptr(
      std::make_shared<autograd::IrGrad>(input_param_values, abs_list, op_num_in_bprop_graph_ * kContainerRatio,
                                         !top_cell_->is_high_order_top_cell(), is_run_recompute_));
  } else {
    top_cell_->set_auto_grad_cell_ptr(
      std::make_shared<autograd::FuncGrad>(input_param_values, op_num_in_bprop_graph_ * kContainerRatio,
                                           !top_cell_->is_high_order_top_cell(), is_run_recompute_));
  }
}

bool GradExecutor::IsCreateIrGrad() {
  if (already_run_top_cell_.find(top_cell_->already_run_cell_id()) == already_run_top_cell_.end()) {
    // If the already run cell id is pipeline top cell map, no need store in already_run_top_cell_ again when run
    // CheckNeedCompileGraph
    if (pipeline_top_cell_map_.find(top_cell_->already_run_cell_id()) == pipeline_top_cell_map_.end()) {
      top_cell_->set_need_compile_graph(true);
      // If top cell can not find in both already_run_top_cell_ and pipeline_top_cell_map_ can be create new ir
      if (!top_cell_->use_dynamic_shape_process()) {
        return true;
      }
    }
    return false;
  }
  return false;
}

void GradExecutor::InitResourceAndDfBuilder(const InputArgsInfoPtr &input_args_info,
                                            bool is_bprop_need_get_forward_graph) {
  MS_LOG(DEBUG) << "InitResourceAndDfBuilder";
  MS_EXCEPTION_IF_NULL(input_args_info);
  forward()->WaitForwardTask();
  // We need wait construct bprop task of outer top cell finish, if the main thread runs quickly when it executes
  // gradnet and clear bprop_queue queue, bprop task of outer top cell may not finish, it will cause not found cnode
  // error.
  WaitBpropTask();
  if (input_args_info->is_grad_topest_cell) {
    MS_LOG(DEBUG) << "Make new topest graph";
    ResetMetaGradInfoForNewTopCell(input_args_info);
    MakeNewTopCell(input_args_info);
  } else if (input_args_info->is_high_order_top_cell) {
    MS_LOG(DEBUG) << "Nested grad graph existed in construct";

    // High-order inputs are uplevel top cell ops output, so need back up meta grad info too.
    for (auto &item : input_args_info->input_arg_value_vec) {
      top_cell_->BackUpValueMetaGradInfo(item);
    }
    ResetMetaGradInfoForNewTopCell(input_args_info);
    MakeNewTopCell(input_args_info);
    // High-order must use ir grad
    top_cell_->set_is_ir_grad(true);
  } else if (is_bprop_need_get_forward_graph) {
    MS_LOG(DEBUG) << "Run custom bprop function and make forward graph";
    // Make top cell just for get forward graph, but no need do anything about grad
    MakeNewTopCell(input_args_info);
    curr_g()->debug_info()->set_name("bprop_forward_graph");
    top_cell_->set_is_bprop_need_get_forward_graph(is_bprop_need_get_forward_graph);
  }

  // Init kPynativeCellPtr with input parameters of top cell
  if (!top_cell_->is_init_kpynative()) {
    auto graph_info_cg = std::make_shared<PyNGraphInfo>();
    top_cell_->SetGraphInfoMap(curr_g(), graph_info_cg);
    HandleInputArgsForTopCell(input_args_info);
    top_cell_->set_init_kpynative(true);
  }
}

void GradExecutor::ResetMetaGradInfoForNewTopCell(const InputArgsInfoPtr &input_args_info) const {
  // To fix the scene that user calls twice forward network with grad flag, and then call grad() interface.
  // We need to clear last top cell's parameters grad info to avoid influencing construct bprop graph of current top
  // cell.
  if (top_cell_ != nullptr) {
    MS_LOG(DEBUG) << "Reset meta grad info for top cell " << top_cell_;
    top_cell_->ResetMetaGradInfo();
  }

  // To fix the scene like 1. net(x1) 2. x2 = deepcopy(x1), 3. net(x2) 3. grad_net(x2). 4. grad_net(x1)
  // x1's auto_grad_meta_data will be copy to x2, x2 grad will use the same auto_grad_meta_data and clear x1's variable
  // and set x2's variable.
  // When execute grad_net(x1), x1's variable will not found, so we need clear input's auto_grad_meta_data before
  // execute.
  for (auto &item : input_args_info->input_arg_value_vec) {
    top_cell_->ClearValueMetaGradInfo(item);
  }
}

void GradExecutor::NewGraphInner(const py::object &obj, const py::args &args) {
  // Run custom bprop function, and bprop function is under high-order
  // If bprop forward graph has been made, new top cell creates severing for it, and current top_cell_ it is.
  bool running_bprop_function = top_cell_ != nullptr && top_cell_->grad_is_running();
  bool is_bprop_need_get_forward_graph = running_bprop_function && top_cell_->is_high_order_top_cell();

  const auto input_args_info = GetInputArgsInfo(obj, args, is_bprop_need_get_forward_graph);
  PushInputArgsInfoStack(input_args_info);
  MS_LOG(DEBUG) << "NewGraphInner start " << args.size() << ", cell_id " << PyNativeAlgo::PyParser::GetIdByPyObj(obj)
                << ", input args info ptr " << input_args_info.get();

  // Make top graph and init resource
  if (input_args_info->is_grad_topest_cell || input_args_info->is_high_order_top_cell ||
      is_bprop_need_get_forward_graph) {
    InitResourceAndDfBuilder(input_args_info, is_bprop_need_get_forward_graph);
  }
}

InputArgsInfoPtr GradExecutor::GetInputArgsInfo(const py::object &obj, const py::args &args,
                                                bool is_bprop_need_get_forward_graph) {
  const auto &input_args_info = std::make_shared<InputArgsInfo>(input_args_info_stack_.empty(), IsHighOrderTopCell());
  ParsePyArgsToInputArgsInfo(input_args_info, obj, args, is_bprop_need_get_forward_graph);

  if (input_args_info->has_custom_bprop) {
    custom_bprop_cell_count_ += 1;
  }
  // CheckAlready run first, grad_order_ will increase 1(highorder scenario)
  // If NetA.set_grad(), so come here first, CheckAlready run later, so grad_order_ need increase 1
  if (input_args_info->is_grad_topest_cell || input_args_info->is_high_order_top_cell) {
    if (grad_order_ == 0) {
      IncreaseGradOrder();
    }
    input_args_info->already_run_cell_id = GetAlreadyRunCellId(input_args_info->cell_id);
    MS_LOG(DEBUG) << "Get already run top cell id " << input_args_info->already_run_cell_id;
    // top_input_args_info_ indicate current running cell info
    top_input_args_info_ = input_args_info;
  }
  return input_args_info;
}

bool GradExecutor::GetTopCellDynamicFlag(const InputArgsInfoPtr &input_args_info,
                                         const std::string &obj_id_with_grad_order) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  // Just has a forward process, and forward is dynamic(by set_inputs)
  if (forward_use_dynamic_shape_process_) {
    MS_LOG(DEBUG) << "Get forward dynamic";
    return true;
  }

  // Set by set_inputs
  if (dynamic_inputs_cells_.find(input_args_info->obj_id) != dynamic_inputs_cells_.end()) {
    MS_LOG(DEBUG) << "Get dynamic from set inputs";
    return true;
  }

  // Dynamic structure
  auto pre_top_cell = GetAlreadyRunTopCell(input_args_info->already_run_cell_id);
  if (pre_top_cell != nullptr && pre_top_cell->use_dynamic_shape_process()) {
    MS_LOG(DEBUG) << "Get dynamic shape from already run top cell";
    return true;
  }

  // Dynamic structure for pipeline top cell
  pre_top_cell = GetPipelineRunTopCell(input_args_info->already_run_cell_id);
  if (pre_top_cell != nullptr && pre_top_cell->use_dynamic_shape_process()) {
    MS_LOG(DEBUG) << "Get dynamic shape from pipeline top cell";
    return true;
  }

  // Dynamic shape
  if (std::any_of(already_run_top_cell_.begin(), already_run_top_cell_.end(),
                  [&obj_id_with_grad_order](const auto &item) {
                    if (item.second != nullptr && item.second->obj_id_with_grad_order() == obj_id_with_grad_order) {
                      return item.second->use_dynamic_shape_process();
                    }
                    return false;
                  })) {
    MS_LOG(DEBUG) << "Get dynamic shape from already run top cell with obj_id_with_grad_order "
                  << obj_id_with_grad_order;
    return true;
  }

  // Dynamic shape for pipeline top cell
  return std::any_of(
    pipeline_top_cell_map_.begin(), pipeline_top_cell_map_.end(), [&obj_id_with_grad_order](const auto &item) {
      const auto &pipe_top_cell_list = item.second;
      if (std::any_of(pipe_top_cell_list.begin(), pipe_top_cell_list.end(),
                      [&obj_id_with_grad_order](const auto &pipe_item) {
                        if (pipe_item != nullptr && pipe_item->obj_id_with_grad_order() == obj_id_with_grad_order) {
                          return pipe_item->use_dynamic_shape_process();
                        }
                        return false;
                      })) {
        MS_LOG(DEBUG) << "Get dynamic shape from pipeline top cell with obj_id_with_grad_order "
                      << obj_id_with_grad_order;
        return true;
      }
      return false;
    });
}

void GradExecutor::MakeNewTopCell(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);

  auto fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name("pynative_forward_graph");
  auto resource = std::make_shared<pipeline::Resource>();

  finded_top_cell_ = nullptr;
  bool new_top_cell_is_pipeline_top_cell = NewTopCellIsPipelineTopCell(input_args_info);

  bool new_top_cell_is_pipeline_high_order =
    input_args_info->is_high_order_top_cell && new_top_cell_is_pipeline_top_cell;
  // If outer layer top cell is also pipeline top cell, top cell stack maybe empty. Here, need push it to top cell stack
  // too when running MakeNestedCnode or running bprop function(and brpop function has anoher grad). Because it is need
  // to known who is outer layer top cell when inner run finished.
  if (top_cell_ != nullptr && top_cell_->is_pipeline_top_cell() &&
      (new_top_cell_is_pipeline_high_order || top_cell_->grad_is_running())) {
    PushTopCellStack(top_cell_);
  }

  const auto &obj_id_with_grad_order = GetAlreadyRunCellId(input_args_info->obj_id);
  MS_LOG(DEBUG) << "Get obj id with grad order " << obj_id_with_grad_order;
  top_cell_ = std::make_shared<TopCellInfo>(
    input_args_info->is_high_order_top_cell, grad_order_, obj_id_with_grad_order, input_args_info->cell_id,
    input_args_info->already_run_cell_id, resource, fg, op_num_in_bprop_graph_ * kContainerRatio);
  top_cell_->set_forward_already_run(true);
  top_cell_->set_input_args_id(input_args_info->input_args_id);
  auto use_dynamic_shape_process = GetTopCellDynamicFlag(input_args_info, obj_id_with_grad_order);
  top_cell_->set_use_dynamic_shape_process(use_dynamic_shape_process);
  top_cell_->set_need_save_dynamic_detect_nodes(
    dynamic_shape()->IsNeedSaveDynamicDetectNodes(top_cell_, use_dynamic_shape_process));
  top_cell_->set_input_args_info(top_input_args_info_);
  if (dynamic_shape()->enable_unknown_shape()) {
    dynamic_shape()->TryChangeTopCellToUnknownShape(top_input_args_info_->obj_id,
                                                    top_input_args_info_->input_arg_base_shape_vec, true);
  }
  top_cell_->set_has_bprop_cut_op(input_args_info->has_custom_bprop);
  top_cell_->set_grad_first(grad_first_);
  grad_first_ = false;
  MS_LOG(DEBUG) << "New top cell, fg ptr " << fg.get() << ", top cell ptr " << top_cell_.get() << " with input args id "
                << top_cell_->input_args_id();

  if (new_top_cell_is_pipeline_top_cell) {
    pipeline_top_cell_map_[input_args_info->already_run_cell_id].emplace_back(top_cell_);
    top_cell_->set_is_pipeline_top_cell(true);
    // If pipeline top cell is high-order, it need to be manage by stack when run MakeNestedCnode, so push it to stack.
    if (top_cell_->is_high_order_top_cell()) {
      PushTopCellStack(top_cell_);
    }
    MS_LOG(DEBUG) << "Create pipeline top cell, input args id " << top_cell_->input_args_id()
                  << ". The pipeline map size now "
                  << pipeline_top_cell_map_[input_args_info->already_run_cell_id].size();
  } else {
    // Common top cell
    PushTopCellStack(top_cell_);
  }
}

bool GradExecutor::NewTopCellIsPipelineTopCell(const InputArgsInfoPtr &input_args_info) {
  // net.set_grad.
  // pipeline, net(input1), grad(net)(input1), net(input2), grad(net)(input2),...
  const auto it = pipeline_top_cell_map_.find(input_args_info->already_run_cell_id);
  if (it != pipeline_top_cell_map_.end()) {
    // First pipeline top cell
    MS_EXCEPTION_IF_CHECK_FAIL(!it->second.empty(), "Pipeline top cel map is empty");

    // net.set_grad
    // grad(net)(input1) -> this will generate a element in already_run_top_cell_ and do a complete grad operation;
    // Then, run another net(input1) -> this will think it do upgrade op info because a complete grad operation have
    // done before; But then run another net(input1) -> this will get pipeline top cell and which should be have 2
    // elements, and they are need compile ir graph because this is the first step for running pipeline top cell.
    // Then, run grad(net)(input1) -> this will find the matched top cell in already_run_top_cell_ because
    // already_run_cell_id is matched, but this is not correct because current process is in pipeline top cell now. So,
    // this will meet a error of auto grad meta.
    // Erase top cell info from already_run_top_cell_ is need.
    auto iter = already_run_top_cell_.find(input_args_info->already_run_cell_id);
    if (iter != already_run_top_cell_.end()) {
      MS_LOG(DEBUG) << "Erase top cell from already run top cell";
      // Need use ir top cell, current top cell in pipeline_top_cell_map_ is func grad. So, need exchange.
      it->second.front() = iter->second;
      it->second.front()->set_need_compile_graph(true);
      already_run_top_cell_.erase(iter);
    }
    it->second.front()->set_is_pipeline_top_cell(true);
    return true;
  }
  // net.set_grad.
  // 1. grad(net)(input), top cell id will include grad_operation_;
  // 2. net(input1), grad(net)(input1), net(input2), grad(net)(input2), ..., top cell id not include grad_operation_.
  // In second step, grad(net)(input) should be pipeline cell too.
  auto iter = std::find_if(pipeline_top_cell_map_.begin(), pipeline_top_cell_map_.end(),
                           [&input_args_info](const auto &iter_pipe) {
                             return input_args_info->already_run_cell_id.find(iter_pipe.first) != std::string::npos;
                           });
  if (iter != pipeline_top_cell_map_.end()) {
    input_args_info->already_run_cell_id = iter->first;
    return true;
  }
  return false;
}

void GradExecutor::SetForwardLastNodeInfo(const ValuePtr &v) const {
  MS_EXCEPTION_IF_NULL(v);
  auto value = v;
  if (v->isa<tensor::CSRTensor>()) {
    auto csr_tensorptr = v->cast<tensor::CSRTensorPtr>();
    value = csr_tensorptr->GetValues();
  } else if (v->isa<tensor::COOTensor>()) {
    auto coo_tensorptr = v->cast<tensor::COOTensorPtr>();
    value = coo_tensorptr->GetValues();
  }
  (void)PyNativeAlgo::Common::SetValueGradInfo(value, top_cell_, InputType::kConstant);
  // Set last output abstract and will be used for sens
  auto fake_v = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(value);
  top_cell()->SetLastOutputValueForwardOutputFlag(fake_v);
  if (forward()->enable_async()) {
    auto auto_grad_cell_ptr = top_cell()->auto_grad_cell_ptr();
    auto task = [auto_grad_cell_ptr, fake_v]() { auto_grad_cell_ptr->UpdateOutputNodeOfTopCell(fake_v); };
    DispatchGradQueueTask(std::move(task));
  } else {
    top_cell()->auto_grad_cell_ptr()->UpdateOutputNodeOfTopCell(fake_v);
  }
}

void GradExecutor::EndGraphInner(const py::object &obj, const py::object &out, const py::args &args) {
  if (input_args_info_stack_.empty()) {
    return;
  }
  const auto input_args_info = input_args_info_stack_.top();
  MS_EXCEPTION_IF_NULL(input_args_info);
  MS_LOG(DEBUG) << "EndGraphInner start " << args.size() << ", cell_id " << PyNativeAlgo::PyParser::GetIdByPyObj(obj)
                << ", input args info ptr " << input_args_info.get();
  if (input_args_info->is_grad_topest_cell) {
    grad_flag_ = false;
  }

  // If there is a custom bprop in the forward running process of the cell, need to do DoGradForCustomBprop;
  // If the top cell is only for obtaining the forward graph, there is no need to do DoGradForCustomBprop.
  bool need_do_custom_bprop_grad = false;
  if (input_args_info->has_custom_bprop && custom_bprop_cell_count_ != 0) {
    --custom_bprop_cell_count_;
    need_do_custom_bprop_grad = custom_bprop_cell_count_ == 0;
  }
  if (!top_cell()->is_bprop_need_get_forward_graph() && need_do_custom_bprop_grad) {
    GetCustomBpropPrim(obj, args, input_args_info);
    runtime::Pipeline::Get().WaitAll();
    input_args_info->out_value = PyNativeAlgo::DataConvert::PyObjToValue(out, false);
    // Recompute need to regardless of non tensor inputs, maybe it is a middle cell and not call EndGraphImpl
    if (input_args_info->is_need_recompute) {
      input_args_info->out_value =
        ConvertOutputValueToTensor(input_args_info->out_value, !top_cell()->jit_out_has_dict());
    }
    const auto &out_id = PyNativeAlgo::Common::GetIdByValue(input_args_info->out_value);
    SetCustomBpropInputs(obj, input_args_info);
    DoGradForCustomBprop(input_args_info, out_id);
  }

  // Get top cell endgraph
  if (input_args_info->cell_id == top_cell()->cell_id()) {
    runtime::Pipeline::Get().WaitAll();
    if (input_args_info->out_value == nullptr) {
      input_args_info->out_value = PyNativeAlgo::DataConvert::PyObjToValue(out, false);
    }
    MS_LOG(DEBUG) << "Get cell output value " << input_args_info->out_value->ToString();
    EndGraphImpl(input_args_info);
  }
  PopInputArgsInfoStack();
}

void GradExecutor::EndGraphImpl(const InputArgsInfoPtr &input_args_info) {
  auto out_tensor = ConvertOutputValueToTensor(input_args_info->out_value, !top_cell()->jit_out_has_dict());
  std::vector<std::string> output_tensors_id;
  PyNativeAlgo::DataConvert::ConvertValueTensorId(out_tensor, &output_tensors_id);
  top_cell()->set_outputs_ids(std::move(output_tensors_id));
  if (out_tensor != nullptr) {
    input_args_info->out_value = out_tensor;
  }

  // If network runs twice, and one of the runs is an empty network, the following judgment will take effect
  if (!top_cell_->use_dynamic_shape_process()) {
    auto op_grad_info = std::make_shared<OpGradInfo>();
    op_grad_info->input_value = {input_args_info->out_value};
    op_grad_info->input_abs = {
      PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_args_info->out_value->ToAbstract())};
    op_grad_info->op_index = top_cell_->op_index();
    dynamic_shape()->CheckNodeDynamic(top_cell_, op_grad_info);
  }

  // Just only dump the last forward graph or bprop forward graph
  if (save_graphs_ || top_cell_->is_bprop_need_get_forward_graph()) {
    auto output_node =
      GetInput(input_args_info->out_value, PyNativeAlgo::Common::GetIdByValue(input_args_info->out_value));
    curr_g()->set_output(output_node);
    PyNativeAlgo::Common::DumpGraphIR("fg.ir", curr_g());
    MS_LOG(DEBUG) << "Save forward graph";
  }

  if (top_cell_->is_bprop_need_get_forward_graph()) {
    MS_LOG(DEBUG) << "Run bprop no need do grad";
    return;
  }

  // Set sens value for grad
  SetForwardLastNodeInfo(input_args_info->out_value);

  if (input_args_info->is_grad_topest_cell) {
    MS_LOG(DEBUG) << "Cur top last cell " << input_args_info->cell_id;
    top_cell()->ClearCellHookOp();
  }

  // Checkout whether you need to compile graph when each top cell has run finished
  CheckNeedCompileGraph(input_args_info);
  if (!top_cell_->grad_first()) {
    DecreaseGradOrder();
  }
  top_input_args_info_ = input_args_info;
  forward()->ClearNodeAbsMap();
}

void GradExecutor::DoGradForCustomBprop(const InputArgsInfoPtr &input_args_info, const std::string &out_id) const {
  MS_EXCEPTION_IF_NULL(input_args_info);
  MS_EXCEPTION_IF_NULL(input_args_info->custom_bprop_prim);
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->base_op_run_info.op_name = input_args_info->custom_bprop_prim->name();
  op_run_info->op_grad_info->op_prim = input_args_info->custom_bprop_prim;
  op_run_info->op_grad_info->input_value = input_args_info->input_arg_value_vec;
  op_run_info->op_grad_info->weight_size = op_run_info->op_grad_info->input_value.size() - input_args_info->input_size;
  op_run_info->op_grad_info->is_need_recompute = input_args_info->is_need_recompute;
  op_run_info->input_size = input_args_info->input_arg_value_vec.size();
  op_run_info->input_value_id = input_args_info->input_arg_id_vec;
  op_run_info->real_out = input_args_info->out_value;
  op_run_info->out_value_id = out_id;
  op_run_info->base_op_run_info.abstract =
    PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_args_info->out_value->ToAbstract());
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    const auto &value = input_args_info->input_arg_value_vec[i];
    (void)op_run_info->op_grad_info->input_abs.emplace_back(
      PyNativeAlgo::Common::SetAbstractValueToAnyValue(value->ToAbstract()));
    op_run_info->op_grad_info->input_value_grad_type[i] =
      PyNativeAlgo::Common::SetValueGradInfo(value, top_cell_, InputType::kConstant);
  }
  op_run_info->op_grad_info->output_size = PyNativeAlgo::Common::GetValueSize(op_run_info->real_out);
  (void)PyNativeAlgo::Common::SetValueGradInfo(op_run_info->real_out, top_cell_, InputType::kOpOutput);
  DoOpGrad(op_run_info);
  RecordForwardGraph(op_run_info);
}

void GradExecutor::GetCustomBpropPrim(const py::object &obj, const py::args &args,
                                      const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  MS_LOG(DEBUG) << "Do grad for custom bprop";
  py::function bprop_func = py::getattr(obj, parse::CUSTOM_BPROP_NAME);
  py::object code_obj = py::getattr(bprop_func, "__code__");
  py::object co_name = py::getattr(code_obj, "co_name");
  if (std::string(py::str(co_name)) == "staging_specialize") {
    MS_LOG(EXCEPTION) << "Decorating bprop with '@jit' is not supported.";
  }

  auto fake_prim = std::make_shared<PrimitivePy>(prim::kPrimHookBackward->name());
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (py::isinstance<Cell>(obj)) {
    const auto &cell_ptr = obj.cast<CellPtr>();
    input_args_info->is_need_recompute = cell_ptr->HasAttr(kNeedRecompute);
    fake_prim->set_bprop_cls_name(cell_ptr->name());
  }
  if (input_args_info->input_arg_value_vec.empty()) {
    for (size_t i = 0; i < args.size(); ++i) {
      (void)input_args_info->input_arg_value_vec.emplace_back(PyNativeAlgo::DataConvert::PyObjToValue(args[i]));
    }
  }
  fake_prim->AddBackwardHookFn(0, bprop_func);

  (void)fake_prim->AddAttr("cell_id", MakeValue(input_args_info->cell_id));
  (void)fake_prim->AddAttr(parse::CUSTOM_BPROP_NAME, MakeValue(true));

  input_args_info->custom_bprop_prim = fake_prim;
}

void GradExecutor::ClearPreTopCell(const TopCellInfoPtr &new_top_cell, bool is_need_clear_device_mem) {
  MS_EXCEPTION_IF_NULL(new_top_cell);
  // Clear already run top cell and device mem
  for (auto iter = already_run_top_cell_.begin(); iter != already_run_top_cell_.end();) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->obj_id_with_grad_order() == new_top_cell->obj_id_with_grad_order()) {
      if (is_need_clear_device_mem) {
        iter->second->ClearDeviceMemory();
        (void)need_gc_top_cell_list_.emplace_back(iter->second);
      }
      iter = already_run_top_cell_.erase(iter);
    } else {
      (void)iter++;
    }
  }
}

void GradExecutor::CheckNeedCompileGraph(const InputArgsInfoPtr &input_args_info) {
  const auto &already_top_cell_id = top_cell()->already_run_cell_id();
  bool is_new_cell_id = false;
  // Get new cell id for common grad, even for dynamic shapes, first step will come in too.
  if (top_cell_->need_compile_graph()) {
    MS_LOG(DEBUG) << "Cell " << already_top_cell_id << " has never been ran, need compile graph";
    // net.set_grad.
    // 1. grad(net)(input), top cell id will include grad_operation_;
    // 2. net(input), grad(net)(input), top cell id not include grad_operation_. But, just only keep one for cccurate
    // find when call GetTopCell
    auto it = std::find_if(
      already_run_top_cell_.begin(), already_run_top_cell_.end(),
      [&already_top_cell_id](const auto &item) { return item.first.find(already_top_cell_id) != std::string::npos; });
    if (it != already_run_top_cell_.end()) {
      already_run_top_cell_.erase(it);
    }
    already_run_top_cell_[already_top_cell_id] = top_cell_;
    is_new_cell_id = true;
  }
  // First step and first top cell prepare for pipeline if it is
  const auto it = pipeline_top_cell_map_.find(top_cell_->already_run_cell_id());
  if (it == pipeline_top_cell_map_.end()) {
    MS_LOG(DEBUG) << "Prepare the first top cell to be pipeline top cell";
    top_cell_->set_need_compile_graph(true);
    pipeline_top_cell_map_[already_top_cell_id].emplace_back(top_cell_);
    // If the top cell is the first top cell, pipeline top cell backup one and return;
    // But, if top cell is not pipeline(run by already run top cell map), in the second step, can not return here, which
    // should go down to judge compile status.
    if (is_new_cell_id) {
      return;
    }
  }
  // Get pipeline top cell in first step, and judge by first top cell have completed a backward run
  if (top_cell_->is_pipeline_top_cell()) {
    MS_EXCEPTION_IF_CHECK_FAIL(!it->second.empty(), "Pipeline top cel map is empty");
    if (!it->second.front()->is_finish_backward()) {
      top_cell_->set_need_compile_graph(true);
      // Get dynamic structure
      if (top_cell_->use_dynamic_shape_process()) {
        it->second.front()->set_use_dynamic_shape_process(true);
      }
      MS_LOG(DEBUG) << "Get pipeline top cell has never been ran, input args " << top_cell_->input_args_id();
      return;
    }
  }

  // Older top cell id or dynamic shape
  MS_EXCEPTION_IF_NULL(input_args_info);
  // In high-order situations, the internal top cell has changed, but the outer top cell remains unchanged. Then outer
  // bprop graph needs to compile again
  if (top_cell_->use_dynamic_shape_process() || top_cell_->force_top_cell_compile()) {
    // Function need compiler every time.
    top_cell_->use_dynamic_shape_process() ? MS_LOG(DEBUG) << "The graph is dynamic, need to compile graph again"
                                           : MS_LOG(DEBUG) << "Force outer graph compile graph";
    if (!top_cell_->is_pipeline_top_cell()) {
      auto has_higher_order = std::any_of(already_run_top_cell_.begin(), already_run_top_cell_.end(),
                                          [](const auto &elem) { return elem.second->is_high_order_top_cell(); });
      ClearPreTopCell(top_cell_, input_args_info->is_grad_topest_cell && !has_higher_order);
      already_run_top_cell_[already_top_cell_id] = top_cell_;
    } else {
      MS_LOG(DEBUG) << "Get pipeline top cell, input args " << top_cell_->input_args_id();
    }
    top_cell_->set_need_compile_graph(true);
    top_cell_->set_force_top_cell_compile(false);
  } else {
    MS_LOG(DEBUG) << "Cell " << already_top_cell_id << " no need to compile graph again";
    if (!top_cell_->is_pipeline_top_cell()) {
      top_cell_->set_need_compile_graph(false);
      auto pre_top_cell = GetAlreadyRunTopCell(already_top_cell_id);
      MS_EXCEPTION_IF_NULL(pre_top_cell);
      pre_top_cell->set_input_args_id(top_cell_->input_args_id());
      // In high order situations, the internal top cell remains unchanged, but the external top cell has changed. Then
      // the graph info of the internal top cell needs to be updated so that the external top cell can perceive it.
      if (!input_args_info->is_grad_topest_cell) {
        pre_top_cell->SetGraphInfoMap(pre_top_cell->fg(), top_cell_->graph_info_map().at(top_cell_->fg()));
      }
      pre_top_cell->set_forward_already_run(true);
      pre_top_cell->set_input_args_info(input_args_info);
      top_cell_stack_.top() = pre_top_cell;
    } else {
      MS_LOG(DEBUG) << "Get pipeline top cell, input args " << top_cell_->input_args_id();
    }
  }
}

TopCellInfoPtr GradExecutor::GetAlreadyRunTopCell(const std::string &already_run_cell_id) const {
  const auto it = already_run_top_cell_.find(already_run_cell_id);
  if (it != already_run_top_cell_.end()) {
    return it->second;
  }
  return nullptr;
}

TopCellInfoPtr GradExecutor::GetPipelineRunTopCell(const std::string &already_run_cell_id) const {
  const auto it = pipeline_top_cell_map_.find(already_run_cell_id);
  if (it != pipeline_top_cell_map_.end()) {
    return it->second.front();
  }
  return nullptr;
}

TopCellInfoPtr GradExecutor::GetPipelineTopCell(const std::string &already_run_cell_id,
                                                const std::string &input_args_id, bool is_reverse_match) const {
  for (const auto &t : pipeline_top_cell_map_) {
    bool is_find = is_reverse_match ? t.first.find(already_run_cell_id) != std::string::npos
                                    : already_run_cell_id.find(t.first) != std::string::npos;
    if (is_find) {
      // If finish backward, skip the first ir top cell
      auto begin =
        !t.second.empty() && t.second.front()->is_finish_backward() ? t.second.begin() + 1 : t.second.begin();
      auto input_args_id_with_top_cell =
        std::find_if(begin, t.second.end(), [input_args_id](const TopCellInfoPtr &pipe_top_cell) {
          return input_args_id == pipe_top_cell->input_args_id();
        });
      if (input_args_id_with_top_cell == t.second.end()) {
        MS_LOG(DEBUG) << "Can not find top cell with input args id " << input_args_id;
        continue;
      }
      MS_LOG(DEBUG) << "Find pipeline top cell with input args id " << input_args_id;
      return *input_args_id_with_top_cell;
    }
  }
  MS_LOG(DEBUG) << "Can not find cell id " << already_run_cell_id << " in pipeline top cell map";
  return nullptr;
}

void GradExecutor::ErasePipelineTopCell(const std::string &already_run_cell_id, const std::string &input_args_id,
                                        bool is_pipeline_ir_top_cell) {
  for (auto &t : pipeline_top_cell_map_) {
    if (already_run_cell_id.find(t.first) == std::string::npos) {
      continue;
    }

    // If top cell is pipeline ir top cell and finish backward, skip the first ir top cell
    auto begin = !is_pipeline_ir_top_cell && !t.second.empty() && t.second.front()->is_finish_backward()
                   ? t.second.begin() + 1
                   : t.second.begin();
    auto input_args_id_with_top_cell = std::find_if(
      begin, t.second.end(),
      [input_args_id](const TopCellInfoPtr &pipe_top_cell) { return input_args_id == pipe_top_cell->input_args_id(); });
    if (input_args_id_with_top_cell == t.second.end()) {
      MS_LOG(DEBUG) << "Can not find top cell with input args id " << input_args_id;
      continue;
    }
    MS_LOG(DEBUG) << "Erase pipeline top cell " << input_args_id_with_top_cell->get() << " with input args id "
                  << input_args_id << ". The pipeline map size now " << t.second.size() - 1;
    t.second.erase(input_args_id_with_top_cell);
    if (t.second.empty()) {
      MS_LOG(DEBUG) << "Pipeline top cell map with already run cell id " << already_run_cell_id
                    << " is empty, erase it from the pipeline map";
      pipeline_top_cell_map_.erase(t.first);
    }
    return;
  }
}

py::object GradExecutor::RunGrad(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                                 const py::object &grad_position, const py::args &args) {
  // Wait forward task finish.
  runtime::Pipeline::Get().WaitAll();

  GetTopCellWithInputArgsRespectTo(grad, obj, args);
  MS_EXCEPTION_IF_NULL(top_cell_);
  MS_LOG(DEBUG) << "Run top cell " << top_cell_;

  // Inputs args info must be update to current even no need compile graph again
  top_input_args_info_ = top_cell_->input_args_info();
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  // Set sens
  SetSensValue(grad, top_input_args_info_, args, !top_cell_->jit_out_has_dict());

  MS_LOG(DEBUG) << "RunGrad start " << args.size() << ", cell_id " << top_input_args_info_->cell_id
                << ", input args info ptr " << top_input_args_info_.get();

  if (!top_cell_->need_compile_graph()) {
    MS_LOG(DEBUG) << "No need compile graph, graph is ir_grad " << top_cell_->is_ir_grad();
    // If no need compile, we can clear construct bprop queue.
    (void)need_gc_top_cell_list_.emplace_back(top_cell_);
    ClearBpropTask();
    top_cell_->ClearMetaGradInfo();
    AsyncClearTopCell();

    // If top cell is pipeline top cell, finded_top_cell_ will be itself;
    // Otherwise, it ir top cell in already_run_top_cell_;
    if (!ReplacePipelineTopCellForwardOutput()) {
      finded_top_cell_->set_shadow_top_cell(top_cell_.get());
      top_cell_ = finded_top_cell_;
      finded_top_cell_ = nullptr;
    }
    top_cell_->UpdateTopCellInfo(false, false, false);
    return RunGradGraph();
  }

  MS_LOG(DEBUG) << "Need compile graph, graph is ir_grad " << top_cell_->is_ir_grad();
  WaitBpropTask();
  AsyncClearTopCell();
  top_cell_ = finded_top_cell_;
  finded_top_cell_ = nullptr;
  op_num_in_bprop_graph_ = top_cell_->op_index();
  top_cell_->set_grad_operation(grad_operation_);
  top_cell_->UpdateTopCellInfo(false, false, true);
  top_cell_->ResumeMetaGradInfo();
  SetBpropGraphJitLevel(obj);
  bool weight_param_is_tuple = true;
  auto w_args = GetWeightsArgs(weights, &weight_param_is_tuple);
  auto p_args = GetGradPositionArgs(grad_position, grad->get_by_position_);
  autograd::GradAttr grad_attr(grad->get_all_, grad->get_by_list_, grad->sens_param_, grad->get_by_position_,
                               weight_param_is_tuple);
  if (top_cell_->is_ir_grad()) {
    GetGradGraph(grad_attr, w_args, p_args);
    return RunGradGraph();
  }
  return RunGradFunc(grad_attr, w_args, p_args);
}

std::string GradExecutor::GetAlreadyRunCellId(const std::string &obj_id) const {
  std::string already_run_cell_id(obj_id);
  already_run_cell_id += "_" + std::to_string(grad_order_ == 0 ? 1 : grad_order_);
  already_run_cell_id += "_" + grad_operation_;
  return already_run_cell_id;
}

void GradExecutor::GetTopCellWithInputArgsRespectTo(const prim::GradOperationPtr &grad, const py::object &obj,
                                                    const py::args &args) {
  auto reset_flag = [this]() {
    if (finded_top_cell_->is_pipeline_top_cell()) {
      top_cell_ = finded_top_cell_;
    } else if (top_cell_ != nullptr &&
               finded_top_cell_->already_run_cell_id().find(top_cell_->already_run_cell_id()) == std::string::npos) {
      // NetA.set_grad, NetB.set_grad
      // then, run NetA(input), NetB(input) for get loss, and then run grad(NetA)(input), grad(NetB)(input).
      // But, when run grad(NetA)(input), finded_top_cell_ is grad of NetA, but top cell is grad(NetB)(input), which is
      // not matched, so need to do exchange.
      // Need do meta grad info reset for NetB because NetB run after NetA and NetB not do this operation in
      // MakeNewTopCell. If have same inputs or weight parameters, auto grad meta maybe meet nullptr.
      top_cell_->ResetMetaGradInfo();
      top_cell_ = finded_top_cell_;
    }
  };

  if (finded_top_cell_ != nullptr) {
    reset_flag();
    return;
  }
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
  const auto &id_v = PyNativeAlgo::PyParser::GetArgsIdAndValue(args_without_sens);
  const auto &cell_id =
    PyNativeAlgo::Common::GetCellId(PyNativeAlgo::PyParser::GetIdByPyObj(obj), id_v.first, id_v.second);
  const auto &already_run_cell_id = GetAlreadyRunCellId(cell_id);
  const auto &input_args_id = GetInputArgsId(args_without_sens);
  MS_LOG(DEBUG) << "Get input cell id " << cell_id << " and already run cell id " << already_run_cell_id
                << ", input args id " << input_args_id;
  finded_top_cell_ = GetTopCell(already_run_cell_id, input_args_id);
  MS_EXCEPTION_IF_NULL(finded_top_cell_);
  reset_flag();
}

bool GradExecutor::ReplacePipelineTopCellForwardOutput() {
  // If top cell is pipeline top cell, need to get its ir top cell
  if (!top_cell_->is_pipeline_top_cell()) {
    return false;
  }
  auto pipeline_ir_top_cell = GetPipelineRunTopCell(top_cell_->already_run_cell_id());
  if (pipeline_ir_top_cell == nullptr) {
    MS_LOG(EXCEPTION) << "Can not find pipeline ir top cell " << top_cell_->already_run_cell_id()
                      << " in pipeline top cell map";
  }
  UpdatePipelineTopCellFowardTensor(pipeline_ir_top_cell->replace_info(), top_cell_->replace_info());
  pipeline_ir_top_cell->set_shadow_top_cell(top_cell_.get());
  top_cell_ = pipeline_ir_top_cell;
  MS_LOG(DEBUG) << "Run no need compile ir top cell " << top_cell_;
  return true;
}

void GradExecutor::GetGradGraph(const autograd::GradAttr &grad_attr, const std::vector<tensor::BaseTensorPtr> &w_args,
                                const std::vector<size_t> &p_args) {
  // Get bprop graph of top cell
  auto bprop_graph = GetBpropGraph(grad_attr, w_args, p_args);
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph, true);
  bprop_graph->ResetOwnNodes();
  // If clear autogradcell before resetownnode, it may corrupt.
  AsyncClearAutoGradCell(top_cell());
  if (top_cell()->has_control_flow()) {
    (void)opt::EnvironConversion(resource);
  }
  if (top_input_args_info_->sens_type == SensType::kDict) {
    PyNativeAlgo::Common::ProcessDictParam(bprop_graph, top_input_args_info_->input_size);
  } else if (top_input_args_info_->sens_type == SensType::kTuple) {
    PyNativeAlgo::Common::ProcessTupleParam(bprop_graph, top_input_args_info_->input_size);
  }
  if (top_cell()->jit_out_has_dict()) {
    MS_LOG(DEBUG) << "Jit out is dict, need convert make dict to pyexecute";
    (void)mindspore::opt::RewriterAfterOptA(resource->func_graph(), resource);
  }
  top_cell()->SaveForwardOutputTensorInfoInBpropGraph(resource->func_graph());
  PyNativeAlgo::Common::DumpGraphIR("launch_bprop_graph.ir", bprop_graph);
  resource->SetBackendAsync([]() { return compile::CreateBackend(); });
  MS_LOG(DEBUG) << "Start task emit action";
  (void)TaskEmitAction(resource);
  MS_LOG(DEBUG) << "Start execute action";
  (void)ExecuteAction(resource);
  top_cell()->UpdateTopCellInfo(false, false, true);
  resource->Clean();
}

std::vector<tensor::BaseTensorPtr> GradExecutor::GetWeightsArgs(const py::object &weights,
                                                                bool *weight_param_is_tuple) const {
  std::vector<tensor::BaseTensorPtr> w_args;
  if (py::hasattr(weights, "__parameter_tuple__")) {
    const auto &weights_tuple = weights.cast<py::tuple>();
    MS_LOG(DEBUG) << "Get weights tuple size " << weights_tuple.size();
    for (size_t i = 0; i < weights_tuple.size(); ++i) {
      const auto value = PyNativeAlgo::DataConvert::PyObjToValue(weights_tuple[i]);
      auto tensor = value->cast<tensor::BaseTensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      (void)w_args.emplace_back(tensor);
    }
  } else {
    MS_LOG(DEBUG) << "No parameter tuple get, try get weights params by input weight";
    if (py::isinstance<py::tuple>(weights) || py::isinstance<py::list>(weights)) {
      auto weights_tuple = py::cast<py::tuple>(weights);
      for (size_t i = 0; i < weights_tuple.size(); ++i) {
        const auto value = PyNativeAlgo::DataConvert::PyObjToValue(weights_tuple[i]);
        auto tensor = value->cast<tensor::BaseTensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        (void)w_args.emplace_back(tensor);
      }
    } else if (!py::isinstance<py::none>(weights)) {
      // Single input
      const auto value = PyNativeAlgo::DataConvert::PyObjToValue(weights);
      auto tensor = value->cast<tensor::BaseTensorPtr>();
      (void)w_args.emplace_back(tensor);
      MS_EXCEPTION_IF_NULL(tensor);
      *weight_param_is_tuple = false;
    } else {
      return GetDefaultWeights();
    }
  }
  return w_args;
}

std::vector<tensor::BaseTensorPtr> GradExecutor::GetDefaultWeights() const {
  std::vector<tensor::BaseTensorPtr> w_args;
  for (const auto &params : top_cell()->param_grad_info()) {
    const auto &tensor = params.first;
    if (tensor->is_parameter()) {
      (void)w_args.emplace_back(tensor);
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

void GradExecutor::CheckParamShapeAndType(const ParameterPtr &param_node, const abstract::AbstractBasePtr &ir_abs,
                                          const abstract::AbstractBasePtr &input_abs) const {
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(ir_abs);
  MS_EXCEPTION_IF_NULL(input_abs);
  const auto &ir_shape = ir_abs->BuildShape()->ToString();
  const auto &input_shape = input_abs->BuildShape()->ToString();
  if (input_shape != "()" && ir_shape != "()") {
    if (input_shape != ir_shape) {
      // Sens shape in ir graph is determined by graph output, so it can be dynamic shape; But input shape is
      // determined by user input, which could not be dynamic shape.
      if (param_node->debug_info()->name() != "sens" || !ir_abs->BuildShape()->IsDynamic()) {
        MS_EXCEPTION(ValueError) << "The shape should be " << ir_shape << ", but got " << input_shape << ", "
                                 << param_node->DebugString() << ", ir_abs " << ir_abs->ToString() << ", input_abs "
                                 << input_abs->ToString();
      }
    }
    const auto &ir_dtype = ir_abs->BuildType()->ToString();
    const auto &input_dtype = input_abs->BuildType()->ToString();
    if (input_dtype != ir_dtype) {
      MS_EXCEPTION(TypeError) << "The dtype should be " << ir_dtype << ", but got " << input_dtype << ", "
                              << param_node->DebugString();
    }
  }
}

void GradExecutor::UpdateParamAbsByArgs(const std::vector<ValuePtr> &input_args,
                                        const FuncGraphPtr &bprop_graph) const {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  const auto &bprop_params = bprop_graph->parameters();
  // bprop_params include inputs, parameters and sens, should be more than inputs size
  if (bprop_params.size() < input_args.size()) {
    MS_LOG(EXCEPTION) << "Df parameters size " << bprop_params.size() << " less than " << input_args.size();
  }
  size_t index = 0;
  for (const auto &param : bprop_params) {
    auto param_node = param->cast<ParameterPtr>();
    if (param_node->has_default()) {
      MS_EXCEPTION_IF_NULL(param_node->abstract());
    } else {
      const auto &input_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_args[index]->ToAbstract());
      if (param_node->abstract() != nullptr) {
        CheckParamShapeAndType(param_node, param_node->abstract(), input_abs);
      } else {
        param_node->set_abstract(input_abs);
      }
      ++index;
    }
  }
}

FuncGraphPtr GradExecutor::GetBpropGraph(const autograd::GradAttr &grad_attr,
                                         const std::vector<tensor::BaseTensorPtr> &w_args,
                                         const std::vector<size_t> &p_args) {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  const auto &auto_grad_cell = std::dynamic_pointer_cast<autograd::IrGrad>(top_cell()->auto_grad_cell_ptr());
  MS_EXCEPTION_IF_NULL(auto_grad_cell);
  // Update bprop_graph_run_by_single_op for bprop graph, if it is true, pass like ConvertMakeTupleInputToDynamicInput
  // will not take effect
  auto_grad_cell->set_bprop_graph_run_by_single_op(top_cell()->use_dynamic_shape_process());
  FuncGraphPtr bprop_graph = auto_grad_cell->Finish(w_args, p_args, grad_attr);
  MS_LOG(DEBUG) << "Top graph input params size " << top_input_args_info_->input_arg_value_vec.size();
  UpdateParamAbsByArgs(top_input_args_info_->input_arg_value_vec, bprop_graph);
  if (top_cell()->need_do_final_opt()) {
    bprop_graph = BpropGraphFinalOpt(bprop_graph, top_cell()->has_control_flow());
  }
  if (top_input_args_info_->is_high_order_top_cell) {
    MS_LOG(DEBUG) << "Get high grad";
    top_cell()->resource()->set_optimize_graph(bprop_graph);
    bool has_bprop_cut = bprop_graph->has_flag(kFlagPyNativeBpropGraphWithBpropCut);
    if (bprop_graph->isa<session::KernelGraph>()) {
      bprop_graph = CloneKernelGraph(bprop_graph);
    } else {
      bprop_graph = BasicClone(bprop_graph);
    }
    if (has_bprop_cut) {
      bprop_graph->set_flag(kFlagPyNativeBpropGraphWithBpropCut, true);
    }
    PyNativeAlgo::Common::ReplaceCNodeWithValueNode(bprop_graph);
  } else {
    top_cell()->resource()->set_optimize_graph(bprop_graph);
  }
  if (bprop_graph->has_flag(kFlagIsControlFlow)) {
    top_cell()->set_has_control_flow(true);
  }
  if (top_cell()->has_control_flow()) {
    bprop_graph = LiftingClone(bprop_graph);
  }
  bprop_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop_graph->set_flag(kFlagIsPynativeBpropGraph, true);
  bprop_graph->set_flag(kFlagPyNativeBpropGraphIsDynamic, top_cell()->use_dynamic_shape_process());

  // Update bprop cut flag. Has two scenario:
  // 1. kHookBackwardName or kCellBackwardHookName
  // 2. Custom op bprop(set in auto_grad.cc by kFlagPyNativeBpropGraphWithBpropCut)
  bprop_graph->set_flag(kFlagPyNativeBpropGraphWithBpropCut,
                        bprop_graph->has_flag(kFlagPyNativeBpropGraphWithBpropCut) || top_cell()->has_bprop_cut_op());

  // Update run graph by single op flag. Has two scenario:
  // 1. Dynamic shape(or structure) or Dynamic structure
  // 2. Has bprop cut op
  // If set_inputs, but has constrol flow, we need run by actor.
  bprop_graph->set_flag(kFlagEnableRunGraphBySingleOp,
                        auto_grad_cell->bprop_graph_run_by_single_op() && !bprop_graph->has_flag(kFlagIsControlFlow));
  top_cell()->set_use_dynamic_shape_process(bprop_graph->has_flag(kFlagEnableRunGraphBySingleOp));
  if (top_cell()->has_call_graph()) {
    bprop_graph->set_flag(kFlagPyNativeWithJitCallGraph, true);
  }
  bool has_control_flow = top_cell()->has_control_flow();
  bprop_graph->set_flag(kFlagIsPyNativeBpropKernelGraph, !has_control_flow);
  // Control graph will generate kernel graph in compile graphs again. Graph id is conflict with default id 0
  if (has_control_flow) {
    auto kernel_graph = bprop_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    kernel_graph->set_graph_id(kernel_graph_id_for_control_flow());
  }
  return bprop_graph;
}

bool GradExecutor::NeedIncreaseGradOrder(const std::string &obj_id) {
  // top_cell_ == nullptr means call by grad first
  // top_cell_->obj_id_with_grad_order() include obj_id and grad_order
  // If top_cell_->obj_id_with_grad_order().find(obj_id) == std::string::npos, means current cell is not top cell,
  // another cell or function needs to get grad, so high-order comes up
  if (top_cell_ == nullptr || top_cell_->obj_id_with_grad_order().find(obj_id + "_") == std::string::npos) {
    IncreaseGradOrder();
    return true;
  }
  return false;
}

py::object GradExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj,
                                         const py::object &weights, const py::object &grad_hash_id,
                                         const py::args &args) {
  const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);

  // The rule of grad order is:
  // scenarios 1. net.set_grad, net(input) calls first, increase 1 before MakeNewTopCell, and decrease 1 when running to
  // EndGraphImpl, indicating that a complete bprop graph construction is completed; Then call grad(net)(input) is won't
  // be affected and get forward_run is true.
  // scenarios 2. If grad(net)(input) calls first, then increase 1 before MakeNewTopCell and decrease 1 in Rungrad. The
  // reason for this design is that if grad(net)(input) calls first and decrease 1 in EndGraphImpl, it will cause
  // matching problems during RunGrad due to the presence of Gradopration information in already_run_cell_id is not the
  // same. Gradopration information include grad order for distinguish high-order.
  // Use flag: grad_first_ for distinguish this two scenarios. If scenarios 1 is taked, grad_first_ will not take
  // effect, otherwise, it works.
  bool neee_increase_grad_order = NeedIncreaseGradOrder(obj_id);

  // Include grad position
  std::string grad_position;
  if (!py::isinstance<py::none>(grad_hash_id)) {
    grad_position = std::string(py::str(grad_hash_id));
  }

  // Include weights id
  // Taking the two times derivative of the a same network, the weights in the grad(net, xxx) api, xxx first time passed
  // as param1, and the second time passed as param2. Except for this difference, everything else is the same.
  // At this point, the cell id is consistent, and the forward process is also exactly the same. If the weight ID does
  // not participate in the comparison, it will not be able to distinguish between these two different derivative
  // calculations
  const auto &weights_obj_id = GetWeightsObjIdsByWeights(weights);

  // Include grad operation
  grad_operation_ = std::to_string(grad->get_all_) + std::to_string(grad->get_by_list_) +
                    std::to_string(grad->sens_param_) + grad_position + weights_obj_id;

  auto input_args_id = GetInputArgsId(args);
  // Under the condition that the stack is empty (forward process completed or no forward process),
  // check whether need to run forward process
  bool forward_run = false;
  if (input_args_info_stack_.empty()) {
    const auto &id_v = PyNativeAlgo::PyParser::GetArgsIdAndValue(args);
    auto cell_id = PyNativeAlgo::Common::GetCellId(obj_id, id_v.first, id_v.second);
    const auto &check_already_run_cell_id = GetAlreadyRunCellId(cell_id);
    MS_LOG(DEBUG) << "Get check already run top cell id " << check_already_run_cell_id;
    auto find_top_cell = GetTopCell(check_already_run_cell_id, input_args_id);
    if (find_top_cell != nullptr) {
      MS_LOG(DEBUG) << "Find already run top cell " << find_top_cell;
      forward_run = find_top_cell->forward_already_run();
      bool input_args_changed =
        !find_top_cell->input_args_id().empty() && find_top_cell->input_args_id() != input_args_id;
      if (forward_run && input_args_changed) {
        MS_LOG(DEBUG) << "The input info " << input_args_id << " is not the same with pre input info "
                      << find_top_cell->input_args_id() << ", forward process will run again";
        forward_run = false;
      }
      // The pipeline top cell finish forward, but grad is the previous pipeline top cell. Need reset auto meta grad
      // info
      if (top_cell_ != nullptr && top_cell_->is_pipeline_top_cell() && top_cell_->input_args_id() != input_args_id) {
        WaitBpropTask();
        top_cell_->ResetMetaGradInfo();
      }
      if (forward_run) {
        // If neee_increase_grad_order is true means grad order increased and prepare to do grad;
        // But forward run is true now, means no need do forward again, so grad order need be decrease.
        if (neee_increase_grad_order) {
          DecreaseGradOrder();
        }
        finded_top_cell_ = find_top_cell;
      }
    }
  }
  if (!forward_run) {
    grad_first_ = true;
  }
  forward_run ? MS_LOG(DEBUG) << "Top cell have already ran with input args id " << input_args_id
              : MS_LOG(DEBUG) << "Top cell no run before with input args id " << input_args_id;
  return BaseRefToPyData(forward_run);
}

py::object GradExecutor::RunGradFunc(const autograd::GradAttr &grad_attr,
                                     const std::vector<tensor::BaseTensorPtr> &w_args,
                                     const std::vector<size_t> &p_args) {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  ValuePtr sens = nullptr;
  if (grad_attr.has_sens) {
    sens = top_input_args_info_->input_arg_value_vec.back();
  }

  MS_LOG(DEBUG) << "Eval run begin";
  MS_EXCEPTION_IF_NULL(top_cell_);
  auto auto_grad_cell = std::dynamic_pointer_cast<autograd::FuncGrad>(top_cell_->auto_grad_cell_ptr());
  MS_EXCEPTION_IF_NULL(auto_grad_cell);
  top_cell_->set_grad_is_running(true);
  auto grads = auto_grad_cell->Finish(w_args, p_args, grad_attr, sens);
  MS_EXCEPTION_IF_NULL(grads);
  MS_EXCEPTION_IF_NULL(top_cell_);
  top_cell_->set_grad_is_running(false);
  top_input_args_info_ = top_cell_->input_args_info();
  MS_LOG(DEBUG) << "Eval run end";

  // Clear top cell resource
  top_cell_->ClearMetaGradInfo();
  // Set auto_grad_cell nullptr to make sure that auto grad cell can async clear.
  auto_grad_cell = nullptr;
  // Func grad need to use auto grad meta in finish, so clear it after finish.
  AsyncClearAutoGradCell(top_cell_);
  ClearGradRes();

  // For custom nested grad, we need to resume grad info when finish custom grad.
  if (top_cell_ != nullptr) {
    top_cell_->ResumeMetaGradInfo();
  }
  return BaseRefToPyData(grads);
}

py::object GradExecutor::RunGradGraph() {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  MS_EXCEPTION_IF_NULL(top_cell_);
  const auto &resource = top_cell_->resource();
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Run top cell " << top_cell_ << " and its shadow top cell " << top_cell_->shadow_top_cell();
  VectorRef arg_list;
  SetGraphInputArgs(top_input_args_info_->input_arg_value_vec, resource, top_cell_->initial_graph_param_size(),
                    top_input_args_info_->sens_type, &arg_list);
  MS_LOG(DEBUG) << "Convert args size " << top_input_args_info_->input_arg_value_vec.size() << ", graph param size "
                << arg_list.size();

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetJitLevel(kAttrJitLevelO0);

  compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
  MS_EXCEPTION_IF_NULL(run);

  MS_LOG(DEBUG) << "Eval run " << MsContext::GetInstance()->backend_policy();
  top_cell_->set_grad_is_running(true);
  BaseRef out_value = (*run)(arg_list);
  MS_EXCEPTION_IF_NULL(top_cell_);
  top_cell_->set_grad_is_running(false);
  top_input_args_info_ = top_cell_->input_args_info();
  MS_LOG(DEBUG) << "Eval run end";

  // Do high-order grad
  MakeNestedCnode(top_input_args_info_->has_custom_bprop, top_input_args_info_->input_arg_value_vec,
                  resource->optimize_graph(), out_value);

  // For custom nested grad, we need to resume grad info when finish custom grad.
  if (top_cell_ != nullptr) {
    top_cell_->ResumeMetaGradInfo();
  }
  return BaseRefToPyData(out_value);
}

void GradExecutor::MakeNestedCnode(bool has_custom_bprop, const std::vector<ValuePtr> &forward_args,
                                   const FuncGraphPtr &cur_run_bprop_graph, const BaseRef &out) {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  if (top_input_args_info_->is_grad_topest_cell) {
    MS_LOG(DEBUG) << "No nested grad find";
    MS_EXCEPTION_IF_NULL(top_cell_);
    top_cell_->ClearMetaGradInfo();
    ClearGradRes();
    return;
  }
  MS_LOG(DEBUG) << "Do high grad";
  // first_grad_fg maybe modified in auto grad, and first_grad_fg can be used multiple times
  auto first_grad_fg = cur_run_bprop_graph;
  MS_LOG(DEBUG) << "Current top cell ptr " << top_cell().get() << " and its shadow top cell "
                << top_cell_->shadow_top_cell();
  top_cell_->set_is_finish_backward(true);
  if (has_custom_bprop) {
    first_grad_fg = curr_g();
    // Bprop top cell just used for getting forward graph
    top_cell_ = PopTopCellStack();
    MS_LOG(DEBUG) << "Bprop nested, after get bprop forward graph, current top cell ptr " << top_cell().get();
  } else {
    RestoreBpropGraphParameter(cur_run_bprop_graph, top_cell()->initial_graph_param_size());
  }

  MS_EXCEPTION_IF_NULL(first_grad_fg);
  PyNativeAlgo::Common::DumpGraphIR("first_grad_fg.ir", first_grad_fg);
  ValuePtrList weights_args;
  const std::string cur_top_cell_id = top_cell()->obj_id_with_grad_order();
  bool use_dynamic_shape_process = top_cell()->use_dynamic_shape_process() || top_cell()->vm_compile();
  bool has_call_graph = top_cell()->has_call_graph();
  auto inner_graph_info = top_cell()->graph_info_map().at(curr_g());
  SwitchTopCell();
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->op_grad_info->input_value = forward_args;
  op_run_info->input_size = forward_args.size();
  auto out_value = PyNativeAlgo::DataConvert::BaseRefToValue(out, true, true, top_cell_->op_index());
  // Get output values
  if (has_custom_bprop && !out_value->isa<ValueSequence>()) {
    std::vector<ValuePtr> out_v{out_value};
    out_value = std::make_shared<ValueTuple>(out_v);
  }
  MS_EXCEPTION_IF_NULL(out_value);
  RecordNestedGraph(first_grad_fg, inner_graph_info, forward_args, out_value);

  // Get input values
  PyNativeAlgo::Common::SetGraphInputAndWeightsInfo(op_run_info, first_grad_fg, top_cell());
  auto grad_fg = first_grad_fg;
  if (has_call_graph) {
    auto r = std::make_shared<pipeline::Resource>();
    jit()->set_eliminate_forward(false);
    (void)first_grad_fg->transforms().erase(kGrad);
    auto opt = opt::Optimizer::MakeEmptyOptimizer(r);
    opt->set_is_first_order_j(false);
    grad_fg = ad::Grad(first_grad_fg, opt);
    jit()->set_eliminate_forward(true);
  }
  op_run_info->op_grad_info->out_value = out_value;
  op_run_info->op_grad_info->output_size = PyNativeAlgo::Common::GetValueSize(op_run_info->op_grad_info->out_value);
  op_run_info->op_grad_info->out_abs = first_grad_fg->output()->abstract();
  auto grad_param = std::make_shared<GradParam>(op_run_info->op_grad_info, use_dynamic_shape_process);
  grad_param->fg = grad_fg;
  grad_param->source_fg = first_grad_fg;
  grad_param->is_control_flow = has_call_graph;
  // If fun grad and ir grad use the same ad grad graph(hit cache), dout will occur wrong by different type(tuple or
  // plant tuple)
  grad_param->graph_cache_key = cur_top_cell_id + std::to_string(top_cell()->is_ir_grad());
  if (!top_cell()->auto_grad_cell_ptr()->KPynativeWithFProp(grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to run ad grad for second grad graph ";
  }
  top_cell()->set_need_do_final_opt(true);
}

void GradExecutor::DoParameterReplace(const FuncGraphPtr &first_grad_fg, const GraphInfoPtr &inner_graph_info,
                                      const std::vector<ValuePtr> &forward_args, AnfNodePtrList *inputs) {
  MS_EXCEPTION_IF_NULL(inner_graph_info);
  auto outer_graph_info = top_cell()->graph_info_map().at(curr_g());
  MS_EXCEPTION_IF_NULL(outer_graph_info);
  for (const auto &forward_arg : forward_args) {
    const auto &id = PyNativeAlgo::Common::GetIdByValue(forward_arg);
    const auto it = outer_graph_info->input_params.find(id);
    if (it != outer_graph_info->input_params.end()) {
      // Can find in outer graph
      MS_LOG(DEBUG) << "Replace input param id " << id;
      // Replace inner graph param by outer graph param
      (void)inputs->emplace_back(it->second);
    } else {
      MS_LOG(DEBUG) << "Can't find input param id " << id;
      // Inner graph input param not find in outer graph, need add to outer graph
      (void)inputs->emplace_back(GetInput(forward_arg, id));
    }
  }
  mindspore::HashSet<std::string> inner_graph_used_weights_set;
  // Weight in inner graph
  const auto &fir_graph_parameters = first_grad_fg->parameters();
  for (const auto &param : fir_graph_parameters) {
    auto weight_tensor = PyNativeAlgo::Common::GetTensorFromParam(param);
    if (weight_tensor != nullptr) {
      (void)inner_graph_used_weights_set.emplace(weight_tensor->id());
    }
  }
  for (const auto &weight : inner_graph_info->weight_params) {
    // If weight used in graph, but not need get grad by gradnet, it will be a valuenode, no need replace
    if (inner_graph_used_weights_set.find(weight.first) == inner_graph_used_weights_set.end()) {
      continue;
    }
    const auto it = outer_graph_info->weight_params.find(weight.first);
    if (it != outer_graph_info->weight_params.end()) {
      // Can find in outer graph
      MS_LOG(DEBUG) << "Replace weight param name " << weight.second->name() << ", id " << weight.first;
      (void)inputs->emplace_back(it->second);
    } else {
      MS_LOG(DEBUG) << "Can't find weight param name " << weight.second->name() << ", id " << weight.first;
      top_cell()->SetParamNodeMapInGraphInfoMap(weight.first, weight.second, true);
      (void)inputs->emplace_back(weight.second);
    }
  }
}

void GradExecutor::SwitchTopCell() {
  ClearPipelineTopCellRes();
  // Get outer top cell
  auto outer_top_cell = PopTopCellStack();
  MS_EXCEPTION_IF_NULL(outer_top_cell);
  MS_LOG(DEBUG) << "Get outer top cell ptr " << outer_top_cell.get();
  // If inner graph compile graph, outer must be compile
  if (top_cell()->vm_compile()) {
    outer_top_cell->set_force_top_cell_compile(true);
    outer_top_cell->set_use_dynamic_shape_process(outer_top_cell->use_dynamic_shape_process() ||
                                                  top_cell()->use_dynamic_shape_process());
  }
  outer_top_cell->ResumeMetaGradInfo();
  set_top_cell(outer_top_cell);
}

void GradExecutor::ClearGlobalRes() const {
  abstract::AnalysisContext::ClearContext();
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  trace::ClearTraceStack();
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
}

void GradExecutor::ClearGradRes() {
  MS_LOG(DEBUG) << "Top cell run finish " << top_cell_ << " and its shadow top cell " << top_cell_->shadow_top_cell();
  // Pop current top cell on stack
  if (!top_cell_->is_pipeline_top_cell()) {
    (void)PopTopCellStack();
  }

  if (!top_cell_stack_.empty() && top_cell_->is_pipeline_top_cell()) {
    MS_LOG(DEBUG) << "Top cell stack real running top cell " << top_cell_stack_.top();
    if (top_cell_stack_.top() == top_cell_) {
      MS_LOG(DEBUG) << "Pop pipeline top cell " << top_cell_stack_.top() << " from stack with input args id "
                    << top_cell_stack_.top()->input_args_id();
      (void)PopTopCellStack();
    }
  }
  auto has_higher_order = std::any_of(already_run_top_cell_.begin(), already_run_top_cell_.end(),
                                      [](const auto &elem) { return elem.second->is_high_order_top_cell(); });
  // High order must not clean
  if (!has_higher_order) {
    top_cell_->ClearDeviceMemory();
  }

  top_cell_->input_args_info()->Reset();
  top_cell_->set_is_finish_backward(true);
  ClearPipelineTopCellRes();
  top_input_args_info_ = nullptr;
  ClearGlobalRes();
  MS_LOG(DEBUG) << "Current top cell stack size " << top_cell_stack_.size() << ", pipeline top cell map size "
                << pipeline_top_cell_map_.size() << ", pipeline top cell map with already run cell id "
                << top_cell_->already_run_cell_id() << " size "
                << (pipeline_top_cell_map_.find(top_cell_->already_run_cell_id()) == pipeline_top_cell_map_.end()
                      ? 0
                      : pipeline_top_cell_map_[top_cell_->already_run_cell_id()].size());
  top_cell_ = nullptr;
  // Nested grad, get outer top cell if exist
  // Run top cell with bprop, and bprop has grad, after running inner grad, top cell should be restore
  if (!top_cell_stack_.empty()) {
    top_cell_ = top_cell_stack_.top();
    MS_LOG(DEBUG) << "Get outer top cell " << top_cell_ << " as the currently running top cell";
  }
}

void GradExecutor::ClearPipelineTopCellRes() {
  // Remove pipipe top cell from pipeline top cell map exclude the first one
  if (top_cell_->is_pipeline_top_cell()) {
    // Run second step and following step
    if (top_cell_->shadow_top_cell() != nullptr) {
      ErasePipelineTopCell(top_cell_->already_run_cell_id(), top_cell_->shadow_top_cell()->input_args_id(), false);
      top_cell_->set_shadow_top_cell(nullptr);
    } else if (!top_cell_->is_ir_grad()) {
      // Pipeline top cell exclude the first top cell
      ErasePipelineTopCell(top_cell_->already_run_cell_id(), top_cell_->input_args_id(), false);
    }
  } else {
    // If top cell is not pipeline, because it is stored in pipeline top cell map in first step, here need to do delete
    // from the map.
    ErasePipelineTopCell(top_cell_->already_run_cell_id(), top_cell_->input_args_id(), true);
  }
  if (top_cell_->grad_first()) {
    DecreaseGradOrder();
  }
  grad_operation_.clear();
}

void GradExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear grad res";
  WaitBpropTask();
  init_ = false;
  grad_flag_ = false;
  enable_grad_ = true;
  is_run_recompute_ = false;
  save_graphs_ = false;
  forward_use_dynamic_shape_process_ = false;

  kernel_graph_id_for_control_flow_ = UINT32_MAX;
  custom_bprop_cell_count_ = 0;
  grad_order_ = 0;
  op_num_in_bprop_graph_ = kDefaultContainerSize;
  grad_operation_.clear();

  top_cell_ = nullptr;
  top_input_args_info_ = nullptr;
  std::stack<InputArgsInfoPtr>().swap(input_args_info_stack_);
  std::stack<TopCellInfoPtr>().swap(top_cell_stack_);
  already_run_top_cell_.clear();
  pipeline_top_cell_map_.clear();
  dynamic_inputs_cells_.clear();
  need_gc_top_cell_list_.clear();
  dynamic_shape()->Clear();
  jit()->Clear();
}

void GradExecutor::AsyncClearTopCell() {
  for (const auto &need_gc_top_cell : need_gc_top_cell_list_) {
    if (forward()->enable_async()) {
      auto task = [need_gc_top_cell]() { need_gc_top_cell->Clear(); };
      DispatchGradQueueTask(std::move(task));
    } else {
      need_gc_top_cell->Clear();
    }
  }
  need_gc_top_cell_list_.clear();
}

void GradExecutor::AsyncClearAutoGradCell(const TopCellInfoPtr &top_cell) {
  if (forward()->enable_async()) {
    auto task = [top_cell] { top_cell->set_auto_grad_cell_ptr(nullptr); };
    DispatchGradQueueTask(std::move(task));
  } else {
    top_cell->set_auto_grad_cell_ptr(nullptr);
  }
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
  if (v->isa<tensor::BaseTensor>() && v->cast<tensor::BaseTensorPtr>()->is_parameter()) {
    const auto item_by_id = graph_info->weight_params.find(id);
    if (item_by_id != graph_info->weight_params.end()) {
      MS_LOG(DEBUG) << "Get weight param " << id;
      return item_by_id->second;
    }
    MS_LOG(DEBUG) << "Add new weight param " << id;
    auto tensor = v->cast<tensor::BaseTensorPtr>();
    const auto &param_info = tensor->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    const auto &param_name = param_info->name();
    // Add new weight param to graph info
    auto weight_param = curr_g()->add_parameter();
    weight_param->set_name(param_name);
    weight_param->debug_info()->set_name(param_name);
    weight_param->set_default_param(tensor);
    weight_param->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(tensor->ToAbstract()));
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
    (void)abs_list.emplace_back(PyNativeAlgo::Common::SetAbstractValueToAnyValue(v_arg->ToAbstract()));
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
  for (const auto &[cell_id, top_cell] : already_run_top_cell_) {
    MS_EXCEPTION_IF_NULL(top_cell);
    MS_LOG(DEBUG) << "Top cell " << top_cell << " with already run cell id " << cell_id << ", input args id "
                  << top_cell->input_args_id();
    // Complete match, means run grad operation first
    if (top_cell->already_run_cell_id() == already_run_cell_id) {
      find_top_cell = top_cell;
      break;
    }
    // Partial match, means run forward first without grad_operation in already run cell id
    if (already_run_cell_id.find(top_cell->already_run_cell_id()) != std::string::npos &&
        top_cell->already_run_cell_id().back() == '_') {
      find_top_cell = top_cell;
      break;
    }
    // Partial match, means run grad first, but follow a other net grad
    if (top_cell->already_run_cell_id().find(already_run_cell_id) != std::string::npos &&
        already_run_cell_id.back() == '_') {
      find_top_cell = top_cell;
      break;
    }
  }

  // Get pipeline top cell
  if (find_top_cell == nullptr) {
    MS_LOG(DEBUG) << "Not find in already run top cell map, try find in pipeline top cell map";
    find_top_cell = GetPipelineTopCell(already_run_cell_id, input_args_id, already_run_cell_id.back() == '_');
  } else if (find_top_cell->is_pipeline_top_cell()) {
    // Delete first pipeline top from already run top cell map
    (void)already_run_top_cell_.erase(find_top_cell->already_run_cell_id());
    if (find_top_cell->input_args_id() != input_args_id) {
      MS_LOG(DEBUG) << "Find top cell input args id " << find_top_cell->input_args_id()
                    << " not match current input args id " << input_args_id << ", try find in pipeline top cell map";
      find_top_cell = GetPipelineTopCell(already_run_cell_id, input_args_id, already_run_cell_id.back() == '_');
    }
  }

  // Same topcell info, but grad operation is not the same, construct backward graph again
  if (find_top_cell != nullptr) {
    if (!find_top_cell->grad_operation().empty() && find_top_cell->grad_operation() != grad_operation_) {
      MS_LOG(DEBUG) << "Already exist grad operation " << find_top_cell->grad_operation() << " is different with new "
                    << grad_operation_;
      (void)already_run_top_cell_.erase(find_top_cell->already_run_cell_id());
      return nullptr;
    }
    return find_top_cell;
  }
  return nullptr;
}

void GradExecutor::ProcessOpGradInfo(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  RecordForwardGraph(op_run_info);
  if (top_cell_->is_bprop_need_get_forward_graph()) {
    MS_LOG(DEBUG) << "Just need forward graph";
    return;
  }
  DoOpGrad(op_run_info);
}

void GradExecutor::SaveOutputNodeMap(const std::string &obj_id, const FrontendOpRunInfoPtr &op_run_info,
                                     const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Cnode is " << cnode->DebugString() << ", out value id " << obj_id;
  // In hook compute, output is a copy of input; If hook input is a input param, follow op use hook output as input,
  // which GetInput will always find input param, so need delete from input param map
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (op_run_info->run_in_vm && kHookOp.find(op_run_info->base_op_run_info.op_name) != kHookOp.end()) {
    for (size_t i = 0; i < op_run_info->input_size; ++i) {
      top_cell()->DeleteParamNodeInfo(curr_g(), op_run_info->input_value_id[i]);
    }
  }
  top_cell()->SetNodeMapInGraphInfoMap(obj_id, cnode);
}

// Run ad grad for curr op and connect grad graph with previous op
void GradExecutor::DoOpGrad(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  top_cell()->GetOpInfo(op_run_info, false);
  auto pre_top_cell = PyNativeAlgo::AutoGrad::FindPreTopcell(this, op_run_info->op_grad_info,
                                                             op_run_info->op_grad_info->op_info, op_run_info->real_out);
  auto &&grad_param = CreateOpGradParam(op_run_info, top_cell());
  if (forward()->enable_async()) {
    auto auto_grad_cell_ptr = top_cell()->auto_grad_cell_ptr();
    auto task = [auto_grad_cell_ptr, grad_param, pre_top_cell, this]() {
      PyNativeAlgo::AutoGrad::UpdateGradOpInfo(this, grad_param->op_grad_info, pre_top_cell, false);
      (void)auto_grad_cell_ptr->KPynativeOp(grad_param);
    };
    DispatchGradQueueTask(std::move(task));
  } else {
    PyNativeAlgo::AutoGrad::UpdateGradOpInfo(this, grad_param->op_grad_info, pre_top_cell, false);
    (void)top_cell()->auto_grad_cell_ptr()->KPynativeOp(grad_param);
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
    }
    if (IsPrimitiveCNode(input_node, prim::kPrimTupleGetItem)) {
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
  AnfNodePtrList inputs;
  (void)inputs.emplace_back(NewValueNode(op_run_info->op_grad_info->op_prim));
  for (size_t i = 0; i < op_run_info->input_size; i++) {
    AnfNodePtr input_node = nullptr;
    const auto node = GetInput(op_run_info->op_grad_info->input_value[i], op_run_info->input_value_id[i]);
    input_node = GetRealInputNodeBySkipHook(node);
    // update abstract
    if (input_node != nullptr) {
      (void)inputs.emplace_back(input_node);
    }
  }
  const auto &cnode = curr_g()->NewCNodeInOrder(inputs);
  if (IsPrimitiveCNode(cnode, prim::kPrimCellBackwardHook)) {
    top_cell()->RecordCellBackwardHookOp(hook_cell_id_, cnode);
  }
  MS_LOG(DEBUG) << "Make CNode for " << op_run_info->base_op_run_info.op_name << ", new cnode is "
                << cnode->DebugString();
  return cnode;
}

void GradExecutor::RecordForwardGraph(const FrontendOpRunInfoPtr &op_run_info) const {
  if (save_graphs_ || top_cell_->is_bprop_need_get_forward_graph()) {
    MS_EXCEPTION_IF_NULL(op_run_info);
    if (op_run_info->input_value_id.empty()) {
      (void)std::transform(op_run_info->op_grad_info->input_value.begin(), op_run_info->op_grad_info->input_value.end(),
                           std::back_inserter(op_run_info->input_value_id),
                           [](const ValuePtr &value) { return PyNativeAlgo::Common::GetIdByValue(value); });
    }
    if (op_run_info->out_value_id.empty()) {
      op_run_info->out_value_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
    }
    const auto &cnode = ConstructForwardGraph(op_run_info);
    MS_EXCEPTION_IF_NULL(cnode);
    // By simple infer, abstract is nullptr
    if (op_run_info->base_op_run_info.abstract == nullptr) {
      cnode->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(op_run_info->real_out->ToAbstract()));
    } else {
      cnode->set_abstract(op_run_info->base_op_run_info.abstract);
    }
    SaveOutputNodeMap(op_run_info->out_value_id, op_run_info, cnode);
  }
}

void GradExecutor::RecordForwardGraphForInput(const ValuePtr &value, const string &input_id,
                                              const abstract::AbstractBasePtr &param_abs) {
  save_graphs_ = MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs_ || top_cell_->is_bprop_need_get_forward_graph()) {
    auto new_param = curr_g()->add_parameter();
    new_param->set_abstract(param_abs);
    if (value->isa<ValueSequence>()) {
      top_cell()->SetNodeMapInGraphInfoMap(input_id, new_param, true);
    }
    top_cell()->SetParamNodeMapInGraphInfoMap(input_id, new_param);
  }
}

void GradExecutor::RecordNestedGraph(const FuncGraphPtr &first_grad_fg, const GraphInfoPtr &inner_graph_info,
                                     const std::vector<ValuePtr> &forward_args, const ValuePtr &out) {
  if (save_graphs_) {
    AnfNodePtrList inputs{NewValueNode(first_grad_fg)};
    DoParameterReplace(first_grad_fg, inner_graph_info, forward_args, &inputs);
    auto cnode = curr_g()->NewCNode(inputs);
    auto out_id = PyNativeAlgo::Common::GetIdByValue(out);
    top_cell()->SetNodeMapInGraphInfoMap(out_id, cnode);
    cnode->set_abstract(first_grad_fg->output()->abstract());
    MS_LOG(DEBUG) << "Nested make cnode is: " << cnode->DebugString() << ", out id " << out_id;
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
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
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

void GradExecutor::SetTopCellDynamicAttr(const py::object &cell) {
  if (top_cell_ == nullptr) {
    return;
  }

  if (top_cell()->use_dynamic_shape_process()) {
    // Top cell is already dynamic, no need to set again.
    return;
  }
  top_cell()->set_use_dynamic_shape_process(dynamic_inputs_cells_.count(PyNativeAlgo::PyParser::GetIdByPyObj(cell)));
}

void GradExecutor::DispatchGradQueueTask(std::function<void(void)> &&task) const {
  const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
  if (!bprop_queue->Push(new (std::nothrow) BpropTask(std::move(task)))) {
    bprop_queue->CheckException();
  }
}

void GradExecutor::ClearBpropTask() const {
  const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
  if (bprop_queue != nullptr) {
    GilReleaseWithCheck gil_release;
    bprop_queue->Clear();
    bprop_queue->CheckException();
  }
}

void GradExecutor::WaitBpropTask() const {
  const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
  if (bprop_queue != nullptr) {
    GilReleaseWithCheck gil_release;
    bprop_queue->Wait();
    bprop_queue->CheckException();
  }
}

void GradExecutor::ChildAfterFork() {
  MS_LOG(DEBUG) << "GradExecutor reinitialize after fork.";
  const auto &bprop_queue = runtime::Pipeline::Get().bprop_stage();
  if (bprop_queue != nullptr) {
    MS_LOG(DEBUG) << "Reinitialize bprop_queue_.";
    bprop_queue->ChildAfterFork();
  }
  runtime::PyBoostOpExecute::GetInstance().ClearBackend();
  MS_LOG(DEBUG) << "GradExecutor reinitialize after fork done.";
}
}  // namespace pynative
}  // namespace mindspore
