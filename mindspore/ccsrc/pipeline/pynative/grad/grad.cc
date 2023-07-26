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
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/pynative_cache.h"
#include "pipeline/jit/ps/pipeline.h"
#include "ir/cell.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/convert_utils_py.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/environ_conversion.h"
#include "frontend/expander/utils.h"
#include "pipeline/jit/ps/pass.h"
#include "frontend/expander/bprop/bprop.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace pynative {
namespace {
const mindspore::HashSet<std::string> kHookOp = {"HookBackward", "CellBackwardHook"};
const char kGrad[] = "grad";
const size_t kMaxCacheDynamicShapeCellNum = 2;
const size_t kContainerRatio = 2;

std::string GetCellId(const py::object &obj, const py::args &args, const InputArgsInfoPtr &input_args_info) {
  auto cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
  auto fn = [&cell_id](const abstract::AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->BuildShape();
    auto type = abs->BuildType();
    cell_id += "_" + shape->ToString();
    cell_id += type->ToString();
  };

  const auto &forward = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor();
  bool id_not_exist = (input_args_info == nullptr);
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &arg_id =
      id_not_exist ? PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) : input_args_info->input_arg_id_vec[i];
    // Find in step process
    auto cache_abs = forward->GetNodeAbsById(arg_id);
    if (cache_abs != nullptr) {
      fn(cache_abs);
    } else {
      auto abs = PyNativeAlgo::DataConvert::PyObjToValue(args[i])->ToAbstract();
      fn(abs);
    }
  }
  return cell_id;
}

void ParsePyArgsToInputArgsInfo(const InputArgsInfoPtr &input_args_info, const py::object &obj, const py::args &args) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  input_args_info->has_custom_bprop = py::hasattr(obj, parse::CUSTOM_BPROP_NAME);
  bool is_top_cell = input_args_info->is_grad_topest_cell || input_args_info->is_high_order_top_cell;
  if (is_top_cell || input_args_info->grad_is_running) {
    pipeline::CheckArgsValid(obj, args);
  }
  // Only the top cell or custom bprop cell requires value conversion
  if (is_top_cell || input_args_info->grad_is_running || input_args_info->has_custom_bprop) {
    input_args_info->obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);
    input_args_info->input_size = args.size();
    for (size_t i = 0; i < input_args_info->input_size; ++i) {
      const auto &id = PyNativeAlgo::PyParser::GetIdByPyObj(args[i]);
      (void)input_args_info->input_arg_id_vec.emplace_back(id);
    }
    for (size_t i = 0; i < input_args_info->input_size; ++i) {
      input_args_info->input_args_id += input_args_info->input_arg_id_vec[i] + "_";
      if (py::isinstance<py::list>(args[i])) {
        (void)input_args_info->input_arg_value_vec.emplace_back(
          PyNativeAlgo::DataConvert::PyObjToValue(py::cast<py::tuple>(args[i])));
      } else {
        (void)input_args_info->input_arg_value_vec.emplace_back(PyNativeAlgo::DataConvert::PyObjToValue(args[i]));
      }
    }
    input_args_info->cell_id = GetCellId(obj, args, input_args_info);
    input_args_info->obj_order_id = input_args_info->cell_id + '_' + std::to_string(input_args_info->obj_order);
    MS_LOG(DEBUG) << "Cell_id is " << input_args_info->cell_id << ", is grad top cell " << is_top_cell;
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

ValuePtr ConvertOutputValueToTensor(const ValuePtr &v) {
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
      return v;
    }
    MS_LOG(DEBUG) << "Output is value sequence, but have tensor and other type mixed. Its value is " << v->ToString();
    return PyNativeAlgo::Common::FilterSensValues(v);
  } else if (v->isa<FloatImm>()) {
    double input_value = v->cast<FP32ImmPtr>()->value();
    return std::make_shared<tensor::Tensor>(input_value, kFloat32);
  } else if (v->isa<BoolImm>()) {
    return std::make_shared<tensor::Tensor>(v->cast<BoolImmPtr>()->value(), kBool);
  } else if (v->isa<IntegerImm>()) {
    int64_t input = v->cast<Int64ImmPtr>()->value();
    return std::make_shared<tensor::Tensor>(input, kInt64);
  } else {
    MS_LOG(DEBUG) << "Output is " << v->ToString() << ", abstract "
                  << PyNativeAlgo::Common::SetAbstractValueToAnyValue(v->ToAbstract());
    return v;
  }
}

bool IsValuePtrEqual(const ValuePtr &v1, const ValuePtr &v2) {
  if (v1 == v2) {
    return true;
  }
  if (v1 == nullptr || v2 == nullptr) {
    return false;
  }
  if (v1->isa<tensor::Tensor>() && v2->isa<tensor::Tensor>()) {
    return v1->cast<tensor::TensorPtr>()->ValueEqual(*(v2->cast<tensor::TensorPtr>()));
  }
  return *v1 == *v2;
}

bool IsDynamicDetectAbsChange(const AbstractBasePtr &old_abs, const AbstractBasePtr &new_abs) {
  if (old_abs == new_abs) {
    return false;
  }
  if (old_abs == nullptr || new_abs == nullptr) {
    MS_LOG(DEBUG) << "Graph is dynamic, old_abs is different with new_abs";
    return true;
  }
  if (!common::IsEqual(old_abs->BuildType(), new_abs->BuildType()) ||
      !common::IsEqual(old_abs->BuildShape(), new_abs->BuildShape())) {
    MS_LOG(DEBUG) << "Graph is dynamic, old_abs is different with new_abs, old abs: " << old_abs->ToString()
                  << " new abs: " << new_abs->ToString();
    return true;
  }
  return false;
}

bool IsDynamicDetectAbsChange(const abstract::AbstractBasePtrList &node_abs,
                              const abstract::AbstractBasePtrList &old_node_abs) {
  if (node_abs.size() != old_node_abs.size()) {
    MS_LOG(DEBUG) << "Graph is dynamic, node_abs size: " << node_abs.size()
                  << " old_node_abs size: " << old_node_abs.size();
    return true;
  }
  for (size_t i = 0; i < node_abs.size(); ++i) {
    if (IsDynamicDetectAbsChange(node_abs[i], old_node_abs[i])) {
      return true;
    }
  }
  return false;
}

bool IsDynamicDetectPrimChange(const PrimitivePtr &old_prim, const PrimitivePtr &new_prim) {
  if (old_prim == nullptr && new_prim == nullptr) {
    return false;
  }
  if (new_prim != nullptr && old_prim != nullptr) {
    return !common::IsEqual(old_prim, new_prim);
  }
  return true;
}

bool IsDynamicDetectNodeInfoChange(const NodeInfo &old_node_info, const NodeInfo &new_node_info) {
  if (new_node_info.grad_type == TensorGradType::kParameter &&
      (old_node_info.grad_type == TensorGradType::kParameter || old_node_info.grad_type == TensorGradType::kConstant)) {
    MS_EXCEPTION_IF_NULL(new_node_info.value);
    MS_EXCEPTION_IF_NULL(old_node_info.value);
    auto new_tensor = new_node_info.value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(new_tensor);
    auto old_tensor = old_node_info.value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(old_tensor);
    if (new_tensor->id() != old_tensor->id()) {
      MS_LOG(DEBUG) << "Graph is dynamic, new node info value: "
                    << (new_node_info.value != nullptr ? new_node_info.value->ToString() : "")
                    << " grad type: " << new_node_info.grad_type << " old node info value: "
                    << (old_node_info.value != nullptr ? old_node_info.value->ToString() : "")
                    << " grad type: " << old_node_info.grad_type;
      return true;
    }
    return false;
  }

  if (new_node_info.grad_type != old_node_info.grad_type) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node info grad type: " << new_node_info.grad_type
                  << " old node info grad type: " << old_node_info.grad_type;
    return true;
  }

  if (new_node_info.grad_type == TensorGradType::kOpOutput && new_node_info.op_index != old_node_info.op_index) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node info op_index: " << new_node_info.op_index
                  << " old node info op_index: " << old_node_info.op_index;
    return true;
  }

  if (new_node_info.grad_type == TensorGradType::kConstant &&
      !IsValuePtrEqual(new_node_info.value, old_node_info.value)) {
    MS_LOG(DEBUG) << "Graph is dynamic, new node info value: "
                  << (new_node_info.value != nullptr ? new_node_info.value->ToString() : "")
                  << " grad type: " << new_node_info.grad_type
                  << " old node info value: " << (old_node_info.value != nullptr ? old_node_info.value->ToString() : "")
                  << " grad type: " << old_node_info.grad_type;
    return true;
  }

  return false;
}

void BuildDynamicDetectNodeInput(const ValuePtr &input, std::vector<std::pair<std::string, NodeInfo>> *node_inputs,
                                 const std::string &value_idx) {
  if (input->isa<tensor::Tensor>()) {
    NodeInfo node_info;
    auto tensor = input->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto auto_meta_data = tensor->auto_grad_meta_data();
    if (auto_meta_data == nullptr) {
      node_info.value = input;
      node_info.grad_type = TensorGradType::kConstant;
      (void)node_inputs->emplace_back(std::make_pair(value_idx, node_info));
      return;
    }
    node_info.grad_type = auto_meta_data->grad_type();
    node_info.op_index = auto_meta_data->op_index();
    if (node_info.grad_type == TensorGradType::kConstant || node_info.grad_type == TensorGradType::kParameter) {
      node_info.value = input;
    }
    (void)node_inputs->emplace_back(std::make_pair(value_idx, node_info));
  } else if (input->isa<ValueSequence>()) {
    auto value_sequence = input->cast<ValueSequencePtr>();
    for (size_t i = 0; i < value_sequence->value().size(); ++i) {
      const string &cur_idx = value_idx + std::to_string(i);
      BuildDynamicDetectNodeInput(value_sequence->value()[i], node_inputs, cur_idx);
    }
  } else if (input->isa<stub::StubNode>()) {
    auto stub_node = input->cast<stub::StubNodePtr>();
    MS_EXCEPTION_IF_NULL(stub_node);
    BuildDynamicDetectNodeInput(stub_node->WaitValue(), node_inputs, value_idx);
  } else {
    NodeInfo node_info;
    node_info.grad_type = TensorGradType::kConstant;
    node_info.value = input;
    (void)node_inputs->emplace_back(std::make_pair(value_idx, node_info));
  }
}

std::vector<std::pair<std::string, NodeInfo>> BuildDynamicDetectNodeInputs(const ValuePtrList &inputs) {
  std::vector<std::pair<std::string, NodeInfo>> node_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const string &tensor_idx = std::to_string(i);
    BuildDynamicDetectNodeInput(inputs[i], &node_inputs, tensor_idx);
  }
  return node_inputs;
}

bool IsDynamicDetectInputChange(const std::vector<std::pair<std::string, NodeInfo>> &old_inputs,
                                const std::vector<std::pair<std::string, NodeInfo>> &new_inputs) {
  if (old_inputs.size() != new_inputs.size()) {
    MS_LOG(DEBUG) << "Graph is dynamic, old_inputs size: " << old_inputs.size()
                  << "new_inputs size: " << new_inputs.size();
    return true;
  }
  for (size_t i = 0; i < old_inputs.size(); ++i) {
    std::string old_tensor_idx = old_inputs[i].first;
    auto old_node_info = old_inputs[i].second;
    std::string new_tensor_idx = new_inputs[i].first;
    auto new_node_info = new_inputs[i].second;
    if (old_tensor_idx != new_tensor_idx) {
      MS_LOG(DEBUG) << "Graph is dynamic, old_tensor_idx: " << old_tensor_idx << "new_tensor_idx: " << new_tensor_idx;
      return true;
    }
    if (IsDynamicDetectNodeInfoChange(old_node_info, new_node_info)) {
      MS_LOG(DEBUG) << "Graph is dynamic, old_node op index is: " << old_node_info.op_index
                    << " value is: " << (old_node_info.value != nullptr ? old_node_info.value->ToString() : "")
                    << " new_node op index is: " << new_node_info.op_index
                    << " value is: " << (new_node_info.value != nullptr ? new_node_info.value->ToString() : "");
      return true;
    }
  }
  return false;
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

void SetGraphInputArgs(const std::vector<ValuePtr> &input_vec, const pipeline::ResourcePtr &res, bool is_tuple_sens,
                       VectorRef *const arg_list) {
  MS_EXCEPTION_IF_NULL(arg_list);
  MS_EXCEPTION_IF_NULL(res);
  auto graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  AnfNodePtrList graph_params = graph->parameters();
  std::size_t graph_params_size = graph_params.size();
  auto input_arg_list = input_vec;
  if (is_tuple_sens) {
    input_arg_list.clear();
    PyNativeAlgo::DataConvert::FlattenArgs(input_vec, &input_arg_list, true);
  }
  (void)std::transform(input_arg_list.begin(), input_arg_list.end(), std::back_inserter(*arg_list),
                       [](const ValuePtr &v) { return v; });
  size_t arg_size = (*arg_list).size();
  if (arg_size != graph_params_size) {
    // Maybe have some default parameter for input
    MS_LOG(DEBUG) << "Get args size " << (*arg_list).size() << ", graph param size " << graph_params_size;
    for (std::size_t i = arg_size; i < graph_params_size; ++i) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      auto param_ptr = (graph_params[i])->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->has_default()) {
        MS_LOG(EXCEPTION) << "Parameter[" << i << "] has no default param, " << param_ptr->DebugString();
      }
      if (!param_ptr->default_param()->isa<tensor::Tensor>()) {
        MS_LOG(EXCEPTION) << "Parameter[" << param_ptr->DebugString()
                          << "] is not initialized, need to call `.init_data()`";
      }
      arg_list->push_back(param_ptr->default_param());
    }
  }
}

void SetSensValue(const prim::GradOperationPtr &grad, const InputArgsInfoPtr &input_args_info, const py::args &args) {
  MS_EXCEPTION_IF_NULL(grad);
  if (!grad->sens_param()) {
    return;
  }
  MS_LOG(DEBUG) << "Get sens param";
  size_t forward_args_size = args.size() - 1;
  auto sens_v = PyNativeAlgo::DataConvert::PyObjToValue(args[forward_args_size]);
  const auto &sens_tensor = ConvertOutputValueToTensor(sens_v);
  if (sens_tensor == nullptr) {
    MS_LOG(EXCEPTION) << "sens convert tensor is nullptr";
  }
  // Sens have already exist, which may be need update
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (input_args_info->input_arg_value_vec.size() == args.size()) {
    input_args_info->input_arg_value_vec.pop_back();
  }
  (void)input_args_info->input_arg_value_vec.emplace_back(sens_tensor);
  input_args_info->has_sens = true;
}

std::string GetWeightsObjIdsByWeights(const py::object &weights) {
  auto is_require_grad = [](const ValuePtr &value) {
    MS_EXCEPTION_IF_NULL(value);
    if (!value->isa<tensor::Tensor>()) {
      return false;
    }
    auto t = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(t);
    if (t->is_parameter() && t->param_info() != nullptr && t->param_info()->requires_grad()) {
      return true;
    }
    return false;
  };

  std::string weights_obj_id;
  auto append_weights_info = [&weights_obj_id, is_require_grad](const py::object &obj) {
    const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(obj);
    if (is_require_grad(v)) {
      weights_obj_id.append("_").append(PyNativeAlgo::Common::GetIdByValue(v));
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
    if (x_is_const_value) {
      op_run_info->op_grad_info->input_value[kIndex1] =
        PyNativeAlgo::Common::CreateFakeTensorWithoutDeviceAddress(op_run_info->base_op_run_info.input_tensor[kIndex1]);
      MS_LOG(DEBUG) << "Clear device address for inputs[1] of " << op_name;
    }
    if (y_is_const_value) {
      op_run_info->op_grad_info->input_value[kIndex0] =
        PyNativeAlgo::Common::CreateFakeTensorWithoutDeviceAddress(op_run_info->base_op_run_info.input_tensor[kIndex0]);
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

autograd::GradParamPtr CreateOpGradParam(const FrontendOpRunInfoPtr &op_run_info, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  bool out_used_in_bporp_graph = true;
  op_run_info->op_grad_info->out_value = op_run_info->real_out;
  // Free bprop not used output
  if (!top_cell->is_high_order_top_cell()) {
    if (op_run_info->input_unused_in_bprop[op_run_info->input_size]) {
      // Process output
      op_run_info->op_grad_info->out_value =
        PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(op_run_info->op_grad_info->out_value);
    }
    // Free special op memory
    FreeSpecialOpValue(op_run_info->op_grad_info->op_prim->name(), op_run_info, &op_run_info->op_grad_info->out_value);
  }

  op_run_info->op_grad_info->out_abs = op_run_info->base_op_run_info.abstract;
  auto grad_param = std::make_shared<autograd::GradParam>(
    op_run_info->op_grad_info, !top_cell->is_high_order_top_cell(), top_cell->use_dynamic_shape_process());
  grad_param->out_used_in_bporp_graph = out_used_in_bporp_graph;
  return grad_param;
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

void GradExecutor::PushInputArgsInfoStack(const InputArgsInfoPtr &input_args_info) {
  input_args_info_stack_.push(input_args_info);
}

void GradExecutor::PopInputArgsInfoStack() {
  if (input_args_info_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack input_args_info_stack_ is empty";
  }
  input_args_info_stack_.pop();
}

bool GradExecutor::IsBpropGraph(const std::string &cell_id) const {
  if (top_cell_ == nullptr) {
    return false;
  }
  return std::any_of(bprop_cell_list_.begin(), bprop_cell_list_.end(),
                     [&cell_id](const std::string &value) { return cell_id.find(value) != std::string::npos; });
}

void GradExecutor::HandleInputArgsForTopCell(const InputArgsInfoPtr &input_args_info, bool is_bprop_top) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (is_bprop_top) {
    // Convert input args to parameters for top cell graph in bprop.
    for (size_t i = 0; i < input_args_info->input_size; ++i) {
      auto new_param = curr_g()->add_parameter();
      MS_EXCEPTION_IF_NULL(input_args_info->input_arg_value_vec[i]);
      new_param->set_abstract(
        PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_args_info->input_arg_value_vec[i]->ToAbstract()));
      top_cell()->SetParamNodeMapInGraphInfoMap(input_args_info->input_arg_id_vec[i], new_param);
      MS_LOG(DEBUG) << "Top bprop graph set input parameter " << input_args_info->input_arg_id_vec;
    }
    return;
  }
  // Convert input args to parameters for top cell graph in construct.
  std::vector<ValuePtr> input_param_values;
  const auto &input_value = input_args_info->input_arg_value_vec;
  if (input_args_info->input_size != 0 && input_value.empty()) {
    MS_LOG(EXCEPTION) << "Input value is empty";
  }
  AbstractBasePtrList abs_list;
  for (size_t i = 0; i < input_args_info->input_size; ++i) {
    const auto &v = input_value[i];
    (void)PyNativeAlgo::Common::SetValueGradInfo(v, top_cell(), TensorGradType::kInput);
    (void)input_param_values.emplace_back(v);
    auto param_i_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(v->ToAbstract());
    (void)abs_list.emplace_back(param_i_abs);
    RecordForwardGraphForInput(v, input_args_info->input_arg_id_vec[i], param_i_abs);
  }
  top_cell()->set_auto_grad_cell_ptr(std::make_shared<autograd::AutoGradCellImpl>(
    input_param_values, abs_list, op_num_in_bprop_graph_ * kContainerRatio));
}

void GradExecutor::InitResourceAndDfBuilder(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  forward()->WaitForwardTask();
  // We need wait construct bprop task of outer top cell finish, if main thread run quickly, when it execute gradnet
  // and clear async_executor queue, bprop task of outer top cell may not finish, it will cause not found cnode
  // error.
  {
    GilReleaseWithCheck gil_release;
    async_executor_->Wait();
  }
  if (input_args_info->is_grad_topest_cell && !input_args_info->grad_is_running) {
    MS_LOG(DEBUG) << "Make new topest graph";
    MakeNewTopGraph(input_args_info);
  } else if (input_args_info->grad_is_running && IsBpropGraph(input_args_info->cell_id)) {
    MS_LOG(DEBUG) << "Run custom bprop cell";
    auto fg = std::make_shared<FuncGraph>();
    top_cell()->set_fg(fg);
    auto graph_info_cg = std::make_shared<PyNGraphInfo>();
    top_cell()->SetGraphInfoMap(fg, graph_info_cg);
    HandleInputArgsForTopCell(input_args_info, true);
    bprop_grad_stack_.push(std::make_pair(input_args_info->cell_id, false));
  } else if (input_args_info->grad_is_running && top_cell()->grad_order() != input_args_info->grad_order) {
    MS_LOG(DEBUG) << "Nested grad graph existed in custom bprop";
    MakeNewTopGraph(input_args_info);
    bprop_grad_stack_.push(std::make_pair(input_args_info->cell_id, true));
  } else if (input_args_info->is_high_order_top_cell) {
    MS_LOG(DEBUG) << "Nested grad graph existed in construct";
    SaveInputTensorGradInfo(input_args_info);
    MakeNewTopGraph(input_args_info);
  }

  // Init kPynativeCellPtr with input parameters of top cell
  if (!top_cell()->is_init_kpynative()) {
    forward()->WaitForwardTask();
    auto graph_info_cg = std::make_shared<PyNGraphInfo>();
    top_cell()->SetGraphInfoMap(curr_g(), graph_info_cg);
    HandleInputArgsForTopCell(input_args_info, false);
    top_cell()->set_need_compile_graph(true);
    top_cell()->set_init_kpynative(true);
  }
}

void GradExecutor::NewGraphInner(const py::object &obj, const py::args &args) {
  const auto input_args_info = GetInputArgsInfo(obj, args);
  PushInputArgsInfoStack(input_args_info);
  MS_LOG(DEBUG) << "NewGraphInner start " << args.size() << ", cell_id " << GetCellId(obj, args, nullptr)
                << ", input args info ptr " << input_args_info.get();
  // Make top graph and init resource
  if (input_args_info->is_grad_topest_cell || input_args_info->grad_order > 1) {
    InitResourceAndDfBuilder(input_args_info);
  }
}

InputArgsInfoPtr GradExecutor::GetInputArgsInfo(const py::object &obj, const py::args &args) {
  const auto &input_args_info = std::make_shared<InputArgsInfo>(
    input_args_info_stack_.empty(), is_high_order_top_cell(), grad_is_running_, obj_order_++);
  ParsePyArgsToInputArgsInfo(input_args_info, obj, args);

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
    // top_input_args_info_ indicate current running cell info
    top_input_args_info_ = input_args_info;
  }
  input_args_info->grad_order = grad_order_;
  return input_args_info;
}

bool GradExecutor::GetTopCellDynamicFlag(const InputArgsInfoPtr &input_args_info,
                                         const std::string &obj_id_with_grad_order) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (forward_use_dynamic_shape_process_) {
    return true;
  }

  if (dynamic_inputs_cells_.count(input_args_info->obj_id) > 0) {
    return true;
  }

  auto pre_top_cell = GetAlreadyRunTopCell(input_args_info->already_run_cell_id);
  if (pre_top_cell != nullptr) {
    return pre_top_cell->use_dynamic_shape_process();
  }

  return std::any_of(already_run_top_cell_.begin(), already_run_top_cell_.end(),
                     [obj_id_with_grad_order](const auto &item) {
                       if (item.second != nullptr && item.second->obj_id_with_grad_order() == obj_id_with_grad_order) {
                         return item.second->use_dynamic_shape_process();
                       }
                       return false;
                     });
}

bool GradExecutor::IsNeedSaveDynamicDetectNodes(const TopCellInfoPtr &top_cell, bool use_dynamic_shape_process) {
  if (use_dynamic_shape_process) {
    // top cell is already dynamic shape, no need save nodes.
    return false;
  }
  MS_EXCEPTION_IF_NULL(top_cell);
  auto cell_iter = cell_id_with_dynamic_detect_nodes_.find(top_cell->obj_id_with_grad_order());
  if (cell_iter == cell_id_with_dynamic_detect_nodes_.end()) {
    // Cell is not found in cell_id_with_dynamic_detect_nodes_, need save nodes first.
    return true;
  }

  const auto &cell_infos = cell_iter->second;
  if (cell_infos.size() == 1) {
    // top_cell->cell_id() is cell id with inputs shape, if cell id in cell_id_with_dynamic_detect_nodes_
    // id same with top_cell->cell_id(), no need save nodes.
    return cell_infos.begin()->first != top_cell->cell_id();
  } else if (cell_infos.size() == kMaxCacheDynamicShapeCellNum) {
    auto cell_infos_iter = cell_infos.find(top_cell->cell_id());
    if (cell_infos_iter == cell_infos.end()) {
      // cell_id_with_dynamic_detect_nodes_ has two cell id already, current cell is is different
      // with them. So set_use_dynamic_shape_process for top cell.
      top_cell->set_use_dynamic_shape_process(true);
      (void)cell_id_with_dynamic_detect_nodes_.erase(top_cell->obj_id_with_grad_order());
      MS_LOG(INFO) << "Set use_dynamic_shape_process: " << use_dynamic_shape_process << ", already cached "
                   << cell_infos.size() << " top cell, cur top cell shape is different:" << top_cell->cell_id();
    }
  } else {
    MS_LOG(EXCEPTION) << "cell_info.size():" << cell_infos.size() << " is invalid";
  }
  return false;
}

void GradExecutor::MakeNewTopGraph(const InputArgsInfoPtr &input_args_info) {
  auto fg = std::make_shared<FuncGraph>();
  fg->debug_info()->set_name("pynative_forward_graph");
  auto resource = std::make_shared<pipeline::Resource>();
  MS_EXCEPTION_IF_NULL(input_args_info);
  const auto &obj_id_with_grad_order = GetAlreadyRunCellId(input_args_info->obj_id);
  // To fix scene that user calls twice forward network with grad flag, and then call grad() interface.
  // We need to clear last top cell's parameters grad info to avoid influencing construct bprop graph of current top
  // cell.
  ClearParamGradInfo(top_cell_);
  top_cell_ = std::make_shared<TopCellInfo>(input_args_info->is_high_order_top_cell, input_args_info->grad_order,
                                            obj_id_with_grad_order, input_args_info->cell_id,
                                            input_args_info->already_run_cell_id, resource, fg,
                                            op_num_in_bprop_graph_ * kContainerRatio);
  top_cell_->set_forward_already_run(true);
  top_cell_->set_input_args_id(input_args_info->input_args_id);
  auto use_dynamic_shape_process = GetTopCellDynamicFlag(input_args_info, obj_id_with_grad_order);
  top_cell_->set_use_dynamic_shape_process(use_dynamic_shape_process);
  top_cell_->set_need_save_dynamic_detect_nodes(IsNeedSaveDynamicDetectNodes(top_cell_, use_dynamic_shape_process));
  top_cell_->set_input_args_info(top_input_args_info_);
  PushHighOrderGraphStack(top_cell_);
  MS_LOG(DEBUG) << "New top graph, fg ptr " << fg.get() << " resource ptr " << resource.get();
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
  (void)PyNativeAlgo::Common::SetValueGradInfo(value, top_cell_, TensorGradType::kConstant);
  // Set last output abstract and will be used for sens
  if (forward()->enable_async()) {
    auto auto_grad_cell_ptr = top_cell()->auto_grad_cell_ptr();
    auto fake_v = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(value);
    auto task = [auto_grad_cell_ptr, fake_v]() { auto_grad_cell_ptr->UpdateOutputNodeOfTopCell(fake_v); };
    DispatchGradQueueTask(task);
  } else {
    top_cell()->auto_grad_cell_ptr()->UpdateOutputNodeOfTopCell(
      PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(value));
  }
}

void GradExecutor::EndGraphInner(const py::object &obj, const py::object &out, const py::args &args) {
  if (input_args_info_stack_.empty()) {
    return;
  }
  const auto input_args_info = input_args_info_stack_.top();
  MS_EXCEPTION_IF_NULL(input_args_info);
  MS_LOG(DEBUG) << "EndGraphInner start " << args.size() << ", cell_id " << GetCellId(obj, args, nullptr)
                << ", input args info ptr " << input_args_info.get();
  if (input_args_info->is_grad_topest_cell) {
    grad_flag_ = false;
    obj_order_ = 0;
  }
  GetCustomBpropPrim(obj, args, input_args_info);
  bool is_top_cell = (input_args_info->cell_id == top_cell()->cell_id());
  bool is_need_do_custom_grad = (input_args_info->has_custom_bprop && custom_bprop_cell_count_ == 0);
  bool is_custom_running = (input_args_info->grad_is_running && !bprop_grad_stack_.empty());
  if (is_top_cell || is_need_do_custom_grad || is_custom_running) {
    forward()->WaitForwardTask();
    input_args_info->out_value = PyNativeAlgo::DataConvert::PyObjToValue(out, false);
    EndGraphImpl(input_args_info);
  }
  PopInputArgsInfoStack();
}

void GradExecutor::EndGraphImpl(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  bool is_top_cell_end = (input_args_info->cell_id == top_cell()->cell_id());
  if (is_top_cell_end) {
    auto out_tensor = ConvertOutputValueToTensor(input_args_info->out_value);
    if (out_tensor != nullptr) {
      input_args_info->out_value = out_tensor;
    }
  }
  const auto &out_id = PyNativeAlgo::Common::GetIdByValue(input_args_info->out_value);
  DoGradForCustomBprop(input_args_info, out_id);
  // Update bprop grad stack
  if (input_args_info->grad_is_running && !bprop_grad_stack_.empty()) {
    if (!bprop_grad_stack_.top().second) {
      ValuePtrList inputs{input_args_info->out_value};
      AbstractBasePtrList abs{
        PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_args_info->out_value->ToAbstract())};
      auto node_info = std::make_shared<DynamicDetectNodeInfo>(nullptr, abs, nullptr);
      CheckGraphDynamic(inputs, node_info);
      auto output_node = GetInput(input_args_info->out_value, out_id);
      curr_g()->set_output(output_node);
      bprop_grad_stack_.pop();
      return;
    } else if (bprop_grad_stack_.top().first == input_args_info->cell_id) {
      bprop_grad_stack_.pop();
    }
  }
  // Just only dump the last forward graph
  if (is_top_cell_end) {
    ValuePtrList inputs{input_args_info->out_value};
    AbstractBasePtrList abs{PyNativeAlgo::Common::SetAbstractValueToAnyValue(input_args_info->out_value->ToAbstract())};
    auto node_info = std::make_shared<DynamicDetectNodeInfo>(nullptr, abs, nullptr);
    CheckGraphDynamic(inputs, node_info);
    SaveForwardGraph(input_args_info->out_value, out_id);
  }
  // Reset grad flag and update output node of the outermost cell
  if (input_args_info->is_grad_topest_cell && is_top_cell_end) {
    MS_LOG(DEBUG) << "Cur top last cell " << input_args_info->cell_id;
    (void)PopHighOrderGraphStack();
    SetForwardLastNodeInfo(input_args_info->out_value);
    top_cell()->ClearCellHookOp();
  }
  // Checkout whether need to compile graph when each top cell has run finished
  if (is_top_cell_end) {
    // In high grad cases, the output of the internal graph may be a tuple, and node needs to be created in the getobj
    if (!input_args_info->is_grad_topest_cell) {
      SetForwardLastNodeInfo(input_args_info->out_value);
    }
    top_cell()->CheckSubCellHookChanged();
    CheckNeedCompileGraph(input_args_info);
    top_input_args_info_ = input_args_info;
  }
}

void GradExecutor::DoGradForCustomBprop(const InputArgsInfoPtr &input_args_info, const std::string &out_id) const {
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (!input_args_info->has_custom_bprop || custom_bprop_cell_count_ != 0) {
    return;
  }
  MS_LOG(DEBUG) << "Do grad for custom bprop";
  MS_EXCEPTION_IF_NULL(input_args_info->custom_bprop_prim);
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->base_op_run_info.op_name = input_args_info->custom_bprop_prim->name();
  op_run_info->op_grad_info->op_prim = input_args_info->custom_bprop_prim;
  op_run_info->op_grad_info->input_value = input_args_info->input_arg_value_vec;
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
      PyNativeAlgo::Common::SetValueGradInfo(value, top_cell(), TensorGradType::kConstant);
  }
  (void)PyNativeAlgo::Common::SetValueGradInfo(op_run_info->real_out, nullptr, TensorGradType::kOpOutput);
  PyNativeAlgo::PyParser::PrepareOpGradInfo(op_run_info);
  DoOpGrad(op_run_info);
  auto node_info = std::make_shared<DynamicDetectNodeInfo>(
    op_run_info->op_grad_info->op_prim, op_run_info->op_grad_info->input_abs, op_run_info->base_op_run_info.abstract);
  CheckGraphDynamic(op_run_info->op_grad_info->input_value, node_info);
  RecordForwardGraph(op_run_info);
}

void GradExecutor::GetCustomBpropPrim(const py::object &obj, const py::args &args,
                                      const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (!input_args_info->has_custom_bprop) {
    return;
  }
  if (--custom_bprop_cell_count_ != 0) {
    return;
  }
  MS_LOG(DEBUG) << "Get custom bprop prim";
  py::function bprop_func = py::getattr(obj, parse::CUSTOM_BPROP_NAME);
  py::object code_obj = py::getattr(bprop_func, "__code__");
  // When the co_names is empty, we will still get a tuple which is empty.
  auto co_names = py::getattr(code_obj, "co_names").cast<py::tuple>();
  for (auto name : co_names) {
    if (!py::hasattr(obj, name)) {
      continue;
    }
    auto var = py::getattr(obj, name);
    if (py::hasattr(var, "__parameter__") && py::isinstance<tensor::MetaTensor>(var)) {
      MS_LOG(EXCEPTION) << "The user defined 'bprop' function does not support using Parameter.";
    }
  }

  py::object co_name = py::getattr(code_obj, "co_name");
  if (std::string(py::str(co_name)) == "staging_specialize") {
    MS_LOG(EXCEPTION) << "Decorating bprop with '@jit' is not supported.";
  }
  // Three parameters self, out and dout need to be excluded
  const size_t inputs_num = static_cast<size_t>(py::getattr(code_obj, "co_argcount").cast<int64_t>() - 3);
  if (inputs_num != args.size()) {
    MS_EXCEPTION(TypeError) << "Size of bprop func inputs[" << inputs_num
                            << "] is not equal to the size of cell inputs[" << args.size() << "]";
  }

  auto bprop_func_cellid = PyNativeAlgo::PyParser::GetIdByPyObj(bprop_func);
  (void)bprop_cell_list_.emplace_back(bprop_func_cellid);
  auto fake_prim = std::make_shared<PrimitivePy>(prim::kPrimHookBackward->name());
  MS_EXCEPTION_IF_NULL(input_args_info);
  if (py::isinstance<Cell>(obj)) {
    const auto &cell_ptr = obj.cast<CellPtr>();
    fake_prim->set_bprop_cls_name(cell_ptr->name());
  }
  fake_prim->AddBackwardHookFn(0, bprop_func);

  (void)fake_prim->AddAttr("cell_id", MakeValue(input_args_info->cell_id));
  (void)fake_prim->AddAttr(parse::CUSTOM_BPROP_NAME, MakeValue(true));
  if (input_args_info->input_arg_value_vec.empty()) {
    for (size_t i = 0; i < args.size(); ++i) {
      (void)input_args_info->input_arg_value_vec.emplace_back(PyNativeAlgo::DataConvert::PyObjToValue(args[i]));
    }
  }
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
      iter++;
    }
  }
}

void GradExecutor::CheckNeedCompileGraph(const InputArgsInfoPtr &input_args_info) {
  const auto &new_top_cell = top_cell();
  const auto &already_top_cell_id = new_top_cell->already_run_cell_id();
  // Update top cell by current cell op info
  auto pre_top_cell = GetAlreadyRunTopCell(already_top_cell_id);
  if (pre_top_cell == nullptr) {
    bool is_dynamic_cell_already_run = false;
    if (new_top_cell->use_dynamic_shape_process()) {
      // The dynamic cell of set_inputs needs to be saved for the first run.
      is_dynamic_cell_already_run =
        std::any_of(already_run_top_cell_.begin(), already_run_top_cell_.end(), [new_top_cell](const auto &item) {
          MS_EXCEPTION_IF_NULL(item.second);
          return item.second->obj_id_with_grad_order() == new_top_cell->obj_id_with_grad_order();
        });
    }
    if (!new_top_cell->use_dynamic_shape_process() || !is_dynamic_cell_already_run) {
      MS_LOG(DEBUG) << "Cell " << already_top_cell_id << " has never been ran, need compile graph";
      already_run_top_cell_[already_top_cell_id] = new_top_cell;
      return;
    }
  }

  MS_EXCEPTION_IF_NULL(input_args_info);
  // In high order situations, the internal top cell has changed, but outer top cell remains unchanged. Then outer
  // bprop graph need compile again
  if (new_top_cell->use_dynamic_shape_process() || new_top_cell->force_top_cell_compile()) {
    // Function need compile every time.
    new_top_cell->use_dynamic_shape_process() ? MS_LOG(DEBUG) << "The graph is dynamic, need to compile graph again"
                                              : MS_LOG(DEBUG) << "Force outer graph compile graph";
    auto has_higher_order = std::any_of(already_run_top_cell_.begin(), already_run_top_cell_.end(),
                                        [](const auto &elem) { return elem.second->is_high_order_top_cell(); });
    ClearPreTopCell(new_top_cell, input_args_info->is_grad_topest_cell && !has_higher_order);
    already_run_top_cell_[already_top_cell_id] = new_top_cell;
    new_top_cell->set_force_top_cell_compile(false);
  } else {
    MS_LOG(DEBUG) << "No need to compile graph again";
    pre_top_cell->set_input_args_id(new_top_cell->input_args_id());
    // In high order situations, the internal top cell remains unchanged, but the external top cell has changed. Then
    // the graph info of the internal top cell needs to be updated so that the external top cell can perceive it.
    if (!input_args_info->is_grad_topest_cell) {
      pre_top_cell->SetGraphInfoMap(pre_top_cell->fg(), new_top_cell->graph_info_map().at(new_top_cell->fg()));
    }
    pre_top_cell->set_forward_already_run(true);
  }
}

TopCellInfoPtr GradExecutor::GetAlreadyRunTopCell(const std::string &already_run_cell_id) const {
  const auto it = already_run_top_cell_.find(already_run_cell_id);
  if (it != already_run_top_cell_.end()) {
    return it->second;
  }
  return nullptr;
}

void GradExecutor::GradNetInner(const prim::GradOperationPtr &grad, const py::object &obj, const py::object &weights,
                                const py::object &grad_position, const py::args &args) {
  GetPreRunTopCell(grad, obj, args);
  SetSensValue(grad, top_input_args_info_, args);
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  MS_LOG(DEBUG) << "GradNetInner start " << args.size() << ", cell_id " << top_input_args_info_->cell_id
                << ", input args info ptr " << top_input_args_info_.get();

  // For async, top can not be change when run SetForwardLastNodeInfo; Change top cell after sync
  auto already_run_top_cell = already_run_top_cell_.at(top_cell()->already_run_cell_id());
  if (!already_run_top_cell->need_compile_graph()) {
    MS_LOG(DEBUG) << "No need compile graph";
    // If no need compile, we can clear construct bprop queue.
    (void)need_gc_top_cell_list_.emplace_back(top_cell());
    {
      GilReleaseWithCheck release_gil;
      async_executor_->Clear();
    }
    async_executor_->CheckException();
    top_cell()->ClearParamGradInfo();
    AsyncClearTopCell();
    set_top_cell(already_run_top_cell);
    top_cell()->UpdateTopCellInfo(false, false, false);
    return;
  }
  MS_LOG(DEBUG) << "Need compile graph";
  {
    GilReleaseWithCheck release_gil;
    async_executor_->Wait();
  }
  async_executor_->CheckException();
  set_top_cell(already_run_top_cell);
  AsyncClearTopCell();
  op_num_in_bprop_graph_ = top_cell()->op_index();
  top_cell()->set_grad_operation(grad_operation_);
  top_cell()->set_input_args_info(nullptr);
  SetBpropGraphJitLevel(obj);
  bool weight_param_is_tuple = true;
  // If current cell's parameter info has been cleared, we need resume its parameter grad info to construct bprop graph.
  ResumeParamGradInfo(top_cell());
  auto w_args = GetWeightsArgs(weights, &weight_param_is_tuple);
  auto p_args = GetGradPositionArgs(grad_position, grad->get_by_position_);
  autograd::GradAttr grad_attr(grad->get_all_, grad->get_by_list_, grad->sens_param_, grad->get_by_position_,
                               weight_param_is_tuple);
  GetGradGraph(grad_attr, w_args, p_args);
  top_cell()->ClearParamGradInfo();
}

std::string GradExecutor::GetAlreadyRunCellId(const std::string &cell_id) const {
  std::string already_run_cell_id(cell_id);
  already_run_cell_id += "_" + std::to_string(grad_order_ == 0 ? 1 : grad_order_);
  already_run_cell_id += "_" + grad_operation_;
  MS_LOG(DEBUG) << "Get already run top cell id " << already_run_cell_id;
  return already_run_cell_id;
}

void GradExecutor::GetPreRunTopCell(const prim::GradOperationPtr &grad, const py::object &obj, const py::args &args) {
  // @wrap_op
  // class A():
  //     def construct(self):
  // def wrap_op(op):
  //     class WrapOp(op):
  //         def __init(self, *args, *kwargs):
  //             self.net = op(*args, *kwargs) # self.net is A also
  //         def __call__(self, *args, *kwargs):
  //             out = super().__call(*args, *kwargs)
  //             Grad(self.net)
  //     return WrapOp
  // Run Grad(A), the following will happen:
  // 1、Create a new top cell for WrapOp, and run construct of A;
  // 2、Create a new top cell A, and get gradient, then set top_cell_ = nullptr;
  // 3、Here top_cell_ is nullptr, but gradient of WrapOp is not get. So, try find it in AlreadyRunCellId.
  if (top_cell_ != nullptr) {
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

  const auto &cell_id = GetCellId(obj, args_without_sens, nullptr);
  MS_LOG(DEBUG) << "Get pre run top cell cell id:" << cell_id;
  const auto &check_already_run_cell_id = GetAlreadyRunCellId(cell_id);
  top_cell_ = GetTopCell(check_already_run_cell_id);
  top_input_args_info_ = top_cell_->input_args_info();
}

void GradExecutor::GetGradGraph(const autograd::GradAttr &grad_attr, const std::vector<tensor::TensorPtr> &w_args,
                                const std::vector<size_t> &p_args) {
  // Get bprop graph of top cell
  auto bprop_graph = GetBpropGraph(grad_attr, w_args, p_args);
  AsyncClearAutoGradCell(top_cell());
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph, true);
  if (top_cell()->has_control_flow()) {
    (void)opt::EnvironConversion(resource);
  }
  if (grad_attr.has_sens && top_input_args_info_->input_arg_value_vec.back()->isa<ValueSequence>()) {
    PyNativeAlgo::Common::ProcessTupleParam(bprop_graph, top_input_args_info_->input_size);
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

std::vector<tensor::TensorPtr> GradExecutor::GetWeightsArgs(const py::object &weights,
                                                            bool *weight_param_is_tuple) const {
  std::vector<tensor::TensorPtr> w_args;
  if (py::hasattr(weights, "__parameter_tuple__")) {
    const auto &weights_tuple = weights.cast<py::tuple>();
    MS_LOG(DEBUG) << "Get weights tuple size " << weights_tuple.size();
    for (size_t i = 0; i < weights_tuple.size(); ++i) {
      const auto value = PyNativeAlgo::DataConvert::PyObjToValue(weights_tuple[i]);
      auto tensor = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      (void)w_args.emplace_back(tensor);
    }
  } else {
    MS_LOG(DEBUG) << "No parameter tuple get, try get weights params by input weight";
    if (py::isinstance<py::tuple>(weights) || py::isinstance<py::list>(weights)) {
      auto weights_tuple = py::cast<py::tuple>(weights);
      for (size_t i = 0; i < weights_tuple.size(); ++i) {
        const auto value = PyNativeAlgo::DataConvert::PyObjToValue(weights_tuple[i]);
        auto tensor = value->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        (void)w_args.emplace_back(tensor);
      }
    } else if (!py::isinstance<py::none>(weights)) {
      // Single input
      const auto value = PyNativeAlgo::DataConvert::PyObjToValue(weights);
      auto tensor = value->cast<tensor::TensorPtr>();
      (void)w_args.emplace_back(tensor);
      MS_EXCEPTION_IF_NULL(tensor);
      *weight_param_is_tuple = false;
    } else {
      return GetDefaultWeights();
    }
  }
  return w_args;
}

std::vector<tensor::TensorPtr> GradExecutor::GetDefaultWeights() const {
  std::vector<tensor::TensorPtr> w_args;
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
                                 << param_node->DebugString();
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

FuncGraphPtr GradExecutor::GetBpropGraph(const autograd::GradAttr &grad_attr, const vector<tensor::TensorPtr> &w_args,
                                         const vector<size_t> &p_args) {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  FuncGraphPtr bprop_graph = top_cell()->auto_grad_cell_ptr()->Finish(w_args, p_args, grad_attr);

  MS_LOG(DEBUG) << "Top graph input params size " << top_input_args_info_->input_arg_value_vec.size();
  UpdateParamAbsByArgs(top_input_args_info_->input_arg_value_vec, bprop_graph);
  if (top_cell()->need_do_final_opt()) {
    bprop_graph = BpropGraphFinalOpt(bprop_graph, top_cell()->has_control_flow());
  }
  if (top_input_args_info_->is_high_order_top_cell) {
    MS_LOG(DEBUG) << "Get high grad";
    top_cell()->resource()->set_optimize_graph(BasicClone(bprop_graph));
    PyNativeAlgo::Common::ReplaceCNodeWithValueNode(bprop_graph);
  } else {
    top_cell()->resource()->set_optimize_graph(bprop_graph);
  }
  if (top_cell()->has_control_flow()) {
    bprop_graph = LiftingClone(bprop_graph);
  }
  bprop_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop_graph->set_flag(kFlagIsPynativeBpropGraph, true);
  bprop_graph->set_flag(kFlagEnableRunGraphBySingleOp, top_cell()->use_dynamic_shape_process());
  if (top_cell()->has_call_graph()) {
    bprop_graph->set_flag(kFlagPyNativeWithJitCallGraph, true);
  }
  bprop_graph->set_attr(kAttrFuncGraphCellId, MakeValue(top_input_args_info_->obj_id));
  return bprop_graph;
}

void GradExecutor::SetGradOrder(const std::string &obj_id) {
  // top_cell_ == nullptr means call by grad first
  // top_cell_->obj_id_with_grad_order() include obj_id and grad_order
  // If top_cell_->obj_id_with_grad_order().find(obj_id) == std::string::npos and have cell info stack, means current
  // cell is not top cell, grad high order come in
  if (top_cell_ == nullptr || top_cell_->obj_id_with_grad_order().find(obj_id + "_") == std::string::npos) {
    IncreaseGradOrder();
  }
  if (!grad_is_running_) {
    MS_LOG(DEBUG) << "Grad not running yet";
    return;
  }
}

py::object GradExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj,
                                         const py::object &weights, const py::object &grad_hash_id,
                                         const py::args &args) {
  const auto &obj_id = PyNativeAlgo::PyParser::GetIdByPyObj(obj);

  // Check current cell grad order and erase it if in current top cell list
  SetGradOrder(obj_id);
  // Include weight param size and required grad flag
  std::string grad_hash_id_str;
  if (!py::isinstance<py::none>(grad_hash_id)) {
    grad_hash_id_str = std::string(py::str(grad_hash_id));
  }

  std::string weights_obj_id = GetWeightsObjIdsByWeights(weights);
  grad_operation_ = std::to_string(static_cast<int>(grad->get_all_)) +
                    std::to_string(static_cast<int>(grad->get_by_list_)) +
                    std::to_string(static_cast<int>(grad->sens_param_)) + grad_hash_id_str + weights_obj_id;

  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += PyNativeAlgo::PyParser::GetIdByPyObj(args[i]) + "_";
  }
  // Under the condition that the stack is empty (forward process completed or no forward process),
  // check whether need to run forward process
  bool forward_run = false;
  if (input_args_info_stack_.empty() && top_cell_ != nullptr) {
    auto cell_id = GetCellId(obj, args, nullptr);
    const auto &check_already_run_cell_id = GetAlreadyRunCellId(cell_id);
    auto find_top_cell = GetTopCell(check_already_run_cell_id);
    if (find_top_cell != nullptr) {
      MS_LOG(DEBUG) << "Find already run top cell " << find_top_cell->cell_id();
      forward_run = top_cell()->forward_already_run();
      bool input_args_changed = !top_cell()->input_args_id().empty() && top_cell()->input_args_id() != input_args_id;
      if (forward_run && input_args_changed) {
        MS_LOG(WARNING) << "The input info " << input_args_id << " is not the same with pre input info "
                        << top_cell()->input_args_id() << ", forward process will run again";
        forward_run = false;
        ClearParamGradInfo(top_cell());
      }
    }
  }
  MS_LOG(DEBUG) << "Graph have already ran " << forward_run << " top cell id " << obj_id;
  return BaseRefToPyData(forward_run);
}

py::object GradExecutor::RunGradGraph() {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  const auto &resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Run cell id " << top_input_args_info_->cell_id << ", resource ptr " << resource.get();
  VectorRef arg_list;
  SetGraphInputArgs(
    top_input_args_info_->input_arg_value_vec, resource,
    top_input_args_info_->has_sens && top_input_args_info_->input_arg_value_vec.back()->isa<ValueSequence>(),
    &arg_list);
  MS_LOG(DEBUG) << "Convert args size " << top_input_args_info_->input_arg_value_vec.size() << ", graph param size "
                << arg_list.size();
  compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
  MS_EXCEPTION_IF_NULL(run);

  const auto &backend = MsContext::GetInstance()->backend_policy();
  MS_LOG(DEBUG) << "Eval run " << backend;
  grad_is_running_ = true;
  // In custom bprop, when running bprop function, top_input_args_info_ will be changed.
  // So, here copy and restore after running finished.
  auto top_input_args_info = top_input_args_info_;
  BaseRef out_value = (*run)(arg_list);
  top_input_args_info_ = top_input_args_info;
  grad_is_running_ = false;
  MS_LOG(DEBUG) << "Eval run end " << out_value.ToString();
  const auto &cur_run_bprop_graph = resource->optimize_graph();
  auto out_abs = cur_run_bprop_graph->output()->abstract();
  MakeNestedCnode(top_input_args_info_->has_custom_bprop, top_input_args_info_->input_arg_value_vec,
                  cur_run_bprop_graph, out_value);
  top_input_args_info_ = nullptr;
  return BaseRefToPyData(out_value, out_abs);
}

void GradExecutor::MakeNestedCnode(bool has_custom_bprop, const std::vector<ValuePtr> &forward_args,
                                   const FuncGraphPtr &cur_run_bprop_graph, const BaseRef &out) {
  MS_EXCEPTION_IF_NULL(top_input_args_info_);
  if (top_input_args_info_->is_grad_topest_cell) {
    MS_LOG(DEBUG) << "No nested grad find";
    ClearGradRes();
    return;
  }
  MS_LOG(DEBUG) << "Do high grad";
  // first_grad_fg maybe modified in auto grad, and first_grad_fg can be used multiple times
  auto first_grad_fg = cur_run_bprop_graph;
  if (has_custom_bprop) {
    first_grad_fg = curr_g();
    MS_LOG(DEBUG) << "Bprop nested";
  }
  MS_EXCEPTION_IF_NULL(first_grad_fg);
  PyNativeAlgo::Common::DumpGraphIR("first_grad_fg.ir", first_grad_fg);
  ValuePtrList weights_args;
  const std::string &cur_top_cell_id = top_cell()->obj_id_with_grad_order();
  bool use_dynamic_shape_process = top_cell()->use_dynamic_shape_process() || top_cell()->vm_compile();
  bool has_call_graph = top_cell()->has_call_graph();
  auto inner_graph_info = top_cell()->graph_info_map().at(curr_g());
  SwitchTopCell();
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->op_grad_info->input_value = forward_args;
  op_run_info->input_size = forward_args.size();
  op_run_info->op_grad_info->input_value_grad_type.resize(op_run_info->input_size);
  op_run_info->input_unused_in_bprop.resize(op_run_info->input_size, false);
  auto out_value = PyNativeAlgo::DataConvert::BaseRefToValue(out, true, true);
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
  auto op_grad_info = std::make_shared<OpGradInfo>();
  op_grad_info->input_value = op_run_info->op_grad_info->input_value;
  op_grad_info->input_abs = op_run_info->op_grad_info->input_abs;
  op_grad_info->out_value = out_value;
  op_grad_info->out_abs = first_grad_fg->output()->abstract();
  op_grad_info->input_value_grad_type = op_run_info->op_grad_info->input_value_grad_type;
  auto grad_param = std::make_shared<autograd::GradParam>(op_grad_info, !top_cell()->is_high_order_top_cell(),
                                                          use_dynamic_shape_process);
  grad_param->fg = grad_fg;
  grad_param->source_fg = first_grad_fg;
  grad_param->is_control_flow = has_call_graph;
  grad_param->graph_cache_key = cur_top_cell_id;
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
  // Clear current top cell res
  DecreaseGradOrder();
  // Get outer top cell
  auto outer_top_cell = PopHighOrderGraphStack();
  MS_EXCEPTION_IF_NULL(outer_top_cell);
  // If inner graph compile graph, outer must be compile
  if (top_cell()->vm_compile()) {
    outer_top_cell->set_force_top_cell_compile(true);
    outer_top_cell->set_use_dynamic_shape_process(top_cell()->use_dynamic_shape_process());
  }
  ResumeParamGradInfo(outer_top_cell);
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
  auto has_higher_order = std::any_of(already_run_top_cell_.begin(), already_run_top_cell_.end(),
                                      [](const auto &elem) { return elem.second->is_high_order_top_cell(); });
  // Custom bprop nested, top cell reset by first time, second time no need clean
  if (top_cell_ != nullptr) {
    // High order must no clean
    if (!has_higher_order) {
      top_cell_->ClearDeviceMemory();
    }
    const auto &pre_top_cell = GetAlreadyRunTopCell(top_cell()->already_run_cell_id());
    if (top_cell_->use_dynamic_shape_process() || pre_top_cell != nullptr) {
      top_cell_ = nullptr;
    }
  }
  top_input_args_info_ = nullptr;
  DecreaseGradOrder();
  ClearGlobalRes();
  grad_operation_.clear();
}

void GradExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear grad res";
  grad_flag_ = false;
  grad_is_running_ = false;
  custom_bprop_cell_count_ = 0;
  grad_order_ = 0;
  top_cell_ = nullptr;
  top_input_args_info_ = nullptr;
  bprop_cell_list_.clear();
  grad_operation_.clear();
  already_run_top_cell_.clear();
  cell_id_with_dynamic_detect_nodes_.clear();
  std::stack<InputArgsInfoPtr>().swap(input_args_info_stack_);
  std::stack<std::pair<std::string, bool>>().swap(bprop_grad_stack_);
  std::stack<TopCellInfoPtr>().swap(high_order_stack_);
  forward_use_dynamic_shape_process_ = false;
}

void GradExecutor::AsyncClearTopCell() {
  for (const auto &need_gc_top_cell : need_gc_top_cell_list_) {
    if (forward()->enable_async()) {
      auto task = [need_gc_top_cell]() { need_gc_top_cell->Clear(); };
      DispatchGradQueueTask(task);
    } else {
      need_gc_top_cell->Clear();
    }
  }
  need_gc_top_cell_list_.clear();
}

void GradExecutor::AsyncClearAutoGradCell(const TopCellInfoPtr &top_cell) {
  if (forward()->enable_async()) {
    auto task = [top_cell] { top_cell->set_auto_grad_cell_ptr(nullptr); };
    DispatchGradQueueTask(task);
  } else {
    top_cell->set_auto_grad_cell_ptr(nullptr);
  }
}

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
    const auto &tensor = v->cast<tensor::TensorPtr>();
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

TopCellInfoPtr GradExecutor::GetTopCell(const std::string &already_run_cell_id) {
  TopCellInfoPtr find_top_cell = nullptr;
  for (const auto &[cell_id, top_cell] : already_run_top_cell_) {
    MS_EXCEPTION_IF_NULL(top_cell);
    MS_LOG(DEBUG) << "Get top cell id " << cell_id;
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
    // Partial match, means run grad first, but follow a other net grad
    if (top_cell->already_run_cell_id().find(already_run_cell_id) != std::string::npos &&
        already_run_cell_id.back() == '_') {
      find_top_cell = top_cell;
      break;
    }
  }
  // Same topcell info, but grad operation is not the same, construct backward graph again
  if (find_top_cell != nullptr) {
    if (!find_top_cell->grad_operation().empty() && find_top_cell->grad_operation() != grad_operation_) {
      MS_LOG(DEBUG) << "Already exist grad operation " << find_top_cell->grad_operation() << " is different with new "
                    << grad_operation_;
      (void)already_run_top_cell_.erase(find_top_cell->already_run_cell_id());
      return nullptr;
    } else {
      return find_top_cell;
    }
  }
  return nullptr;
}

void GradExecutor::SetHookChanged(const py::object &cell) const {
  if (top_cell_ == nullptr) {
    return;
  }
  const auto &cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  if (top_cell_->cell_id().find(cell_id) != std::string::npos) {
    top_cell_->set_hook_changed(true);
  }
  if (RequiresGrad()) {
    top_cell_->set_sub_cell_hook_changed(cell_id);
  }
}

void GradExecutor::ProcessOpGradInfo(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  RecordForwardGraph(op_run_info);
  DoOpGrad(op_run_info);
  if (op_run_info->stub_output != nullptr) {
    op_run_info->stub_output->SetValue(op_run_info->real_out);
  }
  top_cell()->GetOpInfo(op_run_info);
  UpdateTopCellForwardTensorInfoInBpropGraph(op_run_info->op_info, op_run_info->real_out);
  auto node_info = std::make_shared<DynamicDetectNodeInfo>(
    op_run_info->op_grad_info->op_prim, op_run_info->op_grad_info->input_abs, op_run_info->op_grad_info->out_abs);
  CheckGraphDynamic(op_run_info->op_grad_info->input_value, node_info);
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
  if (grad_is_running_ && !bprop_grad_stack_.top().second) {
    MS_LOG(DEBUG) << "Custom bprop, no need do op grad";
    return;
  }
  auto &&grad_param = CreateOpGradParam(op_run_info, top_cell());
  if (forward()->enable_async()) {
    auto auto_grad_cell_ptr = top_cell()->auto_grad_cell_ptr();
    auto task = [auto_grad_cell_ptr, grad_param]() { auto_grad_cell_ptr->KPynativeOp(grad_param); };
    DispatchGradQueueTask(task);
  } else {
    top_cell()->auto_grad_cell_ptr()->KPynativeOp(grad_param);
  }
}

void GradExecutor::UpdateTopCellForwardTensorInfoInBpropGraph(const std::string &op_info, const ValuePtr &v) const {
  const auto &pre_top_cell = GetAlreadyRunTopCell(top_cell()->already_run_cell_id());
  // The shape of the last two steps is the same, and the pre_top_cell is not empty.
  // But if dynamic shape is enabled at this point, you still need to execute SaveTensorIdWithOpInfo.
  if (pre_top_cell == nullptr || use_dynamic_shape_process()) {
    // First run top cell, save op output info for replace
    top_cell()->SaveTensorIdWithOpInfo(op_info, v);
    MS_LOG(DEBUG) << "Top cell " << top_cell_->already_run_cell_id() << " run firstly, op info " << op_info;
    return;
  }

  // In dynamic process, no need replace
  if (top_cell()->use_dynamic_shape_process()) {
    return;
  }

  // Not first run top cell, do update
  MS_LOG(DEBUG) << "Update top cell forward output tensor info " << op_info;
  UpdateForwardOutputTensorInfo(op_info, v, pre_top_cell->replace_info());
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
    top_cell()->RecordCellBackwardHookOp(op_run_info->cell_obj_id, cnode);
  }
  MS_LOG(DEBUG) << "Make CNode for " << op_run_info->base_op_run_info.op_name << ", new cnode is "
                << cnode->DebugString();
  return cnode;
}

void GradExecutor::RecordForwardGraph(const FrontendOpRunInfoPtr &op_run_info) const {
  if (save_graphs_ || grad_is_running_) {
    MS_EXCEPTION_IF_NULL(op_run_info);
    const auto &cnode = ConstructForwardGraph(op_run_info);
    MS_EXCEPTION_IF_NULL(cnode);
    cnode->set_abstract(op_run_info->base_op_run_info.abstract);
    SaveOutputNodeMap(op_run_info->out_value_id, op_run_info, cnode);
  }
}

void GradExecutor::RecordForwardGraphForInput(const ValuePtr &value, const string &input_id,
                                              const abstract::AbstractBasePtr &param_abs) {
  save_graphs_ = MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs_) {
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

void GradExecutor::SaveForwardGraph(const ValuePtr &value, const string &value_id) const {
  auto context = MsContext::GetInstance();
  if (context->CanDump(kIntroductory)) {
    auto output_node = GetInput(value, value_id);
    curr_g()->set_output(output_node);
    PyNativeAlgo::Common::DumpGraphIR("fg.ir", curr_g());
  }
}

void GradExecutor::SaveInputTensorGradInfo(const InputArgsInfoPtr &input_args_info) {
  MS_EXCEPTION_IF_NULL(input_args_info);
  for (size_t i = 0; i < input_args_info->input_size; ++i) {
    const auto &v = input_args_info->input_arg_value_vec[i];
    BackupInputTensorGradInfo(v);
  }
}

void GradExecutor::BackupInputTensorGradInfo(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor_value = value->cast<tensor::TensorPtr>();
    auto auto_grad_meta_data = tensor_value->auto_grad_meta_data();
    if (auto_grad_meta_data != nullptr) {
      top_cell()->AddParamGradInfo(tensor_value, auto_grad_meta_data);
    }
  } else if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>();
    for (auto elem : value_seq->value()) {
      BackupInputTensorGradInfo(elem);
    }
  } else if (value->isa<stub::StubNode>()) {
    auto stub_node = value->cast<stub::StubNodePtr>();
    MS_EXCEPTION_IF_NULL(stub_node);
    BackupInputTensorGradInfo(stub_node->WaitValue());
  }
}

void GradExecutor::ClearParamGradInfo(const TopCellInfoPtr &top_cell) const {
  if (top_cell == nullptr || top_cell->param_grad_info().empty()) {
    return;
  }
  for (auto &params : top_cell->param_grad_info()) {
    params.first->set_auto_grad_meta_data(nullptr);
  }
  top_cell->set_resume_flag(true);
}

void GradExecutor::ResumeParamGradInfo(const TopCellInfoPtr &top_cell) const {
  if (top_cell == nullptr || !top_cell->resume_flag() || top_cell->param_grad_info().empty()) {
    return;
  }
  const auto &param_grad_info = top_cell->param_grad_info();
  for (auto &params : param_grad_info) {
    params.first->set_auto_grad_meta_data(params.second);
  }
  top_cell->set_resume_flag(false);
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

void GradExecutor::SaveDynamicDetectNodeInfoInFirstTime(const ValuePtrList &inputs,
                                                        const DynamicDetectNodeInfoPtr &node, size_t node_idx) const {
  MS_EXCEPTION_IF_NULL(node);
  node->inputs = BuildDynamicDetectNodeInputs(inputs);
  (void)cell_id_with_dynamic_detect_nodes_[top_cell()->obj_id_with_grad_order()][top_cell()->cell_id()].emplace_back(
    node);
  MS_LOG(DEBUG) << "Save node " << (node->op_prim != nullptr ? node->op_prim->name() : "")
                << " firstly, node_idx: " << node_idx << ", is_jit_node: " << node->is_graph_node
                << ", graph_phase:" << node->graph_phase
                << " obj_id_with_grad_order:" << top_cell()->obj_id_with_grad_order()
                << ", cell id:" << top_cell()->cell_id();
}

void GradExecutor::SaveDynamicInputsCells(const py::object &cell) {
  const auto &cell_id = PyNativeAlgo::PyParser::GetIdByPyObj(cell);
  MS_LOG(INFO) << "SaveDynamicInputsCells:" << cell_id;
  dynamic_inputs_cells_.insert(cell_id);
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

bool GradExecutor::IsGraphDynamic(const ValuePtrList &inputs, const DynamicDetectNodeInfoPtr &node,
                                  size_t node_idx) const {
  MS_EXCEPTION_IF_NULL(node);
  if (top_cell()->is_need_save_dynamic_detect_nodes()) {
    SaveDynamicDetectNodeInfoInFirstTime(inputs, node, node_idx);
    // The net is regarded as a static net by default in the first time.
    return false;
  }

  MS_LOG(DEBUG) << "Check node " << (node->op_prim != nullptr ? node->op_prim->name() : "") << " node_idx: " << node_idx
                << ", is_jit_node: " << node->is_graph_node << ", graph_phase:" << node->graph_phase
                << " obj_id_with_grad_order:" << top_cell()->obj_id_with_grad_order()
                << ", cell id:" << top_cell()->cell_id();
  const auto &dynamic_nodes =
    cell_id_with_dynamic_detect_nodes_[top_cell()->obj_id_with_grad_order()][top_cell()->cell_id()];
  if (node_idx >= dynamic_nodes.size()) {
    MS_LOG(DEBUG) << "Old dynamic_nodes size: " << dynamic_nodes.size() << ", cur node_idx is: " << node_idx
                  << ", graph is dynamic.";
    return true;
  }

  // 1.Detect jit phase
  const DynamicDetectNodeInfoPtr &old_node_info = dynamic_nodes[node_idx];
  if (node->is_graph_node) {
    if (!old_node_info->is_graph_node || node->graph_phase != old_node_info->graph_phase) {
      MS_LOG(DEBUG) << "Graph is dynamic, old is_graph_node: " << old_node_info->is_graph_node
                    << " new is_graph_node: " << node->is_graph_node << " old graph_phase "
                    << old_node_info->graph_phase << " new graph_phase: " << node->graph_phase;
      return true;
    }
    return false;
  }

  if (IsDynamicDetectPrimChange(old_node_info->op_prim, node->op_prim)) {
    MS_LOG(DEBUG) << "Graph is dynamic, old node prim: " << old_node_info->op_prim->name()
                  << " new node prim: " << (node->op_prim != nullptr ? node->op_prim->name() : "")
                  << " node_idx: " << node_idx;
    return true;
  }

  // Compare input abs
  if (IsDynamicDetectAbsChange(old_node_info->input_abs, node->input_abs)) {
    return true;
  }

  // Compare out abs
  if (IsDynamicDetectAbsChange(old_node_info->out_abs, node->out_abs)) {
    return true;
  }

  // Get input
  node->inputs = BuildDynamicDetectNodeInputs(inputs);

  // Compare input
  if (IsDynamicDetectInputChange(old_node_info->inputs, node->inputs)) {
    return true;
  }
  return false;
}

void GradExecutor::CheckGraphDynamic(const ValuePtrList &inputs, const DynamicDetectNodeInfoPtr &node) const {
  std::unique_lock<std::mutex> lock(async_mutex_);
  if (top_cell()->use_dynamic_shape_process()) {
    top_cell()->IncreaseOpIndex();
    return;
  }

  const size_t node_idx = top_cell()->op_index();
  bool use_dynamic_shape_process = IsGraphDynamic(inputs, node, node_idx);
  top_cell()->IncreaseOpIndex();
  if (use_dynamic_shape_process) {
    MS_LOG(INFO) << "Set use_dynamic_shape_process: " << use_dynamic_shape_process;
    top_cell()->set_use_dynamic_shape_process(use_dynamic_shape_process);
    py::gil_scoped_acquire gil_acquire;
    (void)cell_id_with_dynamic_detect_nodes_.erase(top_cell()->obj_id_with_grad_order());
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE)) {
      MS_LOG(WARNING) << "Detect dynamic shape or dynamic graph structure, the python stack is:";
      py::gil_scoped_acquire acquire_gil;
      py::exec(R"(
                  import traceback
                  traceback.print_stack()
                  )");
    }
  }
}

void GradExecutor::DispatchGradQueueTask(const std::function<void(void)> task) const {
  bool success = async_executor_->Push(new (std::nothrow) BpropTask(std::move(task)));
  if (!success) {
    async_executor_->CheckException();
  }
}
}  // namespace pynative
}  // namespace mindspore
