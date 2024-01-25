/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/grad/function/meta_grad.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "kernel/pyboost/auto_generate/add.h"
#include "include/common/utils/primitive_utils.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/grad/function/function_utils.h"
#include "ops/framework_ops.h"
#include "ops/math_ops.h"
#include "ops/other_ops.h"
#include "ops/sequence_ops.h"
#include "ops/structure_ops.h"

namespace mindspore::pynative::autograd {
enum class SpecialType { kZerosLikeType = 0, kOnesLikeType = 1 };
const mindspore::HashSet<std::string> kGradBlackList{kMakeTupleOpName,         kMakeListOpName,
                                                     kTupleGetItemOpName,      kStopGradientOpName,
                                                     kUpdateStateOpName,       kNPUAllocFloatStatusOpName,
                                                     kNPUGetFloatStatusOpName, kNPUClearFloatStatusOpName};
namespace {
inline bool IsPrimNeedGrad(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return kGradBlackList.find(prim->name()) == kGradBlackList.end();
}

bool NeedGrad(const ValuePtrList &input_values) {
  for (auto &input_arg : input_values) {
    MS_EXCEPTION_IF_NULL(input_arg);
    if (input_arg->isa<tensor::Tensor>()) {
      const auto &input_tensor = input_arg->cast<tensor::TensorPtr>();
      auto auto_grad_meta_data = input_tensor->auto_grad_meta_data();
      MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
      if (PyNativeAlgo::Common::IsParam(auto_grad_meta_data->input_type())) {
        return true;
      }
      auto variable = auto_grad_meta_data->variable();
      if (variable != nullptr) {
        return true;
      }
    } else if (input_arg->isa<ValueSequence>()) {
      auto value_seq = input_arg->cast<ValueSequencePtr>()->value();
      if (NeedGrad(value_seq)) {
        return true;
      }
    } else if (input_arg->isa<tensor::COOTensor>() || input_arg->isa<tensor::CSRTensor>()) {
      return true;
    }
  }
  return false;
}

void SetGradMetaData(const ValuePtrList &flatten_outs, const VariablePtr &variable) {
  for (size_t i = 0; i < flatten_outs.size(); ++i) {
    const auto &out = flatten_outs[i];
    if (out->isa<tensor::Tensor>()) {
      auto tensor = out->cast<tensor::TensorPtr>();
      auto auto_grad_meta_data = tensor->auto_grad_meta_data();
      if (auto_grad_meta_data == nullptr) {
        MS_LOG(DEBUG) << "tensor has no auto_grad_meta_data";
        auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
        tensor->set_auto_grad_meta_data(auto_grad_meta_data);
      }
      auto_grad_meta_data->set_variable(variable);
      auto_grad_meta_data->set_output_index(i);
    }
  }
}

TensorPtr Add(const TensorPtr &input, const TensorPtr &other) {
  if (input == nullptr) {
    MS_EXCEPTION_IF_NULL(other);
    return other;
  }
  if (other == nullptr) {
    MS_EXCEPTION_IF_NULL(input);
    return input;
  }
  // Create op
  auto op = CREATE_PYBOOST_OP(Add, MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  op->set_primitive(prim::kPrimAdd);
  (void)op->Call(input, other);
  return op->output(0);
}

std::vector<TensorPtr> Add(const std::vector<TensorPtr> &inputs, const TensorPtr &other, size_t input_index) {
  if (input_index >= inputs.size()) {
    MS_LOG(EXCEPTION) << "The input index should less than inputs size";
  }

  std::vector<TensorPtr> outputs(inputs);
  outputs[input_index] = Add(inputs[input_index], other);
  return outputs;
}

std::vector<TensorPtr> SetAlign(const tensor::TensorPtr &grad, size_t output_size, size_t input_index) {
  std::vector<TensorPtr> gradients;
  gradients.reserve(output_size);
  for (size_t i = 0; i < output_size; ++i) {
    if (input_index == i) {
      gradients.emplace_back(grad);
    } else {
      // If gradient is not, we just set nullptr, then we lazy update zero gradient by
      // LazeUpdateZeroGradient method
      gradients.emplace_back(nullptr);
    }
  }
  return gradients;
}

VectorRef GeneratePythonArgs(const ValuePtrList &inputs, const ValuePtr &output) {
  VectorRef args;
  for (const auto &value : inputs) {
    (void)args.emplace_back(value);
  }
  (void)args.emplace_back(output);
  return args;
}

ValuePtr TensorToValue(const std::vector<TensorPtr> tensors) {
  if (tensors.size() == kSizeZero) {
    MS_LOG(EXCEPTION) << "tensors size should not be empty!";
  }
  if (tensors.size() == kSizeOne) {
    return tensors[kIndex0];
  }
  std::vector<ValuePtr> values;
  values.reserve(tensors.size());
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(values), [](const ValuePtr &val) { return val; });
  return std::make_shared<ValueTuple>(values);
}

void ConvertPyObjectToTensor(const py::object &input_object, std::vector<ValuePtr> *tensors) {
  MS_EXCEPTION_IF_NULL(tensors);
  ValuePtr tensor_ptr = nullptr;
  if (py::isinstance<tensor::Tensor>(input_object)) {
    tensor_ptr = py::cast<tensor::TensorPtr>(input_object);
  } else if (IsStubTensor(input_object)) {
    tensor_ptr = ConvertStubTensor(input_object);
  } else if (py::isinstance<py::float_>(input_object)) {
    double input_value = py::cast<py::float_>(input_object);
    tensor_ptr = std::make_shared<tensor::Tensor>(input_value, kFloat32);
  } else if (py::isinstance<py::int_>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<int64_t>(input_object), kInt64);
  } else if (py::isinstance<py::list>(input_object)) {
    auto list_inputs = py::cast<py::list>(input_object);
    for (size_t i = 0; i < list_inputs.size(); ++i) {
      ConvertPyObjectToTensor(list_inputs[i], tensors);
    }
    return;
  } else if (py::isinstance<py::tuple>(input_object)) {
    auto tuple_inputs = py::cast<py::tuple>(input_object);
    for (size_t i = 0; i < tuple_inputs.size(); ++i) {
      ConvertPyObjectToTensor(tuple_inputs[i], tensors);
    }
    return;
  } else if (py::isinstance<tensor::CSRTensor>(input_object)) {
    tensor_ptr = py::cast<tensor::CSRTensorPtr>(input_object);
  } else if (py::isinstance<tensor::COOTensor>(input_object)) {
    tensor_ptr = py::cast<tensor::COOTensorPtr>(input_object);
  } else {
    MS_EXCEPTION(TypeError) << "Unreasonable data type: " << input_object.get_type() << ".";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)tensors->emplace_back(tensor_ptr);
}

void SetGradInfoForInputs(const ValuePtr &value, const VariablePtr &variable) {
  if (value->isa<tensor::Tensor>()) {
    const auto &input_tensor = value->cast<tensor::TensorPtr>();
    const auto &auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_variable(variable);
  } else if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    const auto &indices_tensor = coo_tensor->GetIndices();
    SetGradInfoForInputs(indices_tensor, variable);
  } else if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    const auto &indices_tensor = csr_tensor->GetIndices();
    SetGradInfoForInputs(indices_tensor, variable);
  }
}

bool IsOutputBothEmpty(const ValuePtr &input_grads, const ValuePtr &weight_grads) {
  if (!input_grads->isa<ValueTuple>() || !weight_grads->isa<ValueTuple>()) {
    return false;
  }
  auto input_grads_tuple = input_grads->cast<ValueTuplePtr>();
  auto weight_grads_tuple = weight_grads->cast<ValueTuplePtr>();
  return input_grads_tuple->size() == 0 && weight_grads_tuple->size() == 0;
}

ValuePtr GenerateEmptyTupleValue() {
  std::vector<ValuePtr> value_list;
  auto inputs_value = std::make_shared<ValueTuple>(value_list);
  auto weights_value = std::make_shared<ValueTuple>(value_list);
  std::vector<ValuePtr> tuple_list{inputs_value, weights_value};
  return std::make_shared<ValueTuple>(tuple_list);
}

std::string PrintGradients(const tensor::TensorPtrList &gradients) {
  for (size_t i = 0; i < gradients.size(); ++i) {
    const auto &grad = gradients[i];
    if (grad == nullptr) {
      MS_LOG(DEBUG) << "The " << i << "'th gradient is nullptr!";
      continue;
    }
    auto tensor = std::make_shared<tensor::Tensor>(*grad);
    tensor->data_sync();
    MS_LOG(DEBUG) << "The " << i << "'th gradient: " << tensor->ToStringRepr();
  }
  return "";
}

ValuePtr WrapCOOTensor(const ValuePtr &coo_out, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(coo_out);
  auto coo_tensor = coo_out->cast<tensor::COOTensorPtr>();
  MS_EXCEPTION_IF_NULL(coo_tensor);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(value_tensor);
  auto indices_tensor = coo_tensor->GetIndices();
  auto shape_vector = coo_tensor->shape();
  return std::make_shared<tensor::COOTensor>(indices_tensor, value_tensor, shape_vector);
}

ValuePtr WrapCSRTensor(const ValuePtr &csr_out, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(csr_out);
  auto csr_tensor = csr_out->cast<tensor::CSRTensorPtr>();
  MS_EXCEPTION_IF_NULL(csr_tensor);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(value_tensor);
  auto indptr_tensor = csr_tensor->GetIndptr();
  auto indices_tensor = csr_tensor->GetIndices();
  auto shape_vector = csr_tensor->shape();
  return std::make_shared<tensor::CSRTensor>(indptr_tensor, indices_tensor, value_tensor, shape_vector);
}

ValuePtr BuildSpecialGrad(const ValuePtr &value, const tensor::TensorPtr &grad, const FuncBuilderPtr &func_impl) {
  if (value->isa<tensor::Tensor>()) {
    if (grad != nullptr) {
      return grad;
    }
    return func_impl->Zeros(value);
  } else if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return WrapCSRTensor(csr_tensor, BuildSpecialGrad(csr_tensor->GetValues(), grad, func_impl));
  } else if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    return WrapCOOTensor(coo_tensor, BuildSpecialGrad(coo_tensor->GetValues(), grad, func_impl));
  } else {
    MS_LOG(EXCEPTION) << "The value type not support grad: " << value->ToString();
  }
}
}  // namespace

TensorPtrList FuncBackwardNode::CallBackward(const TensorPtrList &gradients_in) {
  MS_LOG(DEBUG) << "Begin CallBackward: " << name();
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto ir_builder = FuncBuilder(name_, device_target, nullptr);
  auto inputs = PreProcess(gradients_in, &ir_builder);
  ir_builder.SetInputs(name(), &inputs, &attrs_);
  const std::vector<NodePtr> cal_grads_node = func()(&ir_builder);
  ValuePtrList cal_grads_values;
  std::transform(cal_grads_node.begin(), cal_grads_node.end(), std::back_inserter(cal_grads_values),
                 [](const NodePtr &node) { return node->Value(); });
  auto gradients = PostProcess(cal_grads_values);
  MS_LOG(DEBUG) << "End CallBackward" << name();
  return gradients;
}

NodePtrList FuncBackwardNode::PreProcess(const TensorPtrList &dout, FuncBuilder *emitter) {
  NodePtrList node_inputs;
  node_inputs.reserve(op_inputs_.size() + kSizeFive);
  for (size_t i = 0; i < op_inputs_.size(); ++i) {
    (void)node_inputs.emplace_back(emitter->NewFuncNode(op_inputs_[i], grad_type_[i]));
  }
  (void)node_inputs.emplace_back(emitter->NewFuncNode(op_output_, InputType::kOpOutput));
  if (dout.size() == kSizeOne) {
    (void)node_inputs.emplace_back(emitter->NewFuncNode(dout[kIndex0], InputType::kOpOutput));
  } else {
    ValuePtrList value_dout;
    value_dout.reserve(dout.size());
    // If dout is nullptr, lazy update zero tensor to expander.
    std::transform(dout.begin(), dout.end(), std::back_inserter(value_dout), [](const auto &val) -> ValuePtr {
      if (val == nullptr) {
        return kNone;
      }
      return val;
    });
    (void)node_inputs.emplace_back(
      emitter->NewFuncNode(std::make_shared<ValueTuple>(value_dout), InputType::kOpOutput));
  }
  return node_inputs;
}

TensorPtrList FuncBackwardNode::LazeUpdateZeroGradient(const TensorPtrList &dout, FuncBuilder *emitter) {
  if (dout.size() == kSizeOne) {
    return dout;
  }
  auto outputs = FlattenArgs({op_output_});
  if (outputs.size() != dout.size()) {
    MS_LOG(EXCEPTION) << "gradients size should be same as output size! but got output size: " << outputs.size()
                      << ", gradients size: " << dout.size();
  }
  TensorPtrList real_dout(dout.size());
  for (size_t i = 0; i < dout.size(); ++i) {
    if (dout[i] == nullptr) {
      MS_LOG(DEBUG) << "Op " << name() << "has multi outputs, and exist null dout";
      auto zero_value = emitter->Zeros(outputs[i]);
      MS_EXCEPTION_IF_NULL(zero_value);
      auto zero_tensor = zero_value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(zero_tensor);
      real_dout[i] = zero_tensor;
    } else {
      real_dout[i] = dout[i];
    }
  }
  return real_dout;
}

TensorPtrList HookBackwardNode::CallBackward(const TensorPtrList &grads) {
  MS_LOG(DEBUG) << "Begin HookBackwardNode CallBackward ";
  auto gradient = TensorToValue(grads);
  (void)args_.emplace_back(gradient);
  auto out = prim_->RunHookFunction(args_);
  ValuePtrList gradient_values;
  if (utils::isa<PyObjectRef>(out)) {
    PyObjectRef py_ref = utils::cast<PyObjectRef>(out);
    auto out_py_tuple = py_ref.object_;
    ConvertPyObjectToTensor(out_py_tuple, &gradient_values);
  }
  if (gradient_values.empty()) {
    MS_LOG(EXCEPTION) << "Hook fn output is not <PyObjectRef> type!";
  }
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End HookBackwardNode CallBackward";
  return gradient_tensors;
}

TensorPtrList GraphRoot::BuildFlattenSensGradient(const ValuePtrList &sens_gradient) const {
  TensorPtrList real_gradients;
  for (const auto &index : gradient_index_) {
    if (index >= sens_gradient.size()) {
      MS_LOG(EXCEPTION) << "Inputs gradient index should smaller than flatten_values size!";
    }
    const auto &gradient_tensor = sens_gradient[index]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(gradient_tensor);
    (void)real_gradients.emplace_back(gradient_tensor);
  }
  return real_gradients;
}

AutoGradCell::AutoGradCell(const ValuePtrList &input_param_values, size_t op_num_in_bprop_graph, bool grad_by_value) {
  for (size_t i = 0; i < input_param_values.size(); ++i) {
    const auto &input_param_value = input_param_values[i];
    auto func_node = std::make_shared<BackwardNode>("input" + std::to_string(i));
    auto variable = std::make_shared<Variable>(func_node, true);

    if (!input_param_value->isa<ValueSequence>()) {
      SetGradInfoForInputs(input_param_value, variable);
    } else {
      variable->set_is_need_grad(false);
    }
    variable_set_.insert(variable);
    (void)cell_inputs_.emplace_back(std::make_pair(input_param_value, variable));
  }
  device_target_ = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  func_impl_ = std::make_shared<FuncBuilder>("func_emitter", device_target_);
}

bool AutoGradCell::KPynativeOp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);

  auto &prim = grad_param->op_grad_info->op_prim;
  if (!IsPrimNeedGrad(prim) || (grad_by_value_ && !NeedGrad(grad_param->op_grad_info->input_value))) {
    MS_LOG(DEBUG) << "Prim " << prim->name() << " does not need to do op grad.";
    return true;
  }
  if (grad_param->op_grad_info->out_value->isa<tensor::Tensor>()) {
    grad_param->op_grad_info->output_size = 1;
  } else if (grad_param->op_grad_info->out_value->isa<ValueSequence>()) {
    auto seq = grad_param->op_grad_info->out_value->cast<ValueSequencePtr>();
    grad_param->op_grad_info->output_size = seq->size();
  }
  //  auto cloned_value = grad_param->op_grad_info->out_value;
  //  if (grad_param->op_grad_info->out_value->isa<ValueSequence>()) {
  //    cloned_value = ShallowCopyTensorValue(grad_param->op_grad_info->out_value);
  //    PyNativeAlgo::Common::ClearDeviceAddress(cloned_value);
  //  }
  auto flatten_inputs = FlattenArgs(grad_param->op_grad_info->input_value);
  ConstructParameterNodes(flatten_inputs);
  BackwardNodePtr fn = nullptr;
  bool is_custom_prim =
    IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook);
  if (!is_custom_prim) {
    auto handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder(prim->name());
    if (handle != nullptr) {
      fn = BuildFuncBackwardNode(prim, handle->func, flatten_inputs, grad_param->op_grad_info);
    } else {
      fn = BuildCustomBackwardNode(prim, flatten_inputs, grad_param->op_grad_info);
    }
  } else {
    fn = BuildHookBackwardNode(prim, flatten_inputs, grad_param->op_grad_info);
  }
  auto variable = std::make_shared<Variable>(fn, false);
  if (typeid(fn) == typeid(FakeBackwardNode)) {
    variable->set_is_fake_bprop(true);
    variable->set_fake_prim_name(prim->name());
  }

  variable_set_.insert(variable);
  SetGradMetaData(FlattenArgs({grad_param->op_grad_info->out_value}), variable);
  MS_LOG(DEBUG) << "End update next edge for " << variable->ToString();
  return true;
}

void AutoGradCell::UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) {
  sens_value_ = sens_out;
  auto flatten_sens = FlattenArgs({sens_out});
  ConstructParameterNodes(flatten_sens);
}

void AutoGradCell::BuildForwardLastNode(const ValuePtr &sens_gradient) {
  ValuePtrList root_gradient_value;
  if (sens_gradient == nullptr) {
    root_gradient_value = OnsLike(sens_value_);
  } else {
    root_gradient_value = FlattenArgs({sens_gradient});
  }
  auto root = std::make_shared<GraphRoot>("GraphRoot");
  auto flatten_args = FlattenArgs({sens_value_});
  root->UpdateNextEdges(flatten_args);
  root_gradients_ = root->BuildFlattenSensGradient(root_gradient_value);
  auto sens_variable = std::make_shared<Variable>(root, false);
  if (root_gradients_.empty()) {
    sens_variable->set_is_need_grad(false);
  }
  variable_set_.insert(sens_variable);
  last_variable_ = sens_variable;
}

void AutoGradCell::BackPropagate() {
  MS_LOG(DEBUG) << "Begin BackPropagate";
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
  const auto &root_fn = (*last_node_reverse_iter)->fn();
  mindspore::HashMap<BackwardNode *, TensorPtrList> input_buffer;
  input_buffer.insert({root_fn.get(), root_gradients_});
  for (auto iter = last_node_reverse_iter; iter != variable_set_.rend(); ++iter) {
    const auto &variable = *iter;
    const auto &fn = variable->fn();
    MS_LOG(DEBUG) << "Begin caculate op: " << fn->name() << " gradients!";
    if (!variable->is_need_propagate() || !variable->is_need_grad()) {
      MS_LOG(DEBUG) << "No need grad, variable is: " << variable->ToString();
      continue;
    }
    if (static_cast<bool>(MS_UNLIKELY(variable->is_fake_bprop()))) {
      MS_LOG(EXCEPTION) << "Illegal primitive " << variable->fake_prim_name() << "'s bprop not defined";
    }

    if (input_buffer.find(fn.get()) == input_buffer.end()) {
      MS_LOG(EXCEPTION) << "Fn not has gradient";
    }
    const TensorPtrList &gradient_in = input_buffer[fn.get()];
    MS_LOG(DEBUG) << "Begin print gradient in: ";
    MS_LOG(DEBUG) << PrintGradients(gradient_in);
    TensorPtrList gradient_out = fn->CallBackward(gradient_in);
    MS_LOG(DEBUG) << "Begin print gradient out: ";
    MS_LOG(DEBUG) << PrintGradients(gradient_out);
    if (gradient_out.size() != fn->next_edges().size()) {
      MS_LOG(EXCEPTION) << "Fn gradient size should be same as next edges size";
    }
    for (size_t i = 0; i < fn->next_edges().size(); ++i) {
      const auto &next_edge = fn->next_edges()[i];
      const auto &last_variable = next_edge.variable;
      const auto &last_fn = last_variable->fn();
      const auto &last_gradient = gradient_out[i];
      // If last_gradient is nullptr, It represent that this tensor grad is zeros.
      if (last_gradient == nullptr) {
        MS_LOG(DEBUG) << last_variable->ToString() << ", its gradient is nullptr!";
        continue;
      }
      if (input_buffer.find(last_fn.get()) != input_buffer.end()) {
        auto &tmp_grads = input_buffer[last_fn.get()];
        input_buffer[last_fn.get()] = Add(tmp_grads, last_gradient, next_edge.input_index);
      } else {
        input_buffer[last_fn.get()] = SetAlign(last_gradient, last_fn->output_size(), next_edge.input_index);
      }
      last_variable->set_is_need_propagate(true);
    }
    if (variable->is_leaf()) {
      auto grad_tensor = input_buffer[fn.get()];
      if (grad_tensor.empty() || grad_tensor[0] == nullptr) {
        MS_LOG(EXCEPTION) << variable->ToString() << ", "
                          << (grad_tensor.empty() ? "grad is empty" : "grad is nullptr");
      }
      variable->set_grad(grad_tensor[0]);
    }
    input_buffer.erase(fn.get());
  }
  MS_LOG(DEBUG) << "End BackPropagate";
}

OrderedSet<VariablePtr>::reverse_iterator AutoGradCell::GetLastNodeReverseIter() {
  for (auto iter = variable_set_.rbegin(); iter != variable_set_.rend(); ++iter) {
    if (*iter == last_variable_) {
      last_variable_->set_is_need_propagate(true);
      return iter;
    }
  }
  return variable_set_.rend();
}

void AutoGradCell::ConstructParameterNodes(const ValuePtrList &inputs) {
  for (const auto &value : inputs) {
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      auto auto_grad_meta_data = tensor->auto_grad_meta_data();
      MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
      if (auto_grad_meta_data->variable() != nullptr) {
        continue;
      }
      if (auto_grad_meta_data->input_type() == InputType::kParameter) {
        auto fn = std::make_shared<BackwardNode>("parameter");
        auto variable = std::make_shared<Variable>(fn, true);
        auto_grad_meta_data->set_variable(variable);
        variable_set_.insert(variable);
      }
    }
  }
}

BackwardNodePtr AutoGradCell::BuildFuncBackwardNode(const PrimitivePtr &prim,
                                                    const expander::bprop::BpropBuilderFunc &func,
                                                    const ValuePtrList &flatten_inputs,
                                                    const OpGradInfoPtr &op_grad_info) {
  auto fn = std::make_shared<FuncBackwardNode>(prim->name(), func, prim->attrs(), op_grad_info->input_value,
                                               op_grad_info->out_value, op_grad_info->output_size,
                                               op_grad_info->input_value_grad_type);
  fn->set_attrs(prim->attrs());
  fn->UpdateNextEdges(flatten_inputs);
  return fn;
}

BackwardNodePtr AutoGradCell::BuildCustomBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                      const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Try build custom bprop: " << prim->name();
  {
    py::gil_scoped_acquire gil;
    auto prim_py = prim->cast<PrimitivePyPtr>();
    if (prim_py == nullptr) {
      MS_LOG(DEBUG) << "Prim is not PrimitivePy, can not find python bprop";
      return std::make_shared<FakeBackwardNode>(prim->name());
    }
    py::function fn = prim_py->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
    if (!fn || py::isinstance<py::none>(fn)) {
      MS_LOG(INFO) << "Can not find bprop function for " << prim->name() << ". fn: " << py::str(fn);
      return std::make_shared<FakeBackwardNode>(prim->name());
    }
    (void)prim_py->AddBackwardHookFn(0, fn);
    prim_py->AddAttr("custom_op_bprop", MakeValue(true));
  }
  return BuildHookBackwardNode(prim, flatten_inputs, op_grad_info);
}

BackwardNodePtr AutoGradCell::BuildHookBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                    const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim_py);
  prim_py->AddBpropCutPrim(bprop_cut);
  if (prim->HasAttr("cell_id")) {
    auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
    if (!cell_id.empty()) {
      (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
      (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
    }
  }
  // Only custom op need add this attr, hook function not need.
  if (prim->HasAttr("custom_op_bprop")) {
    (void)bprop_cut->AddAttr("custom_op_bprop", MakeValue(true));
  }
  (void)bprop_cut->AddAttr("custom_op_name", MakeValue(prim->name()));

  VectorRef args = GeneratePythonArgs(op_grad_info->input_value, op_grad_info->out_value);
  auto fn = std::make_shared<HookBackwardNode>(prim->name(), bprop_cut, args, op_grad_info->output_size);
  fn->UpdateNextEdges(flatten_inputs);
  return fn;
}

ValuePtr AutoGradCell::GetGrads(const tensor::TensorPtrList &weights, const std::vector<size_t> &grad_position,
                                const GradAttr &grad_attr) {
  auto inputs_grad_ret = GetInputGrads(grad_attr.grad_all_inputs, grad_attr.get_by_position, grad_position);
  auto weights_grad_ret = GetWeightGrads(grad_attr.grad_weights, weights, grad_attr.weight_param_is_tuple);
  // Gradients wrt inputs and weights.
  if (inputs_grad_ret != nullptr && weights_grad_ret != nullptr) {
    if (IsOutputBothEmpty(inputs_grad_ret, weights_grad_ret)) {
      return GenerateEmptyTupleValue();
    } else {
      ValuePtrList gradients{inputs_grad_ret, weights_grad_ret};
      return std::make_shared<ValueTuple>(gradients);
    }
  }
  // Gradients wrt inputs.
  if (inputs_grad_ret != nullptr) {
    return inputs_grad_ret;
  }
  // Gradients wrt weights.
  if (weights_grad_ret != nullptr) {
    return weights_grad_ret;
  }
  // grad_all_inputs, grad_weights and get_by_position are all false.
  if (cell_inputs_.empty()) {
    // If no input nodes, return empty tuple.
    ValuePtrList empty_tuple;
    return std::make_shared<ValueTuple>(empty_tuple);
  } else {
    // If there are input nodes, return gradient of first input node.
    // Tuple, List, scalar will be ignore
    if (cell_inputs_[kIndex0].second->is_need_grad()) {
      auto grad = BuildSpecialGrad(cell_inputs_[kIndex0].first, cell_inputs_[kIndex0].second->grad(), func_impl_);
      return grad;
    } else {
      MS_LOG(DEBUG) << "Get first input node is not tensor " << cell_inputs_[0].first->ToString();
      return kNull;
    }
  }
}

ValuePtr AutoGradCell::GetInputGrads(bool grad_all_inputs, bool get_by_position,
                                     const std::vector<size_t> &grad_position) {
  std::vector<size_t> grad_pos_list;
  if (get_by_position) {
    grad_pos_list = grad_position;
  } else if (grad_all_inputs) {
    grad_pos_list.resize(cell_inputs_.size());
    iota(grad_pos_list.begin(), grad_pos_list.end(), 0);
  } else {
    return nullptr;
  }
  ValuePtrList input_grads;
  input_grads.reserve(cell_inputs_.size());
  if (!cell_inputs_.empty()) {
    for (size_t index : grad_pos_list) {
      if (index >= cell_inputs_.size()) {
        MS_LOG(EXCEPTION) << "Position index " << index << " is exceed input size.";
      }
      // Tuple, List, scalar will be ignore
      if (!cell_inputs_[index].second->is_need_grad()) {
        MS_LOG(DEBUG) << cell_inputs_[index].first->ToString() << "is not need grad!";
        continue;
      }
      ValuePtr real_dout = BuildSpecialGrad(cell_inputs_[index].first, cell_inputs_[index].second->grad(), func_impl_);
      (void)input_grads.emplace_back(real_dout);
    }
    if (get_by_position && input_grads.size() == kSizeOne) {
      return input_grads[kIndex0];
    }
  }
  return std::make_shared<ValueTuple>(input_grads);
}

ValuePtr AutoGradCell::GetWeightGrads(bool grad_weights, const tensor::TensorPtrList &weights,
                                      bool weight_param_is_tuple) {
  // No need to return gradient of weights.
  if (!grad_weights) {
    return nullptr;
  }
  if (weight_param_is_tuple) {
    ValuePtrList weight_grads;
    weight_grads.reserve(weights.size());
    for (const auto &weight : weights) {
      (void)weight_grads.emplace_back(GetWeightGrad(weight));
    }
    return std::make_shared<ValueTuple>(weight_grads);
  } else {
    return GetWeightGrad(weights[0]);
  }
}

ValuePtr AutoGradCell::GetWeightGrad(const tensor::TensorPtr &weight) {
  MS_EXCEPTION_IF_NULL(weight);
  auto auto_grad_meta_data = weight->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    return func_impl_->Zeros(weight);
  }
  auto variable = auto_grad_meta_data->variable();
  if (variable != nullptr && variable->is_need_grad()) {
    // If weight used in the forward network, but requires_grad is false, return zero like.
    if (variable->grad() == nullptr || (weight->param_info() != nullptr && !weight->param_info()->requires_grad())) {
      MS_LOG(INFO) << "weight participate in forward calculation, but requires_grad is false";
      return func_impl_->Zeros(weight);
    }
    auto weight_grad = variable->grad();
    return weight_grad;
  }
  MS_LOG(INFO) << "parameter does not need grad, tensor: " << PyNativeAlgo::Common::GetIdByValue(weight);
  return func_impl_->Zeros(weight);
}

void AutoGradCell::ClearGrads(const TensorPtrList &weights) {
  // Clear input grads.
  for (const auto &input : cell_inputs_) {
    input.second->set_grad(nullptr);
  }
  cell_inputs_.clear();
  // Clear weights grad info
  for (const auto &weight : weights) {
    weight->set_auto_grad_meta_data(nullptr);
  }
}

ValuePtrList AutoGradCell::OnsLike(const ValuePtr &sens) {
  auto flatten_values = FlattenArgs({sens});
  ValuePtrList flatten_sens;
  flatten_sens.reserve(flatten_values.size());
  for (const auto &flatten_value : flatten_values) {
    if (flatten_value->isa<tensor::Tensor>()) {
      (void)flatten_sens.emplace_back(func_impl_->Ones(flatten_value));
    } else {
      (void)flatten_sens.emplace_back();
    }
  }
  return flatten_sens;
}

void AutoGradCell::CheckSensShapeAndType(const ValuePtr &sens_gradient) {
  if (sens_gradient == nullptr) {
    return;
  }
  const auto sens_gradient_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(sens_gradient->ToAbstract());
  const auto out_abs = PyNativeAlgo::Common::SetAbstractValueToAnyValue(sens_value_->ToAbstract());
  const auto &sens_gradient_shape = sens_gradient_abs->BuildShape()->ToString();
  const auto &out_shape = out_abs->BuildShape()->ToString();
  if (sens_gradient_shape != "()" && out_shape != "()") {
    if (sens_gradient_shape != out_shape) {
      // Sens shape in ir graph is determined by graph output, so it can be dynamic shape; But input shape is
      // determined by user input, which could not be dynamic shape.
      if (!sens_gradient_abs->BuildShape()->IsDynamic()) {
        MS_EXCEPTION(ValueError) << "The shape should be " << out_shape << ", but got " << sens_gradient_shape << ", "
                                 << ", sens gradient abs " << sens_gradient_abs->ToString() << ", out abs"
                                 << out_abs->ToString();
      }
    }
    const auto &sens_gradient_dtype = sens_gradient_abs->BuildType()->ToString();
    const auto &out_dtype = out_abs->BuildType()->ToString();
    if (sens_gradient_dtype != out_dtype) {
      MS_EXCEPTION(TypeError) << "The dtype should be " << out_dtype << ", but got " << sens_gradient_dtype << ", "
                              << ", sens gradient abs " << sens_gradient_abs->ToString() << ", out abs"
                              << out_abs->ToString();
    }
  }
}

ValuePtr AutoGradCell::Finish(const TensorPtrList &weights, const std::vector<size_t> &grad_position,
                              const GradAttr &grad_attr, const ValuePtr &sens) {
  CheckSensShapeAndType(sens);
  BuildForwardLastNode(sens);
  if (last_variable_->is_need_grad()) {
    BackPropagate();
  }
  ValuePtr gradients = GetGrads(weights, grad_position, grad_attr);
  ClearGrads(weights);
  return gradients;
}
void ClearPyNativeAutoGradStaticRes() {}
}  // namespace mindspore::pynative::autograd