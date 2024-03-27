/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/auto_grad/native_backward_function.h"
#include <algorithm>
#include <vector>
#include "include/common/expander/core/node.h"
#include "pipeline/pynative/pynative_utils.h"

namespace mindspore {
namespace pijit {
namespace grad {
using FuncBuilder = pynative::autograd::FuncBuilder;

NativeBackwardFuncPtr NativeBackwardFunc::GetInstance(const PrimitivePtr &prim) {
  if (prim == nullptr) {
    return nullptr;
  }
  const auto handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder(prim->name());
  if (handle == nullptr) {
    return nullptr;
  }
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  const FuncBuilderPtr &ir_builder = std::make_shared<FuncBuilder>(prim->name(), device_target);
  return std::make_shared<NativeBackwardFunc>(prim, ir_builder, handle);
}

ValuePtrList NativeBackwardFunc::Run(const ValuePtrList &inputs, const ValuePtr &out, const ValuePtr &dout) {
  if (handle_ == nullptr) {
    return ValuePtrList(GetGradientIndexes().size(), kNone);
  }
  mindspore::HashMap<std::string, ValuePtr> attrs = prim_->attrs();
  expander::NodePtrList node_inputs = PreProcess(inputs, out, dout);
  ir_builder_->SetInputs(GetName(), &node_inputs, &attrs);
  const std::vector<expander::NodePtr> cal_grads_node = handle_->func(ir_builder_.get());
  ValuePtrList cal_grads_values;
  cal_grads_values.reserve(cal_grads_node.size());
  // Binary op grad result may be nulllptr, we need convert to kNone.
  (void)std::transform(cal_grads_node.begin(), cal_grads_node.end(), std::back_inserter(cal_grads_values),
                       [](const expander::NodePtr &node) -> ValuePtr {
                         if (node == nullptr) {
                           return kNone;
                         }
                         return node->Value();
                       });
  return PostProcess(pynative::PyNativeAlgo::DataConvert::FlattenTensorSeqInValueSeq(cal_grads_values));
}

ValuePtrList NativeBackwardFunc::PostProcess(const ValuePtrList &gradient_value) {
  ValuePtrList grad_values;
  (void)std::transform(GetGradientIndexes().begin(), GetGradientIndexes().end(), std::back_inserter(grad_values),
                       [&gradient_value](const auto &index) -> ValuePtr { return gradient_value[index]; });
  return grad_values;
}

InputType GetInputType(const ValuePtr &input) {
  if (input->template isa<Parameter>()) {
    return InputType::kParameter;
  }
  if (!input->template isa<tensor::Tensor>()) {
    return InputType::kConstant;
  }
  return InputType::kInput;
}

expander::NodePtrList NativeBackwardFunc::PreProcess(const ValuePtrList &inputs, const ValuePtr &out,
                                                     const ValuePtr &dout) const {
  expander::NodePtrList node_inputs;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(node_inputs), [this](const auto &input) {
    ValuePtr value = input;
    if (input->template isa<stub::TensorNode>()) {
      value = input->template cast<stub::StubNodePtr>()->WaitValue();
    }
    return ir_builder_->NewFuncNode(value, value->ToAbstract(), GetInputType(value));
  });
  std::for_each(GetGradientIndexes().begin(), GetGradientIndexes().end(), [&node_inputs](const auto &index) {
    std::dynamic_pointer_cast<expander::FuncNode>(node_inputs[index])->set_need_compute_grad_out(true);
  });
  (void)node_inputs.emplace_back(ir_builder_->NewFuncNode(out, out->ToAbstract(), InputType::kOpOutput));
  (void)node_inputs.emplace_back(ir_builder_->NewFuncNode(dout, dout->ToAbstract(), InputType::kOpOutput));
  return node_inputs;
}
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
