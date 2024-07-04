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

#include "pipeline/pynative/grad/variable.h"
#include <memory>
#include "pipeline/pynative/pynative_utils.h"

namespace mindspore::pynative::autograd {
void BackwardNode::UpdateNextEdges(const ValuePtrList &inputs) {
  MS_LOG(DEBUG) << "Get input size " << inputs.size();
  next_edges_.reserve(inputs.size());
  gradient_index_.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &value = inputs[i];
    if (value->isa<tensor::BaseTensor>()) {
      const auto &tensor = value->cast<tensor::BaseTensorPtr>();
      auto auto_grad_meta_data = tensor->auto_grad_meta_data();
      // Get scalar tensor
      if (auto_grad_meta_data == nullptr) {
        continue;
      }
      auto variable = auto_grad_meta_data->variable();
      if (variable == nullptr || !variable->is_need_grad()) {
        continue;
      }
      MS_LOG(DEBUG) << "Add next edge for tensor " << tensor->id();
      (void)next_edges_.emplace_back(variable, auto_grad_meta_data->output_index());
      (void)gradient_index_.emplace_back(i);
    }
    // to do sparse tensor.
  }
}

ValuePtrList BackwardNode::PostProcess(const ValuePtrList &gradient_value) {
  ValuePtrList gradients;
  ValuePtrList flatten_values = PyNativeAlgo::DataConvert::FlattenTensorSeqInValueSeq(gradient_value, false);
  gradients.reserve(flatten_values.size());
  for (const auto index : gradient_index_) {
    if (MS_UNLIKELY(index >= flatten_values.size())) {
      MS_LOG(EXCEPTION) << "Inputs gradient index should smaller than flatten_values size!";
    }
    const auto &gradient_tensor = flatten_values[index];
    (void)gradients.emplace_back(gradient_tensor);
  }
  return gradients;
}

ValuePtrList BackwardNode::LazeUpdateZeroGradient(const ValuePtrList &dout, FuncBuilder *func_builder,
                                                  const ValuePtr &output) {
  if (dout.size() == kSizeOne) {
    return dout;
  }
  ValuePtrList outputs;
  PyNativeAlgo::DataConvert::FlattenValueSeqArg(output, true, false, &outputs);
  if (outputs.size() != dout.size()) {
    MS_LOG(EXCEPTION) << "Gradients size should be same as output size! But got output size: " << outputs.size()
                      << ", gradients size: " << dout.size();
  }
  ValuePtrList real_dout(dout.size());
  for (size_t i = 0; i < dout.size(); ++i) {
    if (dout[i]->isa<None>()) {
      MS_LOG(DEBUG) << "Op " << name() << " has multi outputs, and exist null dout, now do emit zeros";
      auto zero_value =
        PyNativeAlgo::AutoGrad::BuildSpecialValueGrad(outputs[i], nullptr, func_builder, SpecialType::kZerosLikeType);
      MS_EXCEPTION_IF_NULL(zero_value);
      real_dout[i] = zero_value;
    } else {
      real_dout[i] = dout[i];
    }
  }
  return real_dout;
}

std::string FuncVariable::ToString() const {
  std::ostringstream buf;
  buf << "Variable name: " << func_node()->name() << ", is_need_grad: " << is_need_grad()
      << ", is_need_propagate: " << is_need_propagate() << " is_leaf: " << is_leaf() << "\n";
  for (size_t i = 0; i < func_node()->next_edges().size(); ++i) {
    auto last_variable = func_node()->next_edges()[i].variable;
    auto index = func_node()->next_edges()[i].input_index;
    MS_EXCEPTION_IF_NULL(last_variable->func_node());
    buf << "Last edge: " << i << ", variable name: " << last_variable->func_node()->name()
        << ", output index: " << index << "\n";
  }
  return buf.str();
}

std::string IrVariable::ToString() const {
  std::ostringstream buf;
  buf << "Variable id: " << PyNativeAlgo::Common::GetIdByValue(out_value()) << ", is_need_grad: " << is_need_grad()
      << ", is_need_propagate: " << is_need_propagate() << ", is_leaf: " << is_leaf();
  for (size_t i = 0; i < ir_function_node()->next_edges().size(); ++i) {
    auto last_variable = ir_function_node()->next_edges()[i].first;
    auto din = ir_function_node()->next_edges()[i].second;
    buf << ", next edge variable id: " << PyNativeAlgo::Common::GetIdByValue(last_variable->out_value())
        << " din: " << din->DebugString();
  }
  return buf.str();
}

AnfNodePtr IrVariable::RealDout() {
  if (static_cast<bool>(MS_UNLIKELY(PyNativeAlgo::AutoGrad::IsZerosLikeNode(ir_function_node()->accumulate_dout())))) {
    ir_function_node()->set_accumulate_dout(PyNativeAlgo::AutoGrad::BuildSpecialNode(
      ir_function_node()->tape(), out_value(), ir_function_node()->accumulate_dout()->abstract(),
      SpecialType::kZerosLikeType));
  }
  const auto &accumulate_dout = ir_function_node()->accumulate_dout();
  const auto &dout_abs = accumulate_dout->abstract();
  MS_EXCEPTION_IF_NULL(dout_abs);
  // For input, if it is a sparsetensor, we need return a sparsetensor.
  if (out_value()->isa<tensor::BaseTensor>() || dout_abs->isa<abstract::AbstractSparseTensor>()) {
    return accumulate_dout;
  } else if (out_value()->isa<tensor::MetaSparseTensor>()) {
    return PyNativeAlgo::AutoGrad::BuildSparseTensorNode(ir_function_node()->tape(), out_value(), accumulate_dout);
  }
  return accumulate_dout;
}
}  // namespace mindspore::pynative::autograd
