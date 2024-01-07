//
// Created by luochao on 2024/1/19.
//

#include "pipeline/pynative/grad/variable.h"
#include "pipeline/pynative/grad/function/function_utils.h"

namespace mindspore::pynative::autograd {
void BackwardNode::UpdateNextEdges(const ValuePtrList &inputs) {
  next_edges_.reserve(inputs.size());
  gradient_index_.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &value = inputs[i];
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      auto auto_grad_meta_data = tensor->auto_grad_meta_data();
      MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
      auto variable = auto_grad_meta_data->variable();
      if (variable == nullptr || (variable != nullptr && !variable->is_need_grad())) {
        continue;
      }
      (void)next_edges_.emplace_back(Edge(variable, auto_grad_meta_data->output_index()));
      (void)gradient_index_.emplace_back(i);
    }
    // to do sparse tensor.
  }
}

TensorPtrList BackwardNode::PostProcess(const ValuePtrList &gradient_value) {
  TensorPtrList gradients;
  ValuePtrList flatten_values = FlattenArgs(gradient_value);
  gradients.reserve(flatten_values.size());
  for (const auto index : gradient_index_) {
    if (index >= flatten_values.size()) {
      MS_LOG(EXCEPTION) << "Inputs gradient index should smaller than flatten_values size!";
    }
    auto val = flatten_values[index];
    auto gradient_tensor = flatten_values[index]->cast<tensor::TensorPtr>();
    (void)gradients.emplace_back(gradient_tensor);
  }
  return gradients;
}

std::string Variable::ToString() {
  std::ostringstream buf;
  buf << "Variable name: " << fn()->name() << ", is_need_grad: " << is_need_grad_
      << ", is_need_propagate: " << is_need_propagate_ << " is_leaf: " << is_leaf_ << "\n";
  for (size_t i = 0; i < fn()->next_edges().size(); ++i) {
    auto last_variable = fn()->next_edges()[i].variable;
    auto index = fn()->next_edges()[i].input_index;
    buf << "Last edge: " << i << ", variable name: " << last_variable->fn()->name() << "output index: " << index
        << "\n";
  }
  return buf.str();
}
}  // namespace mindspore::pynative::autograd