/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "pre_activate/pass/convert_const_input_to_tensor_input.h"

#include <vector>
#include <memory>

#include "utils/graph_utils.h"
#include "session/anf_runtime_algorithm.h"
#include "session/kernel_graph.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kType32Len = 4;
template <typename T>
tensor::TensorPtr CreateTensorWithValueTuple(const ValueTuplePtr &value_tuple_ptr, const TypePtr &type_ptr,
                                             size_t data_length) {
  MS_EXCEPTION_IF_NULL(value_tuple_ptr);
  MS_EXCEPTION_IF_NULL(type_ptr);
  std::vector<T> values;
  for (const auto &v : value_tuple_ptr->value()) {
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<Scalar>()) {
      ScalarPtr scalar = v->cast<ScalarPtr>();
      values.push_back(GetValue<T>(scalar));
    } else {
      MS_LOG(WARNING) << "The value " << v << "of tuple is not a scalar";
      return nullptr;
    }
  }
  std::vector<int> tensor_shape = {SizeToInt(values.size())};
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_ptr->type_id(), tensor_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, type_ptr};
  tensor->set_device_info(device_info);
  auto data_ptr = tensor->data_c(true);
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto elem_num = values.size() * data_length;
  auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(tensor->data().nbytes()), values.data(), elem_num);
  if (ret_code != 0) {
    MS_LOG(EXCEPTION) << "Failed to copy data into Tensor.";
  }
  return tensor;
}

tensor::TensorPtr CreateTupleTensor(const ValueTuplePtr &value_tuple) {
  MS_EXCEPTION_IF_NULL(value_tuple);
  tensor::TensorPtr tensor = nullptr;
  ValuePtr v = *(value_tuple->value().begin());
  MS_EXCEPTION_IF_NULL(v);
  // Currently we only deal with the scalar tuple
  if (!v->isa<Scalar>()) {
    MS_LOG(WARNING) << "The value " << v << "of tuple is not a scalar";
    return nullptr;
  }
  ScalarPtr scalar = v->cast<ScalarPtr>();
  MS_EXCEPTION_IF_NULL(scalar);
  if (scalar->isa<IntergerImm>()) {
    tensor = CreateTensorWithValueTuple<int>(value_tuple, kInt32, kType32Len);
  } else if (scalar->isa<FloatImm>()) {
    tensor = CreateTensorWithValueTuple<float>(value_tuple, kFloat32, kType32Len);
  } else {
    auto type = scalar->type();
    auto type_str = (type == nullptr) ? "nullptr" : type->ToString();
    MS_LOG(ERROR) << "Invalid scalar type: " << type_str;
    return nullptr;
  }
  return tensor;
}

AnfNodePtr CreateTensorInput(const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  tensor::TensorPtr tensor_ptr = nullptr;
  if (value->isa<Scalar>()) {
    tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  } else if (value->isa<ValueTuple>()) {
    tensor_ptr = CreateTupleTensor(value->cast<ValueTuplePtr>());
  } else {
    MS_LOG(EXCEPTION) << "The value should be a scalar or value tuple";
  }
  if (tensor_ptr == nullptr) {
    MS_LOG(WARNING) << "Create tensor failed";
    return nullptr;
  }
  auto tensor_input = std::make_shared<ValueNode>(tensor_ptr);
  MS_EXCEPTION_IF_NULL(tensor_input);
  tensor_input->set_abstract(tensor_ptr->ToAbstract());
  if (kernel_graph != nullptr) {
    tensor_input = kernel_graph->NewValueNode(tensor_input);
    kernel_graph->AddValueNodeToGraph(tensor_input);
  }
  tensor_input->set_scope(input_node->scope());
  return tensor_input;
}

AnfNodePtr ConstInputToTensorInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> new_inputs;
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  auto inputs = cnode->inputs();
  new_inputs.push_back(inputs[0]);
  bool need_update = false;
  // the first input is primitive node which is not the real input
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    if (IsValueNode<Scalar>(input_node) || IsValueNode<ValueTuple>(input_node)) {
      auto tensor_input = CreateTensorInput(kernel_graph, input_node);
      if (tensor_input == nullptr) {
        new_inputs.push_back(input_node);
        continue;
      }
      new_inputs.push_back(tensor_input);
      need_update = true;
    } else {
      new_inputs.push_back(input_node);
    }
  }
  if (need_update) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto new_cnode = func_graph->NewCNode(new_inputs);
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode->abstract());
    new_cnode->set_scope(cnode->scope());
    AnfAlgo::CopyNodeAttrs(cnode, new_cnode);
    return new_cnode;
  }
  return nullptr;
}
}  // namespace

const AnfNodePtr ConvertConstInputToTensorInput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  if (node == nullptr || func_graph == nullptr || !AnfAlgo::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  return ConstInputToTensorInput(func_graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
