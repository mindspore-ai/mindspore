/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/convert_const_scalar_to_tensor.h"
#include <memory>
#include <utility>
#include "include/common/utils/convert_utils.h"
#include "backend/common/optimizer/helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_graph.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr CreateTensorInput(const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (!input_node->isa<ValueNode>()) {
    return nullptr;
  }
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<Scalar>()) {
    return nullptr;
  }
  tensor::TensorPtr tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  if (tensor_ptr == nullptr) {
    MS_LOG(WARNING) << "Create tensor of" << input_node->DebugString() << "failed";
    return nullptr;
  }
  auto tensor_input = std::make_shared<ValueNode>(tensor_ptr);
  MS_EXCEPTION_IF_NULL(tensor_input);
  tensor_input->set_abstract(tensor_ptr->ToAbstract());
  if (kernel_graph != nullptr) {
    tensor_input = kernel_graph->NewValueNode(tensor_input);
    kernel_graph->AddValueNodeToGraph(tensor_input);
  } else {
    tensor_input = MakeValueNode(tensor_input);
  }
  tensor_input->set_scope(input_node->scope());
  return tensor_input;
}
}  // namespace

const AnfNodePtr ConvertConstScalarToTensor::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  if (node == nullptr || func_graph == nullptr || common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    return nullptr;
  }
  // input is scalar, and link to graph return
  if (node->isa<ValueNode>() && node == func_graph->output()) {
    return CreateTensorInput(func_graph->cast<KernelGraphPtr>(), node);
  }
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  bool input_changed = false;
  for (size_t i = 0; i < cnode->inputs().size(); ++i) {
    if (!AnfAlgo::IsScalarConvertToTensor(cnode->inputs()[i], cnode)) {
      continue;
    }
    auto new_input = CreateTensorInput(func_graph->cast<KernelGraphPtr>(), cnode->inputs()[i]);
    if (new_input != nullptr) {
      cnode->set_input(i, new_input);
      input_changed = true;
    }
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph == nullptr || !input_changed) {
    return nullptr;
  }
  return NewCNode(cnode, kernel_graph);
}
}  // namespace opt
}  // namespace mindspore
