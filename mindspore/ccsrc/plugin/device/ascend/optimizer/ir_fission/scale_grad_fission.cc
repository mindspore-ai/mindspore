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
#include "plugin/device/ascend/optimizer/ir_fission/scale_grad_fission.h"
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/backend/kernel_info.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kScaleGradInputSize = 3;
constexpr double kFloatMinimal = 1e-7;

AnfNodePtr CreateNodeOfBinaryOp(const FuncGraphPtr &graph, const string &op_name, const AnfNodePtr &node1,
                                const AnfNodePtr &node2) {
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(op_name)), node1, node2};
  return CreateNodeBase(graph, new_node_inputs, node1);
}

AnfNodePtr CreateCastNode(const FuncGraphPtr &graph, const AnfNodePtr &node, const TypeId &dst_type_id) {
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(kCastOpName)), node};
  auto new_node = graph->NewCNode(new_node_inputs);
  MS_EXCEPTION_IF_NULL(new_node);

  new_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  new_node->set_scope(node->scope());
  new_node->set_abstract(node->abstract());

  auto types = {dst_type_id};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(node, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, new_node.get());
  return new_node;
}
}  // namespace

const BaseRef ScaleGradFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimScaleGrad, Xs});
}

const AnfNodePtr ScaleGradFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto scale_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(scale_grad_cnode);
  const auto ori_inputs = scale_grad_cnode->inputs();
  auto input_size = ori_inputs.size();
  if (input_size < kScaleGradInputSize) {
    MS_LOG(EXCEPTION) << "ScaleGrad inputs size is less than 3!";
  }

  auto scale_node = ori_inputs[input_size - 1];
  MS_EXCEPTION_IF_NULL(scale_node);
  auto scale_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, input_size - 2);
  auto scale_value_node = scale_node->cast<ValueNodePtr>();
  if (scale_value_node != nullptr) {
    auto tensor = GetValue<tensor::TensorPtr>(scale_value_node->value());
    float scale_value = 0.0;
    if (scale_type_id == kNumberTypeFloat32) {
      scale_value = *(static_cast<float *>(tensor->data_c()));
    } else if (scale_type_id == kNumberTypeFloat16) {
      scale_value = static_cast<float>(*(static_cast<float16 *>(tensor->data_c())));
    } else {
      MS_LOG(EXCEPTION) << "Scale value's type is error, must be float32 or float16, but get "
                        << TypeIdToString(scale_type_id);
    }

    if (std::fabs(static_cast<double>(scale_value) - 1.0) < kFloatMinimal) {
      std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
      (void)make_tuple_inputs.insert(make_tuple_inputs.cend(), ori_inputs.begin() + 1, ori_inputs.end() - 1);
      auto make_tuple = graph->NewCNode(make_tuple_inputs);
      MS_EXCEPTION_IF_NULL(make_tuple);
      return make_tuple;
    }
  }

  AnfNodePtr cast = nullptr;
  if (scale_type_id == kNumberTypeFloat32) {
    cast = CreateCastNode(graph, scale_node, kNumberTypeFloat16);
  } else {
    cast = CreateCastNode(graph, scale_node, kNumberTypeFloat32);
  }

  std::vector<AnfNodePtr> outputs;
  for (size_t index = 1; index < input_size - 1; index++) {
    auto input = ori_inputs[index];
    auto input_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, index - 1);
    if (input_type_id == scale_type_id) {
      auto out = CreateNodeOfBinaryOp(graph, kMulOpName, input, scale_node);
      outputs.push_back(out);
    } else {
      auto out = CreateNodeOfBinaryOp(graph, kMulOpName, input, cast);
      outputs.push_back(out);
    }
  }

  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  (void)make_tuple_inputs.insert(make_tuple_inputs.end(), outputs.cbegin(), outputs.cend());
  auto make_tuple = graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
