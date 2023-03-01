
/**
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
#include "plugin/device/ascend/optimizer/ir_fusion/scatter_nd_fusion.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kScatterNdInputTensorNum = 3;

template <typename T>
void GetShapeValue(const tensor::TensorPtr &tensor, std::vector<T> *new_value) {
  auto *data = static_cast<T *>(tensor->data_c());
  for (size_t i = 0; i < tensor->DataSize(); i++) {
    auto v = *(data + i);
    (void)new_value->emplace_back(v);
  }
}
}  // namespace
const BaseRef ScatterNdFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimScatterNd, Xs});
}

const AnfNodePtr ScatterNdFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto scatter_nd = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(scatter_nd);
  auto input_num = common::AnfAlgo::GetInputTensorNum(scatter_nd);
  if (input_num != kScatterNdInputTensorNum) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << input_num
                      << "] of node " + scatter_nd->DebugString() + " is not equal to " << kScatterNdInputTensorNum
                      << ". " << trace::DumpSourceLines(node);
  }
  auto input_node2 = scatter_nd->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_node2);
  auto input_type2 = common::AnfAlgo::GetOutputInferDataType(input_node2, kIndex0);
  if (input_type2 == kNumberTypeInt8 || input_type2 == kNumberTypeUInt8) {
    std::vector<AnfNodePtr> scatter_nd_d_inputs = {NewValueNode(std::make_shared<Primitive>(kScatterNdDOpName))};
    auto input_node1 = scatter_nd->input(kIndex1);
    MS_EXCEPTION_IF_NULL(input_node1);
    scatter_nd_d_inputs.push_back(input_node1);
    scatter_nd_d_inputs.push_back(input_node2);
    auto scatter_nd_d = NewCNode(scatter_nd_d_inputs, graph);
    MS_EXCEPTION_IF_NULL(scatter_nd_d);
    scatter_nd_d->set_scope(scatter_nd->scope());
    scatter_nd_d->set_abstract(scatter_nd->abstract());

    auto shape_node = scatter_nd->input(kIndex3);
    MS_EXCEPTION_IF_NULL(shape_node);
    auto shape_value_node = shape_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(shape_value_node);
    auto shape_value = shape_value_node->value();
    MS_EXCEPTION_IF_NULL(shape_value);

    auto tensor = shape_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_type = tensor->data_type_c();
    switch (tensor_type) {
      case kNumberTypeInt32: {
        std::vector<int32_t> shape;
        GetShapeValue(tensor, &shape);
        common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(shape), scatter_nd_d);
        break;
      }
      case kNumberTypeInt64: {
        std::vector<int64_t> shape;
        GetShapeValue(tensor, &shape);
        common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(shape), scatter_nd_d);
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "In the process of converting ScatterNd to ScatterNdD, "
                          << "the expected type of Shape(the 3rdinput) is int32 or int64, "
                          << "but it is actually " << tensor_type;
    }
    return scatter_nd_d;
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
