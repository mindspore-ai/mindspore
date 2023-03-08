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

#include "plugin/device/ascend/optimizer/enhancer/insert_pad_for_nms_with_mask.h"
#include <memory>
#include <vector>
#include "backend/common/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kShapeSize = 2;
constexpr int64_t kShapeValue5 = 5;
constexpr int64_t kShapeValue8 = 8;
}  // namespace
const BaseRef InsertPadForNMSWithMask::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimNMSWithMask, Xs});
}

AnfNodePtr InsertPadForNMSWithMask::InsertPadToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                     const TypeId &origin_type,
                                                     const abstract::BaseShapePtr &origin_shape) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_pad_inputs;
  auto prim = std::make_shared<Primitive>(prim::kPrimPadD->name());
  new_pad_inputs.push_back(NewValueNode(prim));
  new_pad_inputs.push_back(input);
  CNodePtr pad = NewCNode(new_pad_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(pad);
  common::AnfAlgo::SetOutputTypeAndDetailShape({origin_type}, {origin_shape}, pad.get());
  return pad;
}

const AnfNodePtr InsertPadForNMSWithMask::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  if (input_num == 0) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(cnode)};
  for (size_t input_idx = 0; input_idx < input_num; input_idx++) {
    auto cur_input = common::AnfAlgo::GetInputNode(cnode, input_idx);
    auto origin_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_idx);
    auto origin_shape_base_ptr = AnfAlgo::GetPrevNodeOutputDetailShape(cnode, input_idx);
    auto origin_shape_ptr = origin_shape_base_ptr->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(origin_shape_ptr);
    auto origin_shape = origin_shape_ptr->shape();
    if (!(origin_shape.size() == kShapeSize && origin_shape[1] == kShapeValue5)) {
      return nullptr;
    }
    origin_shape[1] = kShapeValue8;
    abstract::ShapePtr out_shape_ptr = std::make_shared<abstract::Shape>(origin_shape);
    auto pad = InsertPadToGraph(func_graph, cur_input, origin_type, out_shape_ptr);
    MS_EXCEPTION_IF_NULL(pad);
    pad->set_scope(cnode->scope());
    common::AnfAlgo::SetNodeAttr("paddings", MakeValue(std::vector<std::vector<int64_t>>{{0, 0}, {0, 3}}), pad);
    new_inputs.push_back(pad);
  }
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  CNodePtr new_node = nullptr;
  if (kernel_graph == nullptr) {
    new_node = std::make_shared<CNode>(*cnode);
  } else {
    new_node = NewCNode(cnode, kernel_graph);
  }
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_inputs(new_inputs);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
