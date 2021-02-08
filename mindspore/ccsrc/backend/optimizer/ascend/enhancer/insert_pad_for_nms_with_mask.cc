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

#include "backend/optimizer/ascend/enhancer/insert_pad_for_nms_with_mask.h"
#include <vector>
#include <memory>
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "base/core_ops.h"

namespace mindspore {
namespace opt {
const BaseRef InsertPadForNMSWithMask::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimNMSWithMask, Xs});
}

AnfNodePtr InsertPadToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const TypeId &origin_type,
                            const std::vector<size_t> &origin_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_pad_inputs;
  auto prim = std::make_shared<Primitive>(prim::kPrimPad->name());
  new_pad_inputs.push_back(NewValueNode(prim));
  new_pad_inputs.push_back(input);
  CNodePtr pad = func_graph->NewCNode(new_pad_inputs);
  MS_EXCEPTION_IF_NULL(pad);
  AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {origin_shape}, pad.get());
  return pad;
}

const AnfNodePtr InsertPadForNMSWithMask::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  size_t input_num = AnfAlgo::GetInputTensorNum(node);
  if (input_num == 0) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_inputs = {AnfAlgo::GetCNodePrimitiveNode(cnode)};
  for (size_t input_idx = 0; input_idx < input_num; input_idx++) {
    auto cur_input = AnfAlgo::GetInputNode(cnode, input_idx);
    auto origin_type = AnfAlgo::GetPrevNodeOutputInferDataType(cnode, input_idx);
    auto origin_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, input_idx);
    if (!(origin_shape.size() == 2 && origin_shape[1] == 5)) {
      return nullptr;
    }
    origin_shape[1] = 8;
    auto pad = InsertPadToGraph(func_graph, cur_input, origin_type, origin_shape);
    MS_EXCEPTION_IF_NULL(pad);
    pad->set_scope(cnode->scope());
    AnfAlgo::SetNodeAttr("paddings", MakeValue(std::vector<std::vector<int64_t>>{{0, 0}, {0, 3}}), pad);
    new_inputs.push_back(pad);
  }
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  CNodePtr new_node = nullptr;
  if (kernel_graph == nullptr) {
    new_node = std::make_shared<CNode>(*cnode);
  } else {
    new_node = kernel_graph->NewCNode(cnode);
  }
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_inputs(new_inputs);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
