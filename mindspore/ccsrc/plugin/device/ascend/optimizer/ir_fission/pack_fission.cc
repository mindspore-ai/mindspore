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
#include "plugin/device/ascend/optimizer/ir_fission/pack_fission.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
AnfNodePtr PackFission::CreateNewPack(const FuncGraphPtr &func_graph, const CNodePtr &origin_pack_cnode,
                                      size_t begin_index, size_t offset) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_pack_cnode);
  std::vector<AnfNodePtr> new_pack_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimPack->name()))};
  for (size_t i = begin_index; i < begin_index + offset; ++i) {
    new_pack_inputs.push_back(origin_pack_cnode->input(i));
  }
  CNodePtr new_pack = NewCNode(new_pack_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_pack);
  new_pack->set_scope(origin_pack_cnode->scope());
  new_pack->set_abstract(origin_pack_cnode->abstract());
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, origin_pack_cnode, new_pack);
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(offset)), new_pack);
  common::AnfAlgo::SetNodeAttr(kAttrNum, MakeValue(SizeToLong(offset)), new_pack);
  std::vector<int64_t> dyn_input_sizes{SizeToLong(offset)};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), new_pack);
  // infer shape
  auto output_shape_ptr = AnfAlgo::GetOutputDetailShape(origin_pack_cnode, 0);
  MS_EXCEPTION_IF_NULL(output_shape_ptr);
  auto output_shape = output_shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(output_shape);
  auto axis = common::AnfAlgo::GetNodeAttr<int64_t>(new_pack, kAttrAxis);
  if (axis < 0) {
    axis += SizeToLong(output_shape->shape().size());
  }
  if (axis < 0) {
    MS_LOG(EXCEPTION) << "The concat_dim value " << axis << "is out of range"
                      << trace::DumpSourceLines(origin_pack_cnode);
  }

  ShapeVector new_shape = output_shape->shape();
  auto axis_l = LongToSize(axis);
  if (axis_l < new_shape.size()) {
    new_shape[axis_l] = SizeToLong(offset);
  }
  auto new_output_shape = std::make_shared<abstract::Shape>(new_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape({common::AnfAlgo::GetOutputInferDataType(origin_pack_cnode, 0)},
                                               {new_output_shape}, new_pack.get());
  return new_pack;
}

const BaseRef PackFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPack, Xs});
}

const AnfNodePtr PackFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // The real input begins with index 1.
  size_t origin_input_size = cnode->inputs().size() - 1;
  if (origin_input_size <= inputs_divisor_) {
    return nullptr;
  }
  std::vector<AnfNodePtr> base_concat_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConcatD->name()))};
  size_t cur_input_index = 1;
  // Divide the inputs of pack by inputs_divisor_.
  while (origin_input_size - cur_input_index + 1 >= inputs_divisor_) {
    base_concat_inputs.emplace_back(CreateNewPack(func_graph, cnode, cur_input_index, inputs_divisor_));
    cur_input_index += inputs_divisor_;
  }
  if (cur_input_index <= origin_input_size) {
    (void)base_concat_inputs.emplace_back(
      CreateNewPack(func_graph, cnode, cur_input_index, (origin_input_size - cur_input_index) + 1));
  }

  CNodePtr base_concat = NewCNode(base_concat_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(base_concat);
  base_concat->set_scope(cnode->scope());
  base_concat->set_abstract(cnode->abstract());
  common::AnfAlgo::CopyNodeAttr(kAttrAxis, cnode, base_concat);
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(base_concat_inputs.size() - 1)), base_concat);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(SizeToLong(base_concat_inputs.size() - 1)), base_concat);
  std::vector<int64_t> dyn_input_sizes{SizeToLong(base_concat_inputs.size() - 1)};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), base_concat);

  return base_concat;
}
}  // namespace opt
}  // namespace mindspore
