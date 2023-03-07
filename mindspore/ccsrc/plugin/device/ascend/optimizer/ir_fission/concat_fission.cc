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
#include "plugin/device/ascend/optimizer/ir_fission/concat_fission.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
AnfNodePtr ConcatFission::CreateNewConcat(const FuncGraphPtr &func_graph, const CNodePtr &origin_concat_cnode,
                                          size_t begin_index, size_t offset) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_concat_cnode);
  std::vector<AnfNodePtr> new_concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcatD->name()))};
  for (size_t i = begin_index; i < begin_index + offset; ++i) {
    new_concat_inputs.emplace_back(origin_concat_cnode->input(i));
  }
  CNodePtr new_concat = NewCNode(new_concat_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_concat);
  new_concat->set_scope(origin_concat_cnode->scope());
  // Set attrs
  if (common::AnfAlgo::HasNodeAttr(kAttrAxis, origin_concat_cnode)) {
    common::AnfAlgo::CopyNodeAttr(kAttrAxis, origin_concat_cnode, new_concat);
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrT, origin_concat_cnode)) {
    common::AnfAlgo::CopyNodeAttr(kAttrT, origin_concat_cnode, new_concat);
  }
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(offset)), new_concat);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(SizeToLong(offset)), new_concat);
  std::vector<int64_t> dyn_input_sizes{SizeToLong(offset)};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), new_concat);
  // infer shape
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_concat_cnode, 0);
  auto axis_from_attr = common::AnfAlgo::GetNodeAttr<int64_t>(origin_concat_cnode, kAttrAxis);
  if (axis_from_attr < 0) {
    axis_from_attr += SizeToLong(input_shape.size());
  }
  auto output_shape_ptr = AnfAlgo::GetOutputDetailShape(origin_concat_cnode, 0);
  MS_EXCEPTION_IF_NULL(output_shape_ptr);
  auto output_shapeptr = output_shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(output_shapeptr);
  auto output_shape = output_shapeptr->shape();
  if (axis_from_attr < 0 || axis_from_attr >= SizeToLong(output_shape.size()) ||
      axis_from_attr >= SizeToLong(input_shape.size())) {
    MS_LOG(EXCEPTION) << "The concat_dim value " << axis_from_attr << "is out of range"
                      << trace::DumpSourceLines(origin_concat_cnode);
  }
  auto axis = LongToSize(axis_from_attr);
  output_shape[axis] = 0;
  for (size_t i = begin_index; i < begin_index + offset; ++i) {
    auto last_input_shape_ptr = AnfAlgo::GetPrevNodeOutputDetailShape(origin_concat_cnode, i - 1);
    MS_EXCEPTION_IF_NULL(last_input_shape_ptr);
    auto last_input_shapeptr = last_input_shape_ptr->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(last_input_shapeptr);
    auto last_input_shape = last_input_shapeptr->shape();
    if (last_input_shape[axis] == -1 || output_shape[axis] == -1) {
      output_shape[axis] = -1;
    } else {
      output_shape[axis] += last_input_shape[axis];
    }
  }
  auto concat_output_shape_ptr = std::make_shared<abstract::Shape>(output_shape);
  common::AnfAlgo::SetOutputTypeAndDetailShape({common::AnfAlgo::GetOutputInferDataType(origin_concat_cnode, 0)},
                                               {concat_output_shape_ptr}, new_concat.get());
  return new_concat;
}

const BaseRef ConcatFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimConcatD, Xs});
}

const AnfNodePtr ConcatFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // The real input begins with index 1.
  size_t origin_input_size = cnode->inputs().size() - 1;
  if (origin_input_size <= inputs_divisor_) {
    return nullptr;
  }
  CNodePtr new_cnode = cnode;
  while (origin_input_size > inputs_divisor_) {
    MS_EXCEPTION_IF_NULL(new_cnode);
    std::vector<AnfNodePtr> base_concat_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConcatD->name()))};
    size_t cur_input_index = 1;
    // Divide the inputs of concat by inputs_divisor_.
    while (origin_input_size - cur_input_index + 1 >= inputs_divisor_) {
      base_concat_inputs.push_back(CreateNewConcat(func_graph, new_cnode, cur_input_index, inputs_divisor_));
      cur_input_index += inputs_divisor_;
    }
    for (size_t i = cur_input_index; i <= origin_input_size; i++) {
      base_concat_inputs.emplace_back(new_cnode->input(i));
    }
    CNodePtr base_concat = NewCNode(base_concat_inputs, func_graph);
    MS_EXCEPTION_IF_NULL(base_concat);
    base_concat->set_scope(new_cnode->scope());
    base_concat->set_abstract(new_cnode->abstract());
    // Set attrs
    if (common::AnfAlgo::HasNodeAttr(kAttrAxis, new_cnode)) {
      common::AnfAlgo::CopyNodeAttr(kAttrAxis, new_cnode, base_concat);
    }
    if (common::AnfAlgo::HasNodeAttr(kAttrT, new_cnode)) {
      common::AnfAlgo::CopyNodeAttr(kAttrT, new_cnode, base_concat);
    }

    common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(base_concat_inputs.size() - 1)), base_concat);
    common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(SizeToLong(base_concat_inputs.size() - 1)), base_concat);
    std::vector<int64_t> dyn_input_sizes{SizeToLong(base_concat_inputs.size() - 1)};
    common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), base_concat);

    new_cnode = base_concat;
    origin_input_size = base_concat->inputs().size() - 1;
  }

  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
