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
#include "backend/optimizer/ascend/ir_fission/pack_fission.h"
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr CreateNewPack(const FuncGraphPtr &func_graph, const CNodePtr &origin_pack_cnode, size_t begin_index,
                         size_t offset) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_pack_cnode);
  std::vector<AnfNodePtr> new_pack_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimStack->name()))};
  for (size_t i = begin_index; i < begin_index + offset; ++i) {
    new_pack_inputs.push_back(origin_pack_cnode->input(i));
  }
  CNodePtr new_pack = func_graph->NewCNode(new_pack_inputs);
  MS_EXCEPTION_IF_NULL(new_pack);
  new_pack->set_scope(origin_pack_cnode->scope());
  new_pack->set_abstract(origin_pack_cnode->abstract());
  AnfAlgo::CopyNodeAttr(kAttrAxis, origin_pack_cnode, new_pack);
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(offset)), new_pack);
  AnfAlgo::SetNodeAttr(kAttrNum, MakeValue(SizeToLong(offset)), new_pack);
  std::vector<int64_t> dyn_input_sizes{SizeToLong(offset)};
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), new_pack);
  // infer shape
  auto output_shape = AnfAlgo ::GetOutputInferShape(origin_pack_cnode, 0);
  auto axis = AnfAlgo::GetNodeAttr<int64_t>(new_pack, kAttrAxis);
  if (axis < 0) {
    axis += output_shape.size();
  }
  if (axis < 0) {
    MS_LOG(EXCEPTION) << "The concat_dim value " << axis << "is out of range"
                      << " trace: " << trace::DumpSourceLines(origin_pack_cnode);
  }
  std::vector<size_t> new_shape;
  for (size_t i = 0; i < output_shape.size() + 1; ++i) {
    if (i < LongToSize(axis)) {
      new_shape.push_back(output_shape[i]);
    } else if (i == LongToSize(axis)) {
      new_shape.push_back(offset);
    } else {
      new_shape.push_back(output_shape[SizeToLong(i) - 1]);
    }
  }
  new_shape.erase(new_shape.begin() + axis + 1);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(origin_pack_cnode, 0)}, {new_shape},
                                      new_pack.get());
  return new_pack;
}
}  // namespace

const BaseRef PackFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimStack, Xs});
}

const AnfNodePtr PackFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // The real input begins with index 1.
  size_t origin_input_size = cnode->inputs().size() - 1;
  if (origin_input_size <= inputs_divisor_) {
    return nullptr;
  }
  std::vector<AnfNodePtr> base_concat_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
  size_t cur_input_index = 1;
  // Divide the inputs of pack by inputs_divisor_.
  while (origin_input_size - cur_input_index + 1 >= inputs_divisor_) {
    base_concat_inputs.emplace_back(CreateNewPack(func_graph, cnode, cur_input_index, inputs_divisor_));
    cur_input_index += inputs_divisor_;
  }
  if (cur_input_index <= origin_input_size) {
    base_concat_inputs.emplace_back(
      CreateNewPack(func_graph, cnode, cur_input_index, origin_input_size - cur_input_index + 1));
  }

  CNodePtr base_concat = func_graph->NewCNode(base_concat_inputs);
  MS_EXCEPTION_IF_NULL(base_concat);
  base_concat->set_scope(cnode->scope());
  base_concat->set_abstract(cnode->abstract());
  AnfAlgo::CopyNodeAttr(kAttrAxis, cnode, base_concat);
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(base_concat_inputs.size() - 1)), base_concat);
  AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(SizeToLong(base_concat_inputs.size() - 1)), base_concat);
  std::vector<int64_t> dyn_input_sizes{SizeToLong(base_concat_inputs.size() - 1)};
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), base_concat);

  return base_concat;
}
}  // namespace opt
}  // namespace mindspore
