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
#include "backend/optimizer/ascend/ir_fission/split_fission.h"
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
CNodePtr CreateSplitVNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> splitv_inputs{NewValueNode(std::make_shared<Primitive>(kSplitVOpName)), input_node};
  CNodePtr splitv = func_graph->NewCNode(splitv_inputs);
  MS_EXCEPTION_IF_NULL(splitv);
  splitv->set_scope(input_node->scope());
  return splitv;
}

CNodePtr CreateBaseSplitVNode(const FuncGraphPtr &func_graph, const CNodePtr &origin_cnode) {
  MS_EXCEPTION_IF_NULL(origin_cnode);
  CheckCNodeInputSize(origin_cnode, kSplitInputTensorNum);
  return CreateSplitVNode(func_graph, origin_cnode->input(1));
}

void SetAttrForSplitVNode(const AnfNodePtr &splitv, const std::vector<int64_t> &size_splits, int64_t split_dim,
                          int64_t num_split) {
  AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue(size_splits), splitv);
  AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue(split_dim), splitv);
  AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue(num_split), splitv);
}

size_t GetSmallSplitSize(const AnfNodePtr &split_node, int64_t split_dim, int64_t num_split) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(split_node, 0);
  if (split_dim < 0) {
    split_dim += input_shape.size();
  }
  if (LongToSize(split_dim) >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "The split_dim value should be less than the shape size of input 0";
  }
  return input_shape[split_dim] / num_split;
}

void AddNewOutputs(const FuncGraphPtr &func_graph, const AnfNodePtr &new_splitv, int64_t outputs_num,
                   std::vector<AnfNodePtr> *inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  std::vector<AnfNodePtr> new_splitv_output;
  CreateMultipleOutputsOfAnfNode(func_graph, new_splitv, LongToSize(outputs_num), &new_splitv_output);
  inputs->insert(inputs->end(), new_splitv_output.begin(), new_splitv_output.end());
}

AnfNodePtr CreateTupleGetItem(const FuncGraphPtr &func_graph, const AnfNodePtr &input, size_t index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto idx = NewValueNode(SizeToLong(index));
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(SizeToLong(index));
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, idx});
  return tuple_getitem;
}

void CreateOutputShapeAndTypeId(const CNodePtr &origin_cnode, int64_t split_dim, int64_t split_size, int64_t num_split,
                                std::vector<TypeId> *new_type_ids,
                                std::vector<std::vector<size_t>> *new_output_shapes) {
  MS_EXCEPTION_IF_NULL(new_type_ids);
  MS_EXCEPTION_IF_NULL(new_output_shapes);
  auto output_shape = AnfAlgo::GetOutputInferShape(origin_cnode, 0);
  if (split_dim < 0) {
    split_dim += output_shape.size();
  }
  output_shape[split_dim] = split_size;
  TypeId type_id = AnfAlgo::GetOutputInferDataType(origin_cnode, 0);
  for (int64_t i = 0; i < num_split; ++i) {
    new_type_ids->emplace_back(type_id);
    new_output_shapes->emplace_back(output_shape);
  }
}

void SetAttrAndAbstractForBaseSplitv(const CNodePtr &origin_cnode, const CNodePtr &base_splitv,
                                     const std::vector<AnfNodePtr> &base_splitv_outputs,
                                     const std::vector<int64_t> &size_splits_base, int64_t split_dim,
                                     int64_t num_split) {
  SetAttrForSplitVNode(base_splitv, size_splits_base, split_dim, num_split);
  auto output_shape = AnfAlgo::GetOutputInferShape(origin_cnode, 0);
  TypeId type_id = AnfAlgo::GetOutputInferDataType(origin_cnode, 0);
  std::vector<TypeId> base_type_ids(num_split, type_id);
  std::vector<std::vector<size_t>> base_output_shapes_base;
  if (split_dim < 0) {
    split_dim += output_shape.size();
  }
  for (int64_t i = 0; i < num_split; ++i) {
    output_shape[split_dim] = size_splits_base[i];
    base_output_shapes_base.emplace_back(output_shape);
    AnfAlgo::SetOutputInferTypeAndShape({type_id}, {output_shape}, base_splitv_outputs[i].get());
  }
  AnfAlgo::SetOutputInferTypeAndShape(base_type_ids, base_output_shapes_base, base_splitv.get());
}

AnfNodePtr DoFission(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int64_t num_split, int64_t divisor) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto split_dim = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrAxis);
  CNodePtr base_splitv = CreateBaseSplitVNode(func_graph, cnode);

  // Create new size_splits for "size_splits" attr of each new Splitv node which has full inputs.
  auto small_split_size = SizeToLong(GetSmallSplitSize(cnode, split_dim, num_split));
  std::vector<int64_t> size_splits_new(divisor, small_split_size);
  // Create new output shape and new output type id for each new Splitv node which has full inputs.
  std::vector<TypeId> new_type_ids;
  std::vector<std::vector<size_t>> new_output_shapes;
  CreateOutputShapeAndTypeId(cnode, split_dim, small_split_size, divisor, &new_type_ids, &new_output_shapes);

  // Create make_tuple input to create a make_tuple for replacing the old Split node.
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  // Start to divide the outputs of Split.
  std::vector<int64_t> size_splits_base;
  std::vector<AnfNodePtr> base_splitv_outputs;
  const auto base_split_size = divisor * small_split_size;
  int64_t nodes_num = 0;
  int64_t cur_output_index = 0;
  while (num_split - cur_output_index > divisor) {
    auto tuple_getitem = CreateTupleGetItem(func_graph, base_splitv, nodes_num);
    base_splitv_outputs.push_back(tuple_getitem);
    CNodePtr new_splitv = CreateSplitVNode(func_graph, tuple_getitem);
    SetAttrForSplitVNode(new_splitv, size_splits_new, split_dim, divisor);
    AnfAlgo::SetOutputInferTypeAndShape(new_type_ids, new_output_shapes, new_splitv.get());
    AddNewOutputs(func_graph, new_splitv, divisor, &make_tuple_inputs);
    cur_output_index += divisor;
    size_splits_base.emplace_back(base_split_size);
    nodes_num++;
  }
  if (cur_output_index < num_split) {
    auto last_node_num_split = num_split - cur_output_index;
    if (last_node_num_split > 1) {
      auto tuple_getitem = CreateTupleGetItem(func_graph, base_splitv, nodes_num);
      base_splitv_outputs.push_back(tuple_getitem);
      CNodePtr new_splitv = CreateSplitVNode(func_graph, tuple_getitem);
      std::vector<int64_t> size_splits_new_last(last_node_num_split, small_split_size);
      SetAttrForSplitVNode(new_splitv, size_splits_new_last, split_dim, last_node_num_split);
      // Create new output shape and new output type id for the last Splitv node
      std::vector<TypeId> last_new_type_ids;
      std::vector<std::vector<size_t>> last_new_output_shapes;
      CreateOutputShapeAndTypeId(cnode, split_dim, small_split_size, last_node_num_split, &last_new_type_ids,
                                 &last_new_output_shapes);
      AnfAlgo::SetOutputInferTypeAndShape(last_new_type_ids, last_new_output_shapes, new_splitv.get());
      AddNewOutputs(func_graph, new_splitv, last_node_num_split, &make_tuple_inputs);
      size_splits_base.emplace_back(last_node_num_split * small_split_size);
    } else {
      auto tuple_getitem = CreateTupleGetItem(func_graph, base_splitv, nodes_num);
      base_splitv_outputs.push_back(tuple_getitem);
      make_tuple_inputs.emplace_back(tuple_getitem);
      size_splits_base.emplace_back(small_split_size);
    }
    nodes_num++;
  }
  // Set Attr and abstract for the base splitv
  SetAttrAndAbstractForBaseSplitv(cnode, base_splitv, base_splitv_outputs, size_splits_base, split_dim, nodes_num);
  AnfNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}
}  // namespace

const BaseRef SplitFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto split_prim = std::make_shared<Primitive>(kSplitOpName);
  return VectorRef({split_prim, Xs});
}

const AnfNodePtr SplitFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Check output num
  if (!AnfAlgo::HasNodeAttr(kAttrOutputNum, cnode)) {
    return nullptr;
  }
  auto num_split = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrOutputNum);
  if (num_split <= outputs_divisor_) {
    return nullptr;
  }
  return DoFission(func_graph, cnode, num_split, outputs_divisor_);
}
}  // namespace opt
}  // namespace mindspore
