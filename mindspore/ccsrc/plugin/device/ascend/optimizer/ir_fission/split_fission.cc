/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/split_fission.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
void SetAttrForSplitVNode(const AnfNodePtr &splitv, const std::vector<int64_t> &size_splits, int64_t split_dim,
                          int64_t num_split) {
  common::AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue(size_splits), splitv);
  common::AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue(split_dim), splitv);
  common::AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue(num_split), splitv);
}

void AddNewOutputs(const FuncGraphPtr &func_graph, const AnfNodePtr &new_splitv, int64_t outputs_num,
                   std::vector<AnfNodePtr> *inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  std::vector<AnfNodePtr> new_splitv_output;
  CreateMultipleOutputsOfAnfNode(func_graph, new_splitv, LongToSize(outputs_num), &new_splitv_output);
  (void)inputs->insert(inputs->cend(), new_splitv_output.cbegin(), new_splitv_output.cend());
}

AnfNodePtr CreateTupleGetItem(const FuncGraphPtr &func_graph, const AnfNodePtr &input, int64_t index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto idx = NewValueNode(index);
  MS_EXCEPTION_IF_NULL(idx);
  auto imm = std::make_shared<Int64Imm>(index);
  auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
  idx->set_abstract(abstract_scalar);
  auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, idx});
  return tuple_getitem;
}

void CreateOutputShapeAndTypeId(const CNodePtr &origin_cnode, int64_t split_dim,
                                const std::vector<int64_t> &size_splits_new, std::vector<TypeId> *new_type_ids,
                                std::vector<ShapeVector> *new_output_shapes) {
  MS_EXCEPTION_IF_NULL(new_type_ids);
  MS_EXCEPTION_IF_NULL(new_output_shapes);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(origin_cnode, 0);
  if (split_dim < 0) {
    split_dim += SizeToLong(output_shape.size());
  }
  size_t split_dim_unsigned = LongToSize(split_dim);
  if (split_dim_unsigned >= output_shape.size()) {
    MS_LOG(EXCEPTION) << "Error split dim: " << split_dim << trace::DumpSourceLines(origin_cnode);
  }
  TypeId type_id = common::AnfAlgo::GetOutputInferDataType(origin_cnode, 0);
  for (size_t i = 0; i < size_splits_new.size(); ++i) {
    (void)new_type_ids->emplace_back(type_id);
    output_shape[split_dim_unsigned] = size_splits_new[i];
    (void)new_output_shapes->emplace_back(output_shape);
  }
}

void SetAttrAndAbstractForBaseSplitv(const CNodePtr &origin_cnode, const CNodePtr &base_splitv,
                                     const std::vector<AnfNodePtr> &base_splitv_outputs,
                                     const std::vector<int64_t> &size_splits_base, int64_t split_dim,
                                     int64_t num_split) {
  SetAttrForSplitVNode(base_splitv, size_splits_base, split_dim, num_split);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(origin_cnode, 0);
  TypeId type_id = common::AnfAlgo::GetOutputInferDataType(origin_cnode, 0);
  std::vector<TypeId> base_type_ids(num_split, type_id);
  std::vector<ShapeVector> base_output_shapes_base;
  if (split_dim < 0) {
    split_dim += SizeToLong(output_shape.size());
  }
  if (split_dim < 0) {
    MS_LOG(EXCEPTION) << "Error split dim: " << split_dim << trace::DumpSourceLines(origin_cnode);
  }
  auto split_dim_l = LongToSize(split_dim);
  auto num_split_l = LongToSize(num_split);
  for (size_t i = 0; i < num_split_l; ++i) {
    output_shape[split_dim_l] = size_splits_base[i];
    (void)base_output_shapes_base.emplace_back(output_shape);
    common::AnfAlgo::SetOutputInferTypeAndShape({type_id}, {output_shape}, base_splitv_outputs[i].get());
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(base_type_ids, base_output_shapes_base, base_splitv.get());
}
}  // namespace

CNodePtr SplitFission::CreateSplitVNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> splitv_inputs{NewValueNode(std::make_shared<Primitive>(kSplitVDOpName)), input_node};
  CNodePtr splitv = NewCNode(splitv_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(splitv);
  splitv->set_scope(input_node->scope());
  return splitv;
}

CNodePtr SplitFission::CreateBaseSplitVNode(const FuncGraphPtr &func_graph, const CNodePtr &origin_cnode) const {
  MS_EXCEPTION_IF_NULL(origin_cnode);
  CheckCNodeInputSize(origin_cnode, kSplitInputTensorNum);
  return CreateSplitVNode(func_graph, origin_cnode->input(1));
}

AnfNodePtr SplitFission::DoFission(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int64_t num_split,
                                   int64_t divisor, int64_t split_dim) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr base_splitv = CreateBaseSplitVNode(func_graph, cnode);

  auto size_splits = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrSizeSplits);
  if (size_splits.size() != LongToSize(num_split)) {
    MS_LOG(EXCEPTION) << "The size of size_splits should be equal to num_split[" << num_split << "], but got "
                      << size_splits.size() << ", node: " << cnode->fullname_with_scope()
                      << trace::DumpSourceLines(cnode);
  }

  // Create make_tuple input to create a make_tuple for replacing the old Split node.
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  // Start to divide the outputs of Split.
  std::vector<int64_t> size_splits_base;
  std::vector<AnfNodePtr> base_splitv_outputs;
  int64_t nodes_num = 0;
  int64_t cur_output_index = 0;
  while (num_split - cur_output_index > divisor) {
    auto tuple_getitem = CreateTupleGetItem(func_graph, base_splitv, nodes_num);
    (void)base_splitv_outputs.emplace_back(tuple_getitem);
    CNodePtr new_splitv = CreateSplitVNode(func_graph, tuple_getitem);
    std::vector<int64_t> size_splits_new(size_splits.begin() + nodes_num * divisor,
                                         size_splits.begin() + (nodes_num + 1) * divisor);
    SetAttrForSplitVNode(new_splitv, size_splits_new, split_dim, divisor);
    // Create new output shape and new output type id for each new Splitv node which has full inputs.
    std::vector<TypeId> new_type_ids;
    std::vector<ShapeVector> new_output_shapes;
    CreateOutputShapeAndTypeId(cnode, split_dim, size_splits_new, &new_type_ids, &new_output_shapes);
    common::AnfAlgo::SetOutputInferTypeAndShape(new_type_ids, new_output_shapes, new_splitv.get());
    AddNewOutputs(func_graph, new_splitv, divisor, &make_tuple_inputs);
    cur_output_index += divisor;
    int64_t split_size = std::accumulate(size_splits_new.begin(), size_splits_new.end(), int64_t(0));
    (void)size_splits_base.emplace_back(split_size);
    nodes_num++;
  }
  // create last splitv or getitem when last_node_num_split == 1
  if (cur_output_index < num_split) {
    auto last_node_num_split = num_split - cur_output_index;
    if (last_node_num_split > 1) {
      auto tuple_getitem = CreateTupleGetItem(func_graph, base_splitv, nodes_num);
      (void)base_splitv_outputs.emplace_back(tuple_getitem);
      CNodePtr new_splitv = CreateSplitVNode(func_graph, tuple_getitem);
      std::vector<int64_t> size_splits_new_last(size_splits.begin() + nodes_num * divisor, size_splits.end());
      SetAttrForSplitVNode(new_splitv, size_splits_new_last, split_dim, last_node_num_split);
      // Create new output shape and new output type id for the last Splitv node
      std::vector<TypeId> last_new_type_ids;
      std::vector<ShapeVector> last_new_output_shapes;
      CreateOutputShapeAndTypeId(cnode, split_dim, size_splits_new_last, &last_new_type_ids, &last_new_output_shapes);
      common::AnfAlgo::SetOutputInferTypeAndShape(last_new_type_ids, last_new_output_shapes, new_splitv.get());
      AddNewOutputs(func_graph, new_splitv, last_node_num_split, &make_tuple_inputs);
      int64_t last_split_size = std::accumulate(size_splits_new_last.begin(), size_splits_new_last.end(), int64_t(0));
      (void)size_splits_base.emplace_back(last_split_size);
    } else {
      auto tuple_getitem = CreateTupleGetItem(func_graph, base_splitv, nodes_num);
      (void)base_splitv_outputs.emplace_back(tuple_getitem);
      (void)make_tuple_inputs.emplace_back(tuple_getitem);
      (void)size_splits_base.emplace_back(size_splits[size_splits.size() - 1]);
    }
    nodes_num++;
  }
  // Set Attr and abstract for the base splitv
  SetAttrAndAbstractForBaseSplitv(cnode, base_splitv, base_splitv_outputs, size_splits_base, split_dim, nodes_num);
  AnfNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}

const BaseRef SplitFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto split_prim = std::make_shared<Primitive>(kSplitDOpName);
  return VectorRef({split_prim, Xs});
}

const AnfNodePtr SplitFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Check output num
  if (!common::AnfAlgo::HasNodeAttr(kAttrOutputNum, cnode)) {
    return nullptr;
  }
  auto num_split = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrOutputNum);
  if (num_split <= outputs_divisor_) {
    return nullptr;
  }
  return DoFission(func_graph, cnode, num_split, outputs_divisor_,
                   common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrAxis));
}
}  // namespace opt
}  // namespace mindspore
