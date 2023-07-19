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
#include "plugin/device/ascend/optimizer/ge/expand_dims_for_batchnorm.h"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include "ops/array_op_name.h"
#include "ops/nn_ops.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
namespace {
enum class OutputReshapeType { kNoReshape, kReshapeSingleOutput, kReshapeFirstOutput };

struct ExpandDimsInfo {
  const PrimitivePtr &prim;
  std::set<size_t> input_indexes;
  OutputReshapeType reshape_type;
};

// The inputs to reshaped are determined based on input_indexes.
// Only the first output of BNTrainingReduceGrad and BNTrainingUpdate need to be reshaped.
const std::vector<ExpandDimsInfo> kExpandInfos = {
  {prim::kPrimBNTrainingReduce, {0}, OutputReshapeType::kNoReshape},
  {prim::kPrimBNTrainingReduceGrad, {0, 1}, OutputReshapeType::kReshapeSingleOutput},
  {prim::kPrimBNTrainingUpdate, {0}, OutputReshapeType::kReshapeFirstOutput},
  {prim::kPrimBNTrainingUpdateGrad, {0, 1}, OutputReshapeType::kNoReshape},
};

bool IsBatchNorm(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);

    for (const auto &expand_info : kExpandInfos) {
      if (IsPrimitive(node, expand_info.prim)) {
        return true;
      }
    }
  }

  return false;
}

bool NeedReshape(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return false;
  }

  CNodePtr get_item_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(get_item_node);

  AnfNodePtr input2 = common::AnfAlgo::GetInputNode(get_item_node, kIndex1);
  MS_EXCEPTION_IF_NULL(input2);
  if (!input2->isa<ValueNode>()) {
    return true;
  }

  const ValuePtr &value = GetValueNode(input2);
  MS_EXCEPTION_IF_NULL(value);
  if (!utils::isa<ValueSequencePtr>(value)) {
    MS_EXCEPTION_IF_NULL(value->type());
    if (value->type()->number_type() == kNumberTypeInt64) {
      return GetValue<int64_t>(value) == 0;
    } else {
      return GetValue<int>(value) == 0;
    }
  }

  return true;
}

// Parameter index starts from 0.
bool ExpandInputDims(const FuncGraphPtr &graph, const CNodePtr &bn_node, size_t index,
                     const FuncGraphManagerPtr &manager) {
  ShapeVector input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(bn_node, index);
  size_t dim_len = input_shape.size();
  if (dim_len >= kDim4) {
    return false;
  }

  // Expand shape to 4 dimensions
  ShapeVector new_shape = input_shape;
  for (size_t i = 0; i < kDim4 - dim_len; i++) {
    (void)new_shape.emplace_back(1);
  }

  // Replace the first input of bn with reshape
  AnfNodePtr input_node = common::AnfAlgo::GetInputNode(bn_node, index);
  CNodePtr reshape_node = mindspore::common::CreateReshapeNode(graph, input_node, new_shape);
  manager->SetEdge(bn_node, index + 1, reshape_node);

  MS_LOG(INFO) << "Expand input dims for node " << bn_node->fullname_with_scope() << ", input index: " << index
               << ", input node: " << input_node << ", new reshape node: " << reshape_node->fullname_with_scope();

  return true;
}

bool ExpandInputDims(const FuncGraphPtr &graph, const CNodePtr &bn_node, const ExpandDimsInfo &expand_info,
                     const FuncGraphManagerPtr &manager) {
  bool need_expand = false;
  auto &input_indexes = expand_info.input_indexes;
  (void)std::for_each(input_indexes.begin(), input_indexes.end(), [&](size_t index) {
    if (ExpandInputDims(graph, bn_node, index, manager)) {
      need_expand = true;
    }
  });

  return need_expand;
}

void ExpandSingleOutputDims(const FuncGraphPtr &graph, const CNodePtr &bn_node, const FuncGraphManagerPtr &manager) {
  auto bn_output_shape = common::AnfAlgo::GetOutputInferShape(bn_node, kIndex0);
  CNodePtr reshape_node = mindspore::common::CreateReshapeNode(graph, bn_node, bn_output_shape);
  (void)manager->Replace(bn_node, reshape_node);

  MS_LOG(INFO) << "Expand single output dims for node " << bn_node->fullname_with_scope()
               << ", new reshape node: " << reshape_node->fullname_with_scope();
}

void ExpandMultiOutputDims(const FuncGraphPtr &graph, const CNodePtr &bn_node, const FuncGraphManagerPtr &manager) {
  auto &node_users = manager->node_users();
  auto iter = node_users.find(bn_node);
  if (iter == node_users.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Find users of node " << bn_node->fullname_with_scope() << " failed."
                               << trace::DumpSourceLines(bn_node);
  }

  const AnfNodeIndexSet &user_set = iter->second;
  for (const auto &user_pair : user_set) {
    auto user = user_pair.first;
    MS_EXCEPTION_IF_NULL(user);

    if (!NeedReshape(user)) {
      continue;
    }

    auto get_item_output_shape = common::AnfAlgo::GetOutputInferShape(user, kIndex0);
    CNodePtr reshape_node = mindspore::common::CreateReshapeNode(graph, user, get_item_output_shape);
    (void)manager->Replace(user, reshape_node);

    MS_LOG(INFO) << "Expand output dims for node " << bn_node->fullname_with_scope()
                 << ", user node: " << user->fullname_with_scope()
                 << ", new reshape node: " << reshape_node->fullname_with_scope();
  }
}

void ExpandOutputDims(const FuncGraphPtr &graph, const CNodePtr &bn_node, const ExpandDimsInfo &expand_info,
                      const FuncGraphManagerPtr &manager) {
  if (expand_info.reshape_type == OutputReshapeType::kReshapeSingleOutput) {
    ExpandSingleOutputDims(graph, bn_node, manager);
  } else if (expand_info.reshape_type == OutputReshapeType::kReshapeFirstOutput) {
    ExpandMultiOutputDims(graph, bn_node, manager);
  }
}
}  // namespace

const BaseRef ExpandDimsForBatchNorm::DefinePattern() const {
  VarPtr bn = std::make_shared<CondVar>(IsBatchNorm);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({bn, inputs});
}

const AnfNodePtr ExpandDimsForBatchNorm::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  for (const auto &expand_info : kExpandInfos) {
    if (!IsPrimitiveCNode(node, expand_info.prim)) {
      continue;
    }

    CNodePtr bn_node = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(bn_node);

    bool need_expand = ExpandInputDims(graph, bn_node, expand_info, manager);
    if (need_expand) {
      ExpandOutputDims(graph, bn_node, expand_info, manager);
    }

    return node;
  }

  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
