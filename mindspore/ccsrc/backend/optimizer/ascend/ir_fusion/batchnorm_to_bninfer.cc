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
#include "backend/optimizer/ascend/ir_fusion/batchnorm_to_bninfer.h"
#include <memory>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
CNodePtr CreateBNInfer(const FuncGraphPtr &graph, const CNodePtr &batchnorm, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(batchnorm);
  MS_EXCEPTION_IF_NULL(node);
  auto prim = std::make_shared<Primitive>(kBNInferOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
  for (size_t i = 1; i < batchnorm->size(); ++i) {
    inputs.push_back(batchnorm->input(i));
  }
  auto new_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(batchnorm->scope());
  new_node->set_abstract(node->abstract());
  AnfAlgo::CopyNodeAttr(kAttrIsTraining, batchnorm, new_node);
  AnfAlgo::CopyNodeAttr(kAttrEpsilon, batchnorm, new_node);
  return new_node;
}

bool CheckIndex(const AnfNodePtr &index_node) {
  MS_EXCEPTION_IF_NULL(index_node);
  if (!IsValueNode<Int64Imm>(index_node)) {
    return false;
  }
  ValueNodePtr value_node = index_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto index = GetValue<int64_t>(value_node->value());
  if (index != 0) {
    MS_LOG(DEBUG) << "tuple_getitem must be 0th output of BatchNorm";
    return false;
  }
  return true;
}

bool CheckBatchNorm(const FuncGraphPtr &graph, const CNodePtr &batchnorm) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(batchnorm);
  if (AnfAlgo::GetInputTensorNum(batchnorm) < kBnInputTensorNum) {
    MS_LOG(DEBUG) << "BatchNorm's input less than " << kBnInputTensorNum;
    return false;
  }
  if (!AnfAlgo::HasNodeAttr(kAttrIsTraining, batchnorm)) {
    return false;
  }
  auto is_training = AnfAlgo::GetNodeAttr<bool>(batchnorm, kAttrIsTraining);
  if (is_training) {
    MS_LOG(DEBUG) << "is_training is true, no need do fusion";
    return false;
  }

  if (IsUsedByOthers(graph, batchnorm)) {
    MS_LOG(DEBUG) << "Only the 0th output of BatchNorm is used, then do fusion";
    return false;
  }
  return true;
}

bool NeedFusion(const FuncGraphPtr &graph, const AnfNodePtr &node, CNodePtr *batchnorm) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto tuple_getitem = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  CheckCNodeInputSize(tuple_getitem, kTupleGetItemInputTensorNum);
  AnfNodePtr index_node = tuple_getitem->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(index_node);
  if (!CheckIndex(index_node)) {
    return false;
  }

  AnfNodePtr batchnorm_anf = tuple_getitem->input(kRealInputNodeIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(batchnorm_anf);
  MS_EXCEPTION_IF_NULL(batchnorm);
  *batchnorm = batchnorm_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(*batchnorm);
  return CheckBatchNorm(graph, *batchnorm);
}
}  // namespace

const BaseRef BatchNorm2BNInfer::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Y = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Y);
  VectorRef batchnorm({prim::kPrimBatchNorm, Xs});
  VectorRef pattern({prim::kPrimTupleGetItem, batchnorm, Y});
  return pattern;
}

const AnfNodePtr BatchNorm2BNInfer::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  CNodePtr batchnorm = nullptr;
  if (!NeedFusion(graph, node, &batchnorm)) {
    return nullptr;
  }
  auto bn_infer = CreateBNInfer(graph, batchnorm, node);
  TransferDepend(batchnorm, graph, bn_infer);
  return bn_infer;
}
}  // namespace opt
}  // namespace mindspore
