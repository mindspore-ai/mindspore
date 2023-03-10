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
#include "plugin/device/ascend/optimizer/ir_fission/single_batch_norm_fission.h"
#include <vector>
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kBatchNormRealInputNum = 3;
}  // namespace

AnfNodePtr SingleBatchNormFission::CreateBNTrainingReduce(const FuncGraphPtr &func_graph, const AnfNodePtr &bn) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn);
  auto bn_cnode = bn->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(bn_cnode);
  if (bn_cnode->inputs().size() < kBatchNormRealInputNum + 1) {
    MS_LOG(EXCEPTION) << "The input size of node " + bn_cnode->DebugString() + " is less than "
                      << (kBatchNormRealInputNum + 1) << trace::DumpSourceLines(bn);
  }
  std::vector<AnfNodePtr> bn_training_reduce_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingReduceOpName)), bn_cnode->input(1)};
  auto bn_training_reduce = NewCNode(bn_training_reduce_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(bn_training_reduce);

  // set abstract
  auto bn_input1 = bn_cnode->input(kDim2);
  MS_EXCEPTION_IF_NULL(bn_input1);
  AbstractBasePtrList abstract_list{bn_input1->abstract(), bn_input1->abstract()};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  bn_training_reduce->set_abstract(abstract_tuple);
  bn_training_reduce->set_scope(bn->scope());
  return bn_training_reduce;
}

AnfNodePtr SingleBatchNormFission::CreateBNTrainingUpdateV3(
  const FuncGraphPtr &func_graph, const AnfNodePtr &bn,
  const std::vector<AnfNodePtr> &bn_training_reduce_outputs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn);
  auto bn_cnode = bn->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(bn_cnode);
  if (bn_cnode->inputs().size() < kBatchNormRealInputNum + 1) {
    MS_LOG(EXCEPTION) << "The input size of node " + bn_cnode->DebugString() + " is less than "
                      << (kBatchNormRealInputNum + 1) << trace::DumpSourceLines(bn);
  }
  if (bn_training_reduce_outputs.size() != kBNTrainingReduceOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node bn_training_reduce must be " << kBNTrainingReduceOutputNum
                      << ", but it is " << bn_training_reduce_outputs.size() << trace::DumpSourceLines(bn);
  }
  std::vector<AnfNodePtr> bn_training_update_v3_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateV3OpName)),
    bn_cnode->input(kIndex1),
    bn_training_reduce_outputs[kIndex0],
    bn_training_reduce_outputs[kIndex1],
    bn_cnode->input(kIndex2),
    bn_cnode->input(kIndex3)};
  auto bn_training_update_v3 = NewCNode(bn_training_update_v3_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(bn_training_update_v3);

  auto bn_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn->abstract());
  MS_EXCEPTION_IF_NULL(bn_abstract_tuple);
  if (bn_abstract_tuple->elements().size() != kBnOutputNum) {
    MS_LOG(EXCEPTION) << "The abstract size of node bn must be " << kBnOutputNum << ", but it is "
                      << bn_abstract_tuple->elements().size() << trace::DumpSourceLines(bn);
  }
  bn_training_update_v3->set_abstract(bn->abstract());
  bn_training_update_v3->set_scope(bn->scope());
  common::AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn_cnode, bn_training_update_v3);
  return bn_training_update_v3;
}

const BaseRef SingleBatchNormFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimBatchNorm, Xs});
}

const AnfNodePtr SingleBatchNormFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() < kBatchNormRealInputNum + 1) {
    MS_LOG(INFO) << "The input num of BatchNorm less than" << kBatchNormRealInputNum
                 << ". The node should not be changed";
    return nullptr;
  }
  if (!GetBoolAttr(cnode, kAttrIsTraining)) {
    MS_LOG(INFO) << "Attr is_training should be true if do fusion";
    return nullptr;
  }
  AnfNodePtr bn_training_reduce = CreateBNTrainingReduce(func_graph, node);
  std::vector<AnfNodePtr> bn_training_reduce_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, bn_training_reduce, kBNTrainingReduceOutputNum,
                                 &bn_training_reduce_outputs);

  return CreateBNTrainingUpdateV3(func_graph, node, bn_training_reduce_outputs);
}
}  // namespace opt
}  // namespace mindspore
