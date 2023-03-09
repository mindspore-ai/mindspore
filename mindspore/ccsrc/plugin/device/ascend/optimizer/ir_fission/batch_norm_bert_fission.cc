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
#include "plugin/device/ascend/optimizer/ir_fission/batch_norm_bert_fission.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
const std::vector<int64_t> kOutputIndex{0, 3, 4, 5};
constexpr size_t kBatchNormRealOutputNum = 3;
constexpr size_t kBatchNormRealInputNum = 3;

bool GetBatchNormOutputs(const FuncGraphPtr &func_graph, const AnfNodePtr &bn, std::vector<AnfNodePtr> *bn_outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn_outputs);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(bn) == manager->node_users().end()) {
    return false;
  }
  size_t output_num = 0;
  for (const auto &node_index : manager->node_users()[bn]) {
    const AnfNodePtr &output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (!IsPrimitiveCNode(output, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto tuple_getiterm_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_getiterm_cnode);
    auto index_node = tuple_getiterm_cnode->input(kInputNodeOutputIndexInTupleGetItem);
    MS_EXCEPTION_IF_NULL(index_node);
    auto value_node = index_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto index = GetValue<int64_t>(value_node->value());
    if (std::find(kOutputIndex.begin(), kOutputIndex.end(), index) == kOutputIndex.end()) {
      return false;
    }
    bn_outputs->push_back(output);
    output_num++;
  }
  return output_num == kBatchNormRealOutputNum;
}
}  // namespace

AnfNodePtr BatchNormBertFission::CreateBNTrainingReduce(const FuncGraphPtr &func_graph, const AnfNodePtr &bn) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn);
  auto bn_cnode = bn->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(bn_cnode);
  if (bn_cnode->inputs().size() < kBatchNormRealInputNum + 1) {
    MS_LOG(EXCEPTION) << "The input size of node " + bn_cnode->DebugString() + " is less than "
                      << (kBatchNormRealInputNum + 1) << trace::DumpSourceLines(bn);
  }
  std::vector<AnfNodePtr> bn_training_reduce_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingReduceOpName)), bn_cnode->input(kIndex1)};
  auto bn_training_reduce = NewCNode(bn_training_reduce_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(bn_training_reduce);
  auto bn_input1 = bn_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(bn_input1);
  auto bn_input2 = bn_cnode->input(kIndex3);
  MS_EXCEPTION_IF_NULL(bn_input2);
  AbstractBasePtrList abstract_list{bn_input1->abstract(), bn_input2->abstract()};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  bn_training_reduce->set_abstract(abstract_tuple);
  bn_training_reduce->set_scope(bn->scope());
  common::AnfAlgo::CopyNodeAttrs(bn, bn_training_reduce);
  return bn_training_reduce;
}

AnfNodePtr BatchNormBertFission::CreateBNTrainingUpdateV2(
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
  std::vector<AnfNodePtr> bn_training_update_v2_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateV2OpName)),
    bn_cnode->input(kIndex1),
    bn_training_reduce_outputs[kIndex0],
    bn_training_reduce_outputs[kIndex1],
    bn_cnode->input(kIndex2),
    bn_cnode->input(kIndex3)};
  auto bn_training_update_v2 = NewCNode(bn_training_update_v2_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(bn_training_update_v2);

  auto bn_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn->abstract());
  MS_EXCEPTION_IF_NULL(bn_abstract_tuple);
  if (bn_abstract_tuple->elements().size() != kBnOutputNum) {
    MS_LOG(EXCEPTION) << "The abstract size of node bn must be " << kBnOutputNum << ", but it is "
                      << bn_abstract_tuple->elements().size() << trace::DumpSourceLines(bn);
  }
  std::vector<AbstractBasePtr> abstract_list{bn_abstract_tuple->elements()[kIndex0],
                                             bn_abstract_tuple->elements()[kIndex3],
                                             bn_abstract_tuple->elements()[kIndex4]};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  bn_training_update_v2->set_abstract(abstract_tuple);
  bn_training_update_v2->set_scope(bn->scope());
  common::AnfAlgo::CopyNodeAttrs(bn, bn_training_update_v2);
  return bn_training_update_v2;
}

const BaseRef BatchNormBertFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimBatchNorm, Xs});
}

const AnfNodePtr BatchNormBertFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> bn_outputs;
  if (!GetBatchNormOutputs(func_graph, node, &bn_outputs)) {
    MS_LOG(INFO) << "The BatchNorm node should only have output 0, 3 and 4. The node should not be changed";
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() != kBatchNormRealInputNum + 1) {
    MS_LOG(INFO) << "The input size of BatchNorm should be " << kBatchNormRealInputNum
                 << ". The node should not be changed";
    return nullptr;
  }
  AnfNodePtr bn_training_reduce = CreateBNTrainingReduce(func_graph, node);
  std::vector<AnfNodePtr> bn_training_reduce_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, bn_training_reduce, kBNTrainingReduceOutputNum,
                                 &bn_training_reduce_outputs);

  AnfNodePtr bn_training_update_v2 = CreateBNTrainingUpdateV2(func_graph, node, bn_training_reduce_outputs);
  std::vector<AnfNodePtr> bn_training_update_v2_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, bn_training_update_v2, kBNTrainingUpdateV2OutputNum,
                                 &bn_training_update_v2_outputs);
  if (bn_training_update_v2_outputs.size() != kBNTrainingUpdateV2OutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node bn_training_reduce must be " << kBNTrainingUpdateV2OutputNum
                      << ", but it is " << bn_training_update_v2_outputs.size() << trace::DumpSourceLines(node);
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  sort(bn_outputs.begin(), bn_outputs.end(), CompareTupleGetitem);
  size_t output_index = 0;
  for (const auto &output : bn_outputs) {
    (void)manager->Replace(output, bn_training_update_v2_outputs[output_index]);
    output_index++;
  }
  // Return the new node.
  return bn_training_update_v2;
}
}  // namespace opt
}  // namespace mindspore
