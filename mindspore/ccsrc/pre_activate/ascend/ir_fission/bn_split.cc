/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pre_activate/ascend/ir_fission/bn_split.h"

#include <vector>
#include <string>
#include <memory>

#include "utils/utils.h"
#include "utils/context/ms_context.h"
#include "pre_activate/common/helper.h"
#include "device/kernel_info.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
bool CreateOutputsOfBNTrainingReduce(const FuncGraphPtr &graph, const CNodePtr &bn_cnode,
                                     std::vector<AnfNodePtr> *bn_training_reduce_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_cnode);
  if (bn_cnode->inputs().size() != kBnInputNum) {
    MS_LOG(INFO) << "FusedbatchNorm's input size less than " << kBnInputNum << ". " << bn_cnode->DebugString();
    return false;
  }
  std::vector<AnfNodePtr> bn_training_reduce_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingReduceOpName))};
  bn_training_reduce_inputs.push_back(bn_cnode->input(1));
  auto bn_training_reduce = graph->NewCNode(bn_training_reduce_inputs);
  MS_EXCEPTION_IF_NULL(bn_training_reduce);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  bn_training_reduce->set_kernel_info(kernel_info);
  std::vector<size_t> bn_shape_i0 = AnfAlgo::GetPrevNodeOutputInferShape(bn_cnode, 0);
  if (bn_shape_i0.size() < kShape2dDims) {
    MS_LOG(INFO) << "The FusedBatchNorm's first input's shape dims less than " << kShape2dDims;
    return false;
  }
  std::vector<size_t> bn_training_reduce_shape = {bn_shape_i0[1]};
  auto types = {kNumberTypeFloat32, kNumberTypeFloat32};
  auto shapes = {bn_training_reduce_shape, bn_training_reduce_shape};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, bn_training_reduce.get());
  bn_training_reduce->set_scope(bn_cnode->scope());
  AnfAlgo::CopyNodeAttrs(bn_cnode, bn_training_reduce);

  CreateMultipleOutputsOfAnfNode(graph, bn_training_reduce, kBNTrainingReduceOutputNum, bn_training_reduce_outputs);
  return true;
}

AnfNodePtr CreateOutputsOfBNTrainingUpdate(const FuncGraphPtr &graph, const CNodePtr &bn_cnode,
                                           const std::vector<AnfNodePtr> &bn_training_reduce_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_cnode);
  if (bn_cnode->inputs().size() != kBnInputNum) {
    MS_LOG(EXCEPTION) << "BN node has wrong input size";
  }
  if (bn_training_reduce_outputs.size() != kBNTrainingReduceOutputNum) {
    MS_LOG(EXCEPTION) << "BN1 outputs has wrong input size";
  }
  // the inputs of BNTrainingUpdate are from the outputs of BNTrainingReduce and the inputs of BN
  std::vector<AnfNodePtr> bn_training_update_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateOpName))};
  bn_training_update_inputs.push_back(bn_cnode->input(1));
  bn_training_update_inputs.push_back(bn_training_reduce_outputs[0]);
  bn_training_update_inputs.push_back(bn_training_reduce_outputs[1]);
  bn_training_update_inputs.push_back(bn_cnode->input(2));
  bn_training_update_inputs.push_back(bn_cnode->input(3));
  bn_training_update_inputs.push_back(bn_cnode->input(4));
  bn_training_update_inputs.push_back(bn_cnode->input(5));
  auto bn_training_update = graph->NewCNode(bn_training_update_inputs);
  MS_EXCEPTION_IF_NULL(bn_training_update);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  bn_training_update->set_kernel_info(kernel_info);
  bn_training_update->set_abstract(bn_cnode->abstract());
  bn_training_update->set_scope(bn_cnode->scope());
  auto factor = AnfAlgo::GetNodeAttr<float>(bn_cnode, kAttrMomentum);
  AnfAlgo::SetNodeAttr(kAttrFactor, MakeValue<float>(factor), bn_training_update);
  AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn_cnode, bn_training_update);
  AnfAlgo::SetNodeAttr(kAttrIsRef, MakeValue(true), bn_training_update);
  return bn_training_update;
}

AnfNodePtr SplitFusedBatchNormForTBE(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() < kBnInputNum) {
    MS_LOG(INFO) << "op[FusedBatchNorm] has less than " << kBnInputNum << " inputs.";
    return nullptr;
  }
  // Create BNTrainingReduce node and get outputs of BNTrainingReduce
  std::vector<AnfNodePtr> bn_training_reduce_outputs;
  if (!CreateOutputsOfBNTrainingReduce(func_graph, cnode, &bn_training_reduce_outputs)) {
    MS_LOG(WARNING) << "Create BNTrainingReduce fail, quit split";
    return nullptr;
  }
  if (bn_training_reduce_outputs.size() != kBN1OutputNum) {
    MS_LOG(EXCEPTION) << "make outputs of op BNTrainingReduce fail";
  }

  // Create BNTrainingUpdate node
  return CreateOutputsOfBNTrainingUpdate(func_graph, cnode, bn_training_reduce_outputs);
}
}  // namespace

const BaseRef BnSplit::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef({prim::kPrimFusedBatchNorm, Xs});
}

const AnfNodePtr BnSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  return SplitFusedBatchNormForTBE(func_graph, node);
}
}  // namespace opt
}  // namespace mindspore
