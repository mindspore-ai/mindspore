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

#include "pre_activate/ascend/ir_fusion/conv_bn_add_relu_fusion.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_info.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kBn2AddReluOutputNum = 4;
enum Bn2AddReluOutput {
  kBn2AddReluOutput = 0,
  kBn2AddReluRunningMean,
  kBn2AddReluRunningVariance,
  kBn2AddReluSaveInvVariance,
};

std::tuple<CNodePtr, CNodePtr, CNodePtr, CNodePtr> GetUsedCNode(const AnfNodePtr &node) {
  auto relu_cnode = CheckAnfNodeIfCNodeAndInputSize(node, kReluInputNum);
  MS_EXCEPTION_IF_NULL(relu_cnode);
  auto add_cnode = CheckAnfNodeIfCNodeAndInputSize(relu_cnode->input(1), kAddInputNum);
  MS_EXCEPTION_IF_NULL(add_cnode);
  auto add_input1_cnode = CheckAnfNodeIfCNodeAndInputSize(add_cnode->input(1), kTupleGetitemInputNum);
  MS_EXCEPTION_IF_NULL(add_input1_cnode);
  auto bn_cnode = CheckAnfNodeIfCNodeAndInputSize(add_input1_cnode->input(1), kBnInputNum);
  MS_EXCEPTION_IF_NULL(bn_cnode);
  auto conv_cnode = CheckAnfNodeIfCNodeAndInputSize(bn_cnode->input(kX), kConvInputNum);

  return std::make_tuple(conv_cnode, bn_cnode, add_cnode, relu_cnode);
}

void CreateOutputsOfBn2AddRelu(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &conv_bn1_outputs,
                               const CNodePtr &bn_node, const CNodePtr &add_node, const CNodePtr &relu_node,
                               std::vector<AnfNodePtr> *bn2_add_relu_outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(add_node);
  MS_EXCEPTION_IF_NULL(relu_node);
  MS_EXCEPTION_IF_NULL(bn_node);
  auto prim = std::make_shared<Primitive>(kBN2AddReluOpName);
  std::vector<AnfNodePtr> bn2_add_relu_inputs = {NewValueNode(prim)};
  // The inputs of bn2_add_relu are from the outputs of conv_bn1, the 2nd input of add, and the 2nd to 5th inputs of bn
  (void)std::copy(conv_bn1_outputs.begin(), conv_bn1_outputs.end(), std::back_inserter(bn2_add_relu_inputs));
  bn2_add_relu_inputs.push_back(add_node->input(2));
  for (size_t i = kX + 1; i <= kVariance; i++) {
    bn2_add_relu_inputs.push_back(bn_node->input(i));
  }
  auto bn2_add_relu_cnode = func_graph->NewCNode(bn2_add_relu_inputs);
  MS_EXCEPTION_IF_NULL(bn2_add_relu_cnode);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  bn2_add_relu_cnode->set_kernel_info(kernel_info);

  // Set attr for bn2_add_relu
  AnfAlgo::CopyNodeAttrs(bn_node, bn2_add_relu_cnode);
  AnfAlgo::CopyNodeAttr("epsilon", "eps", bn_node, bn2_add_relu_cnode);

  // Set abstract of bn2_add_relu
  auto bn_abstract_tuple = dyn_cast<abstract::AbstractTuple>(bn_node->abstract());
  MS_EXCEPTION_IF_NULL(bn_abstract_tuple);
  if (bn_abstract_tuple->elements().size() != kBnOutputNum) {
    MS_LOG(EXCEPTION) << "Abstract tuple size of FusedBatchNorm must be " << kBnOutputNum << ", but it is "
                      << bn_abstract_tuple->elements().size();
  }
  auto relu_abstract = relu_node->abstract();
  MS_EXCEPTION_IF_NULL(relu_abstract);
  // The abstracts of node bn2_add_relu are from the some abstracts of bn and relu nodes.
  AbstractBasePtrList bn2_add_relu_abstract_list{relu_abstract, bn_abstract_tuple->elements()[kRunningMean],
                                                 bn_abstract_tuple->elements()[kRunningVariance],
                                                 bn_abstract_tuple->elements()[kSaveInvVariance]};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(bn2_add_relu_abstract_list);
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  bn2_add_relu_cnode->set_abstract(abstract_tuple);

  CreateMultipleOutputsOfAnfNode(func_graph, bn2_add_relu_cnode, kBn2AddReluOutputNum, bn2_add_relu_outputs);
}
}  // namespace

const BaseRef ConvBnAddReluFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  VarPtr W = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(W);
  VarPtr Ys = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Ys);
  VarPtr Zs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Zs);

  return VectorRef(
    {prim::kPrimRelu,
     PatternListType(
       {prim::kPrimTensorAdd,
        PatternListType({prim::kPrimTupleGetItem,
                         PatternListType({prim::kPrimFusedBatchNorm, PatternListType({prim::kPrimConv2D, Ys}), Zs}),
                         W}),
        X})});
}

const AnfNodePtr ConvBnAddReluFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  CNodePtr conv_cnode = nullptr;
  CNodePtr bn_cnode = nullptr;
  CNodePtr add_cnode = nullptr;
  CNodePtr relu_cnode = nullptr;
  std::tie(conv_cnode, bn_cnode, add_cnode, relu_cnode) = GetUsedCNode(node);
  // Create conv_bn1 node and get outputs of conv_bn1
  std::vector<AnfNodePtr> conv_bn1_outputs;
  CreateOutputsOfConvBn1(func_graph, conv_cnode, bn_cnode, &conv_bn1_outputs);
  if (conv_bn1_outputs.size() != kConvBn1OutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node conv_bn1 must be " << kConvBn1OutputNum << ", but it is "
                      << conv_bn1_outputs.size();
  }
  // Replace conv_node with the output 0 of conv_bn1 directly because the conv node may be used as input by others
  (void)manager->Replace(conv_cnode, conv_bn1_outputs[kData]);

  // Create bn2_add_relu node and get outputs of bn2_add_relu
  std::vector<AnfNodePtr> bn2_add_relu_outputs;
  CreateOutputsOfBn2AddRelu(func_graph, conv_bn1_outputs, bn_cnode, add_cnode, relu_cnode, &bn2_add_relu_outputs);
  if (bn2_add_relu_outputs.size() != kBn2AddReluOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node bn2_add_relu must be " << kBn2AddReluOutputNum << ", but it is "
                      << bn2_add_relu_outputs.size();
  }

  // Create a make_tuple to replace the bn node here, the outputs are from node bn2_add_relu and conv_bn1.
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple),
                                            bn2_add_relu_outputs[kBn2AddReluOutput],
                                            bn2_add_relu_outputs[kBn2AddReluRunningMean],
                                            bn2_add_relu_outputs[kBn2AddReluRunningVariance],
                                            conv_bn1_outputs[kMean],
                                            bn2_add_relu_outputs[kBn2AddReluSaveInvVariance]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  (void)manager->Replace(bn_cnode, make_tuple);
  return bn2_add_relu_outputs[kBn2AddReluOutput];
}
}  // namespace opt
}  // namespace mindspore
