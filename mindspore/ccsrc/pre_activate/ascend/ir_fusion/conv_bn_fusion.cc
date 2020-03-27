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

#include "pre_activate/ascend/ir_fusion/conv_bn_fusion.h"
#include <memory>
#include <vector>
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_info.h"

namespace mindspore {
namespace opt {
const BaseRef ConvBnFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  VarPtr Ys = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Ys);
  return VectorRef({prim::kPrimFusedBatchNorm, PatternListType({prim::kPrimConv2D, Xs}), Ys});
}

const AnfNodePtr ConvBnFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The bn node is expected to be a cnode";
  }
  auto bn_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(bn_cnode);
  if (bn_cnode->inputs().size() < kVariance + 1) {
    auto op_name = AnfAlgo::GetCNodeName(bn_cnode);
    MS_LOG(EXCEPTION) << "op[" << op_name << "] has less than " << kVariance + 1 << " inputs.";
  }
  AnfNodePtr conv_node = bn_cnode->input(kX);
  MS_EXCEPTION_IF_NULL(conv_node);
  if (!conv_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The conv node is expected to be a cnode";
  }
  auto conv_cnode = conv_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(conv_cnode);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  // Create conv_bn1 node and get outputs of conv_bn1
  std::vector<AnfNodePtr> conv_bn1_outputs;
  CreateOutputsOfConvBn1(func_graph, conv_cnode, bn_cnode, &conv_bn1_outputs);
  if (conv_bn1_outputs.size() != kConvBn1OutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node conv_bn1 must be " << kConvBn1OutputNum << ", but it is "
                      << conv_bn1_outputs.size();
  }
  // Replace conv_node with the output 0 of conv_bn1 directly because the conv node may be used as input by other
  (void)manager->Replace(conv_node, conv_bn1_outputs[kData]);

  // Create bn2 node and get outputs of bn2
  std::vector<AnfNodePtr> bn2_outputs;
  std::vector<AnfNodePtr> bn1_outputs = {conv_bn1_outputs[2], conv_bn1_outputs[1]};
  CreateOutputsOfFusedBn2(func_graph, bn1_outputs, bn_cnode, &bn2_outputs);
  if (bn2_outputs.size() != kBN2OutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node fusedbn2 must be " << kBN2OutputNum << ", but it is "
                      << bn2_outputs.size();
  }

  // Create bn3 node and get outputs of bn3
  std::vector<AnfNodePtr> bn3_outputs;
  CreateOutputsOfFusedBn3(func_graph, conv_bn1_outputs[0], bn1_outputs, bn2_outputs, bn_cnode, &bn3_outputs);

  if (bn3_outputs.size() != kBN3OutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node fusedbn3 must be " << kBN3OutputNum << ", but it is "
                      << bn3_outputs.size();
  }

  // Return a make_tuple to replace the bn node here, the outputs are from node bn2 and conv_bn1.
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple),
                                            bn3_outputs[0],
                                            bn2_outputs[1],
                                            bn2_outputs[2],
                                            conv_bn1_outputs[2],
                                            bn2_outputs[0]};

  return func_graph->NewCNode(make_tuple_inputs);
}
}  // namespace opt
}  // namespace mindspore
