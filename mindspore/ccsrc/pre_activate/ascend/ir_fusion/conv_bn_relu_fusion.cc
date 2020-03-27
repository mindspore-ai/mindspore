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
#include "pre_activate/ascend/ir_fusion/conv_bn_relu_fusion.h"

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <tuple>

#include "utils/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "common/utils.h"
#include "device/kernel_info.h"

namespace mindspore {
namespace opt {
namespace {
std::tuple<CNodePtr, CNodePtr, CNodePtr> GetPrevNodes(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto relu_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(relu_node);
  if (relu_node->inputs().size() < kReluInputNum) {
    MS_LOG(EXCEPTION) << "relu has wrong input size";
  }
  auto tuple_getitem_anf = relu_node->input(1);
  MS_EXCEPTION_IF_NULL(tuple_getitem_anf);
  auto tuple_getitem = tuple_getitem_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  if (tuple_getitem->inputs().size() < kTupleGetitemInputNum) {
    MS_LOG(EXCEPTION) << "tuple getitem has wrong input size";
  }
  auto bn_node_anf = tuple_getitem->input(1);
  MS_EXCEPTION_IF_NULL(bn_node_anf);
  auto bn_node = bn_node_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(bn_node);
  if (bn_node->inputs().size() < kBnInputNum) {
    MS_LOG(EXCEPTION) << "bn_node has wrong input size";
  }
  auto conv_node_anf = bn_node->input(1);
  MS_EXCEPTION_IF_NULL(conv_node_anf);
  CNodePtr conv_node = conv_node_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(conv_node);
  return std::make_tuple(bn_node, bn_node, conv_node);
}

void CreateOutputsOfBn2Relu(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &conv_bn1_outputs,
                            const CNodePtr &bn_node, const CNodePtr &relu_node,
                            std::vector<AnfNodePtr> *bn2_relu_outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn_node);
  MS_EXCEPTION_IF_NULL(relu_node);
  // The inputs of bn2_relu are from the outputs of conv_bn1 and the 2nd to 5th inputs of bn
  std::vector<AnfNodePtr> bn2_relu_inputs = {NewValueNode(std::make_shared<Primitive>(kBN2ReLUOpName))};
  (void)std::copy(conv_bn1_outputs.begin(), conv_bn1_outputs.end(), std::back_inserter(bn2_relu_inputs));
  for (size_t i = 2; i <= 5; i++) {
    bn2_relu_inputs.push_back(bn_node->input(i));
  }
  auto bn2_relu = func_graph->NewCNode(bn2_relu_inputs);
  MS_EXCEPTION_IF_NULL(bn2_relu);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  bn2_relu->set_kernel_info(kernel_info);
  auto types = {AnfAlgo::GetOutputInferDataType(relu_node, 0), AnfAlgo::GetOutputInferDataType(bn_node, 1),
                AnfAlgo::GetOutputInferDataType(bn_node, 2), AnfAlgo::GetOutputInferDataType(bn_node, 4)};
  auto shapes = {AnfAlgo::GetOutputInferShape(relu_node, 0), AnfAlgo::GetOutputInferShape(bn_node, 1),
                 AnfAlgo::GetOutputInferShape(bn_node, 2), AnfAlgo::GetOutputInferShape(bn_node, 4)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, bn2_relu.get());
  // Set attr for bn2_add_relu
  AnfAlgo::CopyNodeAttrs(bn_node, bn2_relu);
  AnfAlgo::CopyNodeAttr("epsilon", "eps", bn_node, bn2_relu);

  CreateMultipleOutputsOfAnfNode(func_graph, bn2_relu, kBn2ReluOutputNum, bn2_relu_outputs);
}
}  // namespace

const BaseRef ConvBnReluFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr Ys = std::make_shared<SeqVar>();
  VarPtr Z = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(Xs);
  MS_EXCEPTION_IF_NULL(Ys);
  MS_EXCEPTION_IF_NULL(Z);
  return VectorRef(
    {prim::kPrimRelu,
     PatternListType({prim::kPrimTupleGetItem,
                      PatternListType({prim::kPrimFusedBatchNorm, PatternListType({prim::kPrimConv2D, Xs}), Ys}), Z})});
}

const AnfNodePtr ConvBnReluFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  CNodePtr relu_node = nullptr;
  CNodePtr bn_node = nullptr;
  CNodePtr conv_node = nullptr;
  std::tie(relu_node, bn_node, conv_node) = GetPrevNodes(node);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  std::vector<AnfNodePtr> conv_bn1_outputs;
  CreateOutputsOfConvBn1(func_graph, conv_node, bn_node, &conv_bn1_outputs);
  if (conv_bn1_outputs.size() != kConvBn1OutputNum) {
    MS_LOG(EXCEPTION) << "conv_bn1 outputs has wrong size: " << conv_bn1_outputs.size();
  }
  (void)manager->Replace(conv_node, conv_bn1_outputs[0]);

  std::vector<AnfNodePtr> bn2_relu_outputs;
  CreateOutputsOfBn2Relu(func_graph, conv_bn1_outputs, bn_node, relu_node, &bn2_relu_outputs);
  if (bn2_relu_outputs.size() != kBn2ReluOutputNum) {
    MS_LOG(EXCEPTION) << "bn2_relu outputs has wrong size: " << bn2_relu_outputs.size();
  }
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple),
                                            bn2_relu_outputs[0],
                                            bn2_relu_outputs[1],
                                            bn2_relu_outputs[2],
                                            conv_bn1_outputs[2],
                                            bn2_relu_outputs[3]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  (void)manager->Replace(bn_node, make_tuple);
  return bn2_relu_outputs[0];
}
}  // namespace opt
}  // namespace mindspore
