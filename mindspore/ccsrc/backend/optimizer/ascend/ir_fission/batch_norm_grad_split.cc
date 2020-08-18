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
#include "backend/optimizer/ascend/ir_fission/batch_norm_grad_split.h"

#include <vector>
#include <memory>

#include "utils/utils.h"
#include "utils/ms_context.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
void CreateOutputsOfUpdateGrad(const FuncGraphPtr &graph, const CNodePtr &bn_grad_node,
                               std::vector<AnfNodePtr> *bn_update_grad_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_grad_node);
  const auto &bn_grad_inputs = bn_grad_node->inputs();
  if (bn_grad_inputs.size() < kBNGradInputNum) {
    MS_LOG(EXCEPTION) << "BNGrad has wrong inputs size";
  }
  std::vector<AnfNodePtr> bn_update_grad_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingUpdateGradOpName)), bn_grad_inputs[1], bn_grad_inputs[2],
    bn_grad_inputs[4], bn_grad_inputs[5]};
  auto bn_update_grad = graph->NewCNode(bn_update_grad_inputs);
  MS_EXCEPTION_IF_NULL(bn_update_grad);
  bn_update_grad->set_kernel_info(std::make_shared<device::KernelInfo>());
  bn_update_grad->set_scope(bn_grad_node->scope());

  auto types = {AnfAlgo::GetOutputInferDataType(bn_grad_node, 1), AnfAlgo::GetOutputInferDataType(bn_grad_node, 2)};
  auto shapes = {AnfAlgo::GetOutputInferShape(bn_grad_node, 1), AnfAlgo::GetOutputInferShape(bn_grad_node, 2)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, bn_update_grad.get());

  AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn_grad_node, bn_update_grad);
  CreateMultipleOutputsOfAnfNode(graph, bn_update_grad, kBNTrainingUpdateGradOutputNum, bn_update_grad_outputs);
}

void CreateOutputsOfReduceGrad(const FuncGraphPtr &graph, const CNodePtr &bn_grad_node,
                               const std::vector<AnfNodePtr> &bn_update_grad_outputs,
                               std::vector<AnfNodePtr> *bn_reduce_grad_outputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(bn_grad_node);
  MS_EXCEPTION_IF_NULL(bn_reduce_grad_outputs);
  const auto &bn_grad_inputs = bn_grad_node->inputs();
  if (bn_grad_inputs.size() < kBNGradInputNum) {
    MS_LOG(EXCEPTION) << "BNGrad has wrong inputs size";
  }
  if (bn_update_grad_outputs.size() != kBNTrainingUpdateGradOutputNum) {
    MS_LOG(EXCEPTION) << "BNTrainingReduceGrad_outputs has wrong size";
  }
  std::vector<AnfNodePtr> bn_reduce_grad_inputs = {
    NewValueNode(std::make_shared<Primitive>(kBNTrainingReduceGradOpName)),
    bn_grad_inputs[1],
    bn_grad_inputs[2],
    bn_update_grad_outputs[0],
    bn_update_grad_outputs[1],
    bn_grad_inputs[3],
    bn_grad_inputs[4],
    bn_grad_inputs[5]};
  auto bn_reduce_grad = graph->NewCNode(bn_reduce_grad_inputs);
  MS_EXCEPTION_IF_NULL(bn_reduce_grad);
  bn_reduce_grad->set_kernel_info(std::make_shared<device::KernelInfo>());
  bn_reduce_grad->set_scope(bn_grad_node->scope());

  auto types = {AnfAlgo::GetOutputInferDataType(bn_grad_node, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(bn_grad_node, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, bn_reduce_grad.get());

  AnfAlgo::CopyNodeAttr(kAttrEpsilon, bn_grad_node, bn_reduce_grad);
  (*bn_reduce_grad_outputs).push_back(bn_reduce_grad);
}
}  // namespace
const BaseRef BatchNormGradSplit::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kBatchNormGradOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr BatchNormGradSplit::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  if (!primitive->HasAttr(kAttrIsTraining)) {
    MS_LOG(INFO) << "Op BatchNormGrad must have attrs of is_training";
    return nullptr;
  }
  if (!AnfAlgo::GetNodeAttr<bool>(cnode, kAttrIsTraining)) {
    MS_LOG(INFO) << "is_training must be true";
    return nullptr;
  }

  std::vector<AnfNodePtr> bn_update_grad_outputs;
  CreateOutputsOfUpdateGrad(func_graph, cnode, &bn_update_grad_outputs);
  if (bn_update_grad_outputs.size() != kBNTrainingUpdateGradOutputNum) {
    MS_LOG(EXCEPTION) << "bn_update_grad_outputs has wrong size";
  }

  std::vector<AnfNodePtr> bn_reduce_grad_outputs;
  CreateOutputsOfReduceGrad(func_graph, cnode, bn_update_grad_outputs, &bn_reduce_grad_outputs);
  if (bn_reduce_grad_outputs.size() != kSingleOutputNum) {
    MS_LOG(EXCEPTION) << "bn_reduce_grad_outputs has wrong size";
  }

  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), bn_reduce_grad_outputs[0],
                                               bn_update_grad_outputs[0], bn_update_grad_outputs[1]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
