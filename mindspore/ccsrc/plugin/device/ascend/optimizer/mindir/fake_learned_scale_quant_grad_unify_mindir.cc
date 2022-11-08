/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/mindir/fake_learned_scale_quant_grad_unify_mindir.h"

#include <vector>
#include <memory>

#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "backend/common/optimizer/helper.h"
#include "runtime/device/kernel_info.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
void FakeLearnedScaleQuantPerLayerGradUnifyMindIR::CreateOutputsOfLSQPerLayerGradD(
  const FuncGraphPtr &graph, const CNodePtr &lsq_perlayer_grad_node,
  std::vector<AnfNodePtr> *const lsq_perlayer_grad_d_outputs) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(lsq_perlayer_grad_node);
  const auto &lsq_perlayer_grad_inputs = lsq_perlayer_grad_node->inputs();
  if (lsq_perlayer_grad_inputs.size() < kFakeLearnedScaleQuantGradInputNum) {
    MS_LOG(EXCEPTION) << "Lsq_perlayer_grad_node has wrong inputs size, should be not less than "
                      << kFakeLearnedScaleQuantGradInputNum << ", but got " << lsq_perlayer_grad_inputs.size()
                      << trace::DumpSourceLines(lsq_perlayer_grad_node);
  }
  std::vector<AnfNodePtr> lsq_perlayer_grad_d_inputs = {
    NewValueNode(std::make_shared<Primitive>(kFakeLearnedScaleQuantPerLayerGradDOpName)),
    lsq_perlayer_grad_inputs[kIndex1], lsq_perlayer_grad_inputs[kIndex2], lsq_perlayer_grad_inputs[kIndex3],
    lsq_perlayer_grad_inputs[kIndex4]};
  auto lsq_perlayer_grad_d = NewCNode(lsq_perlayer_grad_d_inputs, graph);
  MS_EXCEPTION_IF_NULL(lsq_perlayer_grad_d);
  lsq_perlayer_grad_d->set_scope(lsq_perlayer_grad_node->scope());

  auto types = {common::AnfAlgo::GetOutputInferDataType(lsq_perlayer_grad_node, 0UL),
                common::AnfAlgo::GetOutputInferDataType(lsq_perlayer_grad_node, 0UL)};
  auto shapes = {common::AnfAlgo::GetOutputDetailShape(lsq_perlayer_grad_node, 0UL),
                 common::AnfAlgo::GetOutputDetailShape(lsq_perlayer_grad_node, 0UL)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, lsq_perlayer_grad_d.get());

  common::AnfAlgo::CopyNodeAttr(kAttrNeg_trunc, lsq_perlayer_grad_node, lsq_perlayer_grad_d);
  CreateMultipleOutputsOfAnfNode(graph, lsq_perlayer_grad_d, kFakeLearnedScaleQuantGradDOutputNum,
                                 lsq_perlayer_grad_d_outputs);
}

void FakeLearnedScaleQuantPerLayerGradUnifyMindIR::CreateOutputsOfLSQPerLayerReduceGrad(
  const FuncGraphPtr &graph, const CNodePtr &lsq_perlayer_grad_node,
  const std::vector<AnfNodePtr> &lsq_perlayer_grad_d_outputs,
  std::vector<AnfNodePtr> *const lsq_perlayer_reduce_grad_outputs) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(lsq_perlayer_grad_node);
  MS_EXCEPTION_IF_NULL(lsq_perlayer_reduce_grad_outputs);
  const auto &lsq_perlayer_grad_inputs = lsq_perlayer_grad_node->inputs();
  if (lsq_perlayer_grad_inputs.size() < kFakeLearnedScaleQuantGradInputNum) {
    MS_LOG(EXCEPTION) << "Lsq_perlayer_grad_node has wrong inputs size, should be not less than "
                      << kFakeLearnedScaleQuantGradInputNum << ", but got " << lsq_perlayer_grad_inputs.size()
                      << trace::DumpSourceLines(lsq_perlayer_grad_node);
  }
  if (lsq_perlayer_grad_d_outputs.size() != kFakeLearnedScaleQuantGradDOutputNum) {
    MS_LOG(EXCEPTION) << "Lsq_perlayer_grad_d_outputs has wrong inputs size, should be "
                      << kFakeLearnedScaleQuantGradDOutputNum << ", but got " << lsq_perlayer_grad_d_outputs.size()
                      << trace::DumpSourceLines(lsq_perlayer_grad_node);
  }
  std::vector<AnfNodePtr> lsq_perlayer_reduce_grad_inputs = {
    NewValueNode(std::make_shared<Primitive>(kFakeLearnedScaleQuantPerLayerGradDReduceOpName)),
    lsq_perlayer_grad_d_outputs[kIndex1]};
  auto lsq_perlayer_reduce_grad = NewCNode(lsq_perlayer_reduce_grad_inputs, graph);
  MS_EXCEPTION_IF_NULL(lsq_perlayer_reduce_grad);
  lsq_perlayer_reduce_grad->set_scope(lsq_perlayer_grad_node->scope());

  auto types = {common::AnfAlgo::GetOutputInferDataType(lsq_perlayer_grad_node, 1UL)};
  auto shapes = {common::AnfAlgo::GetOutputDetailShape(lsq_perlayer_grad_node, 1UL)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, lsq_perlayer_reduce_grad.get());

  (*lsq_perlayer_reduce_grad_outputs).push_back(lsq_perlayer_reduce_grad);
}

void FakeLearnedScaleQuantPerChannelGradUnifyMindIR::CreateOutputsOfLSQPerChannelGradD(
  const FuncGraphPtr &graph, const CNodePtr &lsq_perchannel_grad_node,
  std::vector<AnfNodePtr> *const lsq_perchannel_grad_d_outputs) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(lsq_perchannel_grad_node);
  const auto &lsq_perchannel_grad_inputs = lsq_perchannel_grad_node->inputs();
  if (lsq_perchannel_grad_inputs.size() < kFakeLearnedScaleQuantGradInputNum) {
    MS_LOG(EXCEPTION) << "Lsq_perchannel_grad_node has wrong inputs size, should be not less than "
                      << kFakeLearnedScaleQuantGradInputNum << ", but got " << lsq_perchannel_grad_inputs.size()
                      << trace::DumpSourceLines(lsq_perchannel_grad_node);
  }
  std::vector<AnfNodePtr> lsq_perchannel_grad_d_inputs = {
    NewValueNode(std::make_shared<Primitive>(kFakeLearnedScaleQuantPerChannelGradDOpName)),
    lsq_perchannel_grad_inputs[kIndex1], lsq_perchannel_grad_inputs[kIndex2], lsq_perchannel_grad_inputs[kIndex3],
    lsq_perchannel_grad_inputs[kIndex4]};
  auto lsq_perchannel_grad_d = NewCNode(lsq_perchannel_grad_d_inputs, graph);
  MS_EXCEPTION_IF_NULL(lsq_perchannel_grad_d);
  lsq_perchannel_grad_d->set_scope(lsq_perchannel_grad_node->scope());

  auto types = {common::AnfAlgo::GetOutputInferDataType(lsq_perchannel_grad_node, 0UL),
                common::AnfAlgo::GetOutputInferDataType(lsq_perchannel_grad_node, 0UL)};
  auto shapes = {common::AnfAlgo::GetOutputDetailShape(lsq_perchannel_grad_node, 0UL),
                 common::AnfAlgo::GetOutputDetailShape(lsq_perchannel_grad_node, 0UL)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, lsq_perchannel_grad_d.get());

  common::AnfAlgo::CopyNodeAttr(kAttrNeg_trunc, lsq_perchannel_grad_node, lsq_perchannel_grad_d);
  common::AnfAlgo::CopyNodeAttr(kAttrChannelAxis, lsq_perchannel_grad_node, lsq_perchannel_grad_d);
  CreateMultipleOutputsOfAnfNode(graph, lsq_perchannel_grad_d, kFakeLearnedScaleQuantGradDOutputNum,
                                 lsq_perchannel_grad_d_outputs);
}

void FakeLearnedScaleQuantPerChannelGradUnifyMindIR::CreateOutputsOfLSQPerChannelReduceGrad(
  const FuncGraphPtr &graph, const CNodePtr &lsq_perchannel_grad_node,
  const std::vector<AnfNodePtr> &lsq_perchannel_grad_d_outputs,
  std::vector<AnfNodePtr> *const lsq_perchannel_reduce_grad_outputs) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(lsq_perchannel_grad_node);
  MS_EXCEPTION_IF_NULL(lsq_perchannel_reduce_grad_outputs);
  const auto &lsq_perchannel_grad_inputs = lsq_perchannel_grad_node->inputs();
  if (lsq_perchannel_grad_inputs.size() < kFakeLearnedScaleQuantGradInputNum) {
    MS_LOG(EXCEPTION) << "Lsq_perchannel_grad_node has wrong inputs size, should be not less than "
                      << kFakeLearnedScaleQuantGradInputNum << ", but got " << lsq_perchannel_grad_inputs.size()
                      << trace::DumpSourceLines(lsq_perchannel_grad_node);
  }
  if (lsq_perchannel_grad_d_outputs.size() != kFakeLearnedScaleQuantGradDOutputNum) {
    MS_LOG(EXCEPTION) << "Lsq_perchannel_grad_d_outputs has wrong inputs size, should be "
                      << kFakeLearnedScaleQuantGradDOutputNum << ", but got " << lsq_perchannel_grad_inputs.size()
                      << trace::DumpSourceLines(lsq_perchannel_grad_node);
  }
  std::vector<AnfNodePtr> lsq_perchannel_reduce_grad_inputs = {
    NewValueNode(std::make_shared<Primitive>(kFakeLearnedScaleQuantPerChannelGradDReduceOpName)),
    lsq_perchannel_grad_d_outputs[kIndex1]};
  auto lsq_perchannel_reduce_grad = NewCNode(lsq_perchannel_reduce_grad_inputs, graph);
  MS_EXCEPTION_IF_NULL(lsq_perchannel_reduce_grad);
  lsq_perchannel_reduce_grad->set_scope(lsq_perchannel_grad_node->scope());

  auto types = {common::AnfAlgo::GetOutputInferDataType(lsq_perchannel_grad_node, 1UL)};
  auto shapes = {common::AnfAlgo::GetOutputDetailShape(lsq_perchannel_grad_node, 1UL)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, lsq_perchannel_reduce_grad.get());
  common::AnfAlgo::CopyNodeAttr(kAttrChannelAxis, lsq_perchannel_grad_node, lsq_perchannel_reduce_grad);
  (*lsq_perchannel_reduce_grad_outputs).push_back(lsq_perchannel_reduce_grad);
}

const BaseRef FakeLearnedScaleQuantPerLayerGradUnifyMindIR::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kFakeLearnedScaleQuantPerLayerGradOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr FakeLearnedScaleQuantPerLayerGradUnifyMindIR::Process(const FuncGraphPtr &func_graph,
                                                                       const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);

  std::vector<AnfNodePtr> lsq_perlayer_grad_d_outputs;
  CreateOutputsOfLSQPerLayerGradD(func_graph, cnode, &lsq_perlayer_grad_d_outputs);
  if (lsq_perlayer_grad_d_outputs.size() != kFakeLearnedScaleQuantGradOutputNum) {
    MS_LOG(EXCEPTION) << "Fake_learned_scale_quant_perlayer_grad_d_outputs has wrong inputs size, should be "
                      << kFakeLearnedScaleQuantGradOutputNum << ", but got " << lsq_perlayer_grad_d_outputs.size()
                      << trace::DumpSourceLines(node);
  }

  std::vector<AnfNodePtr> lsq_perlayer_reduce_grad_outputs;
  CreateOutputsOfLSQPerLayerReduceGrad(func_graph, cnode, lsq_perlayer_grad_d_outputs,
                                       &lsq_perlayer_reduce_grad_outputs);
  if (lsq_perlayer_reduce_grad_outputs.size() != kSingleOutputNum) {
    MS_LOG(EXCEPTION) << "Fake_learned_scale_quant_perlayer_reduce_grad_outputs has wrong inputs size, should be "
                      << kSingleOutputNum << ", but got " << lsq_perlayer_reduce_grad_outputs.size()
                      << trace::DumpSourceLines(node);
  }

  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), lsq_perlayer_grad_d_outputs[0],
                                               lsq_perlayer_reduce_grad_outputs[0]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}

const BaseRef FakeLearnedScaleQuantPerChannelGradUnifyMindIR::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kFakeLearnedScaleQuantPerChannelGradOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr FakeLearnedScaleQuantPerChannelGradUnifyMindIR::Process(const FuncGraphPtr &func_graph,
                                                                         const AnfNodePtr &node,
                                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);

  std::vector<AnfNodePtr> lsq_perchannel_grad_d_outputs;
  CreateOutputsOfLSQPerChannelGradD(func_graph, cnode, &lsq_perchannel_grad_d_outputs);
  if (lsq_perchannel_grad_d_outputs.size() != kFakeLearnedScaleQuantGradOutputNum) {
    MS_LOG(EXCEPTION) << "Fake_learned_scale_quant_perchannel_grad_d_outputs has wrong inputs size, should be "
                      << kFakeLearnedScaleQuantGradOutputNum << ", but got " << lsq_perchannel_grad_d_outputs.size()
                      << trace::DumpSourceLines(node);
  }

  std::vector<AnfNodePtr> lsq_perchannel_reduce_grad_outputs;
  CreateOutputsOfLSQPerChannelReduceGrad(func_graph, cnode, lsq_perchannel_grad_d_outputs,
                                         &lsq_perchannel_reduce_grad_outputs);
  if (lsq_perchannel_reduce_grad_outputs.size() != kSingleOutputNum) {
    MS_LOG(EXCEPTION) << "Fake_learned_scale_quant_perchannel_reduce_grad_outputs has wrong inputs size, should be "
                      << kSingleOutputNum << ", but got " << lsq_perchannel_reduce_grad_outputs.size()
                      << trace::DumpSourceLines(node);
  }

  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), lsq_perchannel_grad_d_outputs[0],
                                               lsq_perchannel_reduce_grad_outputs[0]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
