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
#include "plugin/device/ascend/optimizer/ir_fission/layer_norm_grad_split.h"

#include <memory>
#include <vector>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_info.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kLayerNormGradOutputGammaIndex = 1;
constexpr size_t kLayerNormGradOutputBetaIndex = 2;
constexpr size_t kLayerNormGradInputGammaIndex = 4;
}  // namespace

void LayerNormGradSplit::CreateOutputsOfLayerNormXBackpropV2(const FuncGraphPtr &graph, const CNodePtr &layer_norm_grad,
                                                             std::vector<AnfNodePtr> *layer_norm_x_backprop_outputs,
                                                             bool is_dynamic) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(layer_norm_grad);
  MS_EXCEPTION_IF_NULL(layer_norm_x_backprop_outputs);
  auto prim = std::make_shared<Primitive>(kLayerNormXBackpropV2OpName);
  std::vector<AnfNodePtr> layer_norm_x_backprop_inputs = {NewValueNode(prim)};
  for (size_t i = 1; i < layer_norm_grad->inputs().size(); ++i) {
    layer_norm_x_backprop_inputs.push_back(layer_norm_grad->input(i));
  }
  auto layer_norm_x_backprop = NewCNode(layer_norm_x_backprop_inputs, graph);
  MS_EXCEPTION_IF_NULL(layer_norm_x_backprop);
  layer_norm_x_backprop->set_scope(layer_norm_grad->scope());
  auto types = {common::AnfAlgo::GetOutputInferDataType(layer_norm_grad, 0), kNumberTypeFloat32};
  auto shapes = {AnfAlgo::GetOutputDetailShape(layer_norm_grad, 0),
                 AnfAlgo::GetPrevNodeOutputDetailShape(layer_norm_grad, 1)};
  if (is_dynamic) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), layer_norm_x_backprop);
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), layer_norm_x_backprop);
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, layer_norm_x_backprop.get());

  CreateMultipleOutputsOfAnfNode(graph, layer_norm_x_backprop, kLayerNormXBackpropV2OutputNum,
                                 layer_norm_x_backprop_outputs);
}

void LayerNormGradSplit::CreateOutputsOfLayerNormBetaGammaBackpropV2(
  const FuncGraphPtr &graph, const CNodePtr &layer_norm_grad, const AnfNodePtr &res_for_gamma,
  std::vector<AnfNodePtr> *layer_norm_beta_gamma_backprop_outputs, bool is_dynamic) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(layer_norm_grad);
  auto prim = std::make_shared<Primitive>(kLayerNormBetaGammaBackpropV2OpName);
  std::vector<AnfNodePtr> layer_norm_beta_gamma_backprop_inputs = {NewValueNode(prim), layer_norm_grad->input(kIndex2),
                                                                   res_for_gamma};
  auto layer_norm_beta_gamma_backprop = NewCNode(layer_norm_beta_gamma_backprop_inputs, graph);
  MS_EXCEPTION_IF_NULL(layer_norm_beta_gamma_backprop);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  layer_norm_beta_gamma_backprop->set_kernel_info(kernel_info);
  layer_norm_beta_gamma_backprop->set_scope(layer_norm_grad->scope());
  if (is_dynamic) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), layer_norm_beta_gamma_backprop);
  }
  auto types = {common::AnfAlgo::GetOutputInferDataType(layer_norm_grad, kLayerNormGradOutputGammaIndex),
                common::AnfAlgo::GetOutputInferDataType(layer_norm_grad, kLayerNormGradOutputBetaIndex)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(layer_norm_grad, kLayerNormGradOutputGammaIndex),
                 AnfAlgo::GetOutputDetailShape(layer_norm_grad, kLayerNormGradOutputBetaIndex)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, layer_norm_beta_gamma_backprop.get());

  // get device shape of LayerNormGrad's 5th Input, and convert it to attr
  auto shape_gamma = common::AnfAlgo::GetPrevNodeOutputInferShape(layer_norm_grad, kLayerNormGradInputGammaIndex);
  common::AnfAlgo::SetNodeAttr(kAttrShapeGamma, MakeValue(shape_gamma), layer_norm_beta_gamma_backprop);

  CreateMultipleOutputsOfAnfNode(graph, layer_norm_beta_gamma_backprop, kLayerNormBetaGammaBackpropOutputNum,
                                 layer_norm_beta_gamma_backprop_outputs);
}

const BaseRef LayerNormGradSplit::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimLayerNormGrad, Xs});
  return pattern;
}

const AnfNodePtr LayerNormGradSplit::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  // LayerNormXBackpropV2 is not supported in acl.
  if (graph->has_flag(kAttrMutableKernel)) {
    MS_LOG(INFO) << "Skip LayerNormGradSplit for acl op";
    return nullptr;
  }

  auto cnode = node->cast<CNodePtr>();
  if (common::AnfAlgo::GetInputTensorNum(cnode) != kLayerNormGradInputTensorNum) {
    return nullptr;
  }
  bool is_dynamic_shape = common::AnfAlgo::IsDynamicShape(cnode);
  // create layer_norm_x_backprop
  std::vector<AnfNodePtr> layer_norm_x_backprop_outputs;
  CreateOutputsOfLayerNormXBackpropV2(graph, cnode, &layer_norm_x_backprop_outputs, is_dynamic_shape);
  if (layer_norm_x_backprop_outputs.size() != kLayerNormXBackpropV2OutputNum) {
    MS_LOG(EXCEPTION) << "layer_norm_grad_outputs has wrong size" << trace::DumpSourceLines(node);
  }

  // create layer_norm_beta_gamma_backprop
  std::vector<AnfNodePtr> layer_norm_beta_gamma_backprop_outputs;
  CreateOutputsOfLayerNormBetaGammaBackpropV2(graph, cnode, layer_norm_x_backprop_outputs[1],
                                              &layer_norm_beta_gamma_backprop_outputs, is_dynamic_shape);
  if (layer_norm_beta_gamma_backprop_outputs.size() != kLayerNormBetaGammaBackpropOutputNum) {
    MS_LOG(EXCEPTION) << "layer_norm_beta_gamma_outputs has wrong size" << trace::DumpSourceLines(node);
  }

  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple), layer_norm_x_backprop_outputs[0],
                                               layer_norm_beta_gamma_backprop_outputs[0],
                                               layer_norm_beta_gamma_backprop_outputs[1]};
  auto make_tuple = graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
