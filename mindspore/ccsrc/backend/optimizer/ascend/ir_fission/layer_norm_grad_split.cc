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
#include "backend/optimizer/ascend/ir_fission/layer_norm_grad_split.h"

#include <memory>
#include <vector>

#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_info.h"
#include "ir/primitive.h"
#include "utils/utils.h"
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
  auto layer_norm_x_backprop = graph->NewCNode(layer_norm_x_backprop_inputs);
  MS_EXCEPTION_IF_NULL(layer_norm_x_backprop);
  layer_norm_x_backprop->set_scope(layer_norm_grad->scope());
  auto types = {AnfAlgo::GetOutputInferDataType(layer_norm_grad, 0), kNumberTypeFloat32};
  auto shapes = {AnfAlgo::GetOutputDetailShape(layer_norm_grad, 0),
                 AnfAlgo::GetPrevNodeOutputDetailShape(layer_norm_grad, 1)};
  if (is_dynamic) {
    AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), layer_norm_x_backprop);
    AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), layer_norm_x_backprop);
    AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), layer_norm_x_backprop);
  }
  AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, layer_norm_x_backprop.get());

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
  auto layer_norm_beta_gamma_backprop = graph->NewCNode(layer_norm_beta_gamma_backprop_inputs);
  MS_EXCEPTION_IF_NULL(layer_norm_beta_gamma_backprop);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  layer_norm_beta_gamma_backprop->set_kernel_info(kernel_info);
  layer_norm_beta_gamma_backprop->set_scope(layer_norm_grad->scope());
  if (is_dynamic) {
    AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), layer_norm_beta_gamma_backprop);
    AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), layer_norm_beta_gamma_backprop);
  }
  auto types = {AnfAlgo::GetOutputInferDataType(layer_norm_grad, kLayerNormGradOutputGammaIndex),
                AnfAlgo::GetOutputInferDataType(layer_norm_grad, kLayerNormGradOutputBetaIndex)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(layer_norm_grad, kLayerNormGradOutputGammaIndex),
                 AnfAlgo::GetOutputDetailShape(layer_norm_grad, kLayerNormGradOutputBetaIndex)};
  AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, layer_norm_beta_gamma_backprop.get());

  // get device shape of LayerNormGrad's 5th Input, and convert it to attr
  std::vector<size_t> shape_gamma =
    AnfAlgo::GetPrevNodeOutputInferShape(layer_norm_grad, kLayerNormGradInputGammaIndex);
  AnfAlgo::SetNodeAttr(kAttrShapeGamma, MakeValue(opt::Convert2Long(shape_gamma)), layer_norm_beta_gamma_backprop);

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
  auto cnode = node->cast<CNodePtr>();
  if (AnfAlgo::GetInputTensorNum(cnode) != kLayerNormGradInputTensorNum) {
    return nullptr;
  }
  bool is_dynamic_shape = AnfAlgo::IsDynamicShape(cnode);
  // create layer_norm_x_backprop
  std::vector<AnfNodePtr> layer_norm_x_backprop_outputs;
  CreateOutputsOfLayerNormXBackpropV2(graph, cnode, &layer_norm_x_backprop_outputs, is_dynamic_shape);
  if (layer_norm_x_backprop_outputs.size() != kLayerNormXBackpropV2OutputNum) {
    MS_LOG(EXCEPTION) << "layer_norm_grad_outputs has wrong size"
                      << " trace: " << trace::DumpSourceLines(node);
  }

  // create layer_norm_beta_gamma_backprop
  std::vector<AnfNodePtr> layer_norm_beta_gamma_backprop_outputs;
  CreateOutputsOfLayerNormBetaGammaBackpropV2(graph, cnode, layer_norm_x_backprop_outputs[1],
                                              &layer_norm_beta_gamma_backprop_outputs, is_dynamic_shape);
  if (layer_norm_beta_gamma_backprop_outputs.size() != kLayerNormBetaGammaBackpropOutputNum) {
    MS_LOG(EXCEPTION) << "layer_norm_beta_gamma_outputs has wrong size"
                      << " trace: " << trace::DumpSourceLines(node);
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
