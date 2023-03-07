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
#include "plugin/device/ascend/optimizer/ir_fusion/derelu_fusion.h"
#include <memory>
#include <string>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kReluV2OutputNum = 2;

CNodePtr GetRelu(const CNodePtr &relu_grad) {
  MS_EXCEPTION_IF_NULL(relu_grad);
  CheckCNodeInputSize(relu_grad, kReluGradInputTensorNum);
  auto relu_anf = relu_grad->input(kIndex2);
  MS_EXCEPTION_IF_NULL(relu_anf);
  return relu_anf->cast<CNodePtr>();
}
}  // namespace

CNodePtr DereluFusion::CreateReluV2(const FuncGraphPtr &graph, const CNodePtr &relu) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(relu);
  CheckCNodeInputSize(relu, kReluInputTensorNum);
  constexpr auto kMaskShapeSize = 4;
  auto prim = std::make_shared<Primitive>(kReluV2OpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), relu->input(kIndex1)};
  auto new_node = opt::NewCNode(inputs, graph, {relu});
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(relu->scope());

  // ReluV2's 2rd output is mask whose data type is uint8
  TypeId mask_dtype = kNumberTypeUInt8;
  if (common::AnfAlgo::IsDynamicShape(relu)) {
    return nullptr;
  }
  auto mask_shape = common::AnfAlgo::GetOutputInferShape(relu, 0);
  if (mask_shape.size() != kMaskShapeSize) {
    MS_LOG(DEBUG) << "relu's infer shape size not equal 4";
    return nullptr;
  }
  auto input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(relu, 0);
  constexpr auto kMultiplierInt8 = 31;
  constexpr auto kDivisorInt8 = 32;
  constexpr auto kMultiplier = 15;
  constexpr auto kDivisor = 16;
  constexpr auto kThirdShapeInt8 = 4;
  constexpr auto kThirdShape = 2;

  if (input_dtype == kNumberTypeUInt8 || input_dtype == kNumberTypeInt8) {
    mask_shape[1] = (mask_shape[1] + kMultiplierInt8) / kDivisorInt8;
    mask_shape.push_back(kThirdShapeInt8);
  } else {
    mask_shape[1] = (mask_shape[1] + kMultiplier) / kDivisor;
    mask_shape.push_back(kThirdShape);
  }

  auto types = {common::AnfAlgo::GetOutputInferDataType(relu, 0), mask_dtype};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(relu, 0), mask_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, new_node.get());
  return new_node;
}

CNodePtr DereluFusion::CreateReluGradV2(const FuncGraphPtr &graph, const CNodePtr &relu_grad,
                                        const AnfNodePtr &second_input) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(relu_grad);
  MS_EXCEPTION_IF_NULL(second_input);

  auto prim = std::make_shared<Primitive>(kReluGradV2OpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), relu_grad->input(1), second_input};
  auto new_node = opt::NewCNode(inputs, graph, {relu_grad});
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(relu_grad->scope());
  new_node->set_abstract(relu_grad->abstract());
  return new_node;
}

const BaseRef DereluFusion::DefinePattern() const {
  VarPtr i0 = std::make_shared<Var>();
  VarPtr i1 = std::make_shared<Var>();
  VectorRef relu({prim::kPrimRelu, i1});
  VectorRef relu_grad({prim::kPrimReluGrad, i0, relu});
  return relu_grad;
}

const AnfNodePtr DereluFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto relu_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(relu_grad);
  auto relu = GetRelu(relu_grad);
  MS_EXCEPTION_IF_NULL(relu);

  auto relu_v2 = CreateReluV2(graph, relu);
  if (relu_v2 == nullptr) {
    return nullptr;
  }
  std::vector<AnfNodePtr> relu_v2_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, relu_v2, kReluV2OutputNum, &relu_v2_node_outputs);

  auto relu_grad_v2 = CreateReluGradV2(graph, relu_grad, relu_v2_node_outputs[1]);
  // Add attr mapping from original nodes to fusion nodes
  auto original_names =
    MakeValue<std::vector<std::string>>({relu->fullname_with_scope(), relu_grad->fullname_with_scope()});
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, original_names, relu_v2);
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, original_names, relu_grad_v2);
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpIsMultiop, MakeValue(true), relu_v2);
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpIsMultiop, MakeValue(true), relu_grad_v2);

  auto manage = graph->manager();
  MS_EXCEPTION_IF_NULL(manage);
  (void)manage->Replace(relu, relu_v2_node_outputs[0]);
  return relu_grad_v2;
}
}  // namespace opt
}  // namespace mindspore
