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
#include "plugin/device/gpu/optimizer/relu_v2_pass.h"
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
const size_t kReluV2OutputNum = 2;
const int64_t kBitPerUInt = 32;

CNodePtr GetRelu(const CNodePtr &relu_grad) {
  MS_EXCEPTION_IF_NULL(relu_grad);
  CheckCNodeInputSize(relu_grad, kReluGradInputTensorNum);
  auto relu_anf = relu_grad->input(2);
  MS_EXCEPTION_IF_NULL(relu_anf);
  return relu_anf->cast<CNodePtr>();
}

kernel::KernelBuildInfoPtr GenerateKernelBuildInfo(CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<std::string> inputs_format;
  std::vector<std::string> outputs_format;
  std::vector<TypeId> inputs_type;
  std::vector<TypeId> outputs_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;

  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    inputs_type.push_back(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, input_index));
    inputs_format.push_back(kOpFormat_DEFAULT);
  }
  size_t output_num = AnfAlgo::GetOutputElementNum(node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(node, output_index));
    outputs_format.push_back(kOpFormat_DEFAULT);
  }
  builder.SetInputsDeviceType(inputs_type);
  builder.SetInputsFormat(inputs_format);
  builder.SetOutputsDeviceType(outputs_type);
  builder.SetOutputsFormat(outputs_format);
  return builder.Build();
}

CNodePtr CreateReluV2(const FuncGraphPtr &graph, const CNodePtr &relu) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(relu);
  CheckCNodeInputSize(relu, kReluInputTensorNum);

  auto prim = std::make_shared<Primitive>(kReLUV2OpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), relu->input(1)};
  auto new_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(relu->scope());

  if (common::AnfAlgo::IsDynamicShape(relu)) {
    return nullptr;
  }
  auto output_shape = common::AnfAlgo::GetOutputInferShape(relu, 0);
  auto element_num = std::accumulate(output_shape.begin(), output_shape.end(), int64_t(1), std::multiplies<int64_t>());

  std::vector<int64_t> mask_shape = {(element_num + kBitPerUInt - 1) / kBitPerUInt};
  std::vector<BaseShapePtr> shapes = {AnfAlgo::GetOutputDetailShape(relu, 0),
                                      std::make_shared<abstract::Shape>(mask_shape)};
  auto types = {common::AnfAlgo::GetOutputInferDataType(relu, 0), kNumberTypeUInt32};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, new_node.get());

  auto build_info = GenerateKernelBuildInfo(new_node);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, new_node.get());
  return new_node;
}

CNodePtr CreateReluGradV2(const FuncGraphPtr &graph, const CNodePtr &relu_grad, const AnfNodePtr &second_input) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(relu_grad);
  MS_EXCEPTION_IF_NULL(second_input);

  auto prim = std::make_shared<Primitive>(kReluGradV2OpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), relu_grad->input(1), second_input};
  auto new_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(relu_grad->scope());
  new_node->set_abstract(relu_grad->abstract());

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  size_t output_num = AnfAlgo::GetOutputTensorNum(relu_grad);
  for (size_t i = 0; i < output_num; i++) {
    types.push_back(common::AnfAlgo::GetOutputInferDataType(relu_grad, i));
    shapes.push_back(AnfAlgo::GetOutputDetailShape(relu_grad, i));
  }

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, new_node.get());
  new_node->set_scope(relu_grad->scope());

  auto build_info = GenerateKernelBuildInfo(new_node);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, new_node.get());

  return new_node;
}
}  // namespace

const BaseRef ReluV2Pass::DefinePattern() const {
  VectorRef relu_grad({prim::kPrimReluGrad, dy_, VectorRef({prim::kPrimReLU, x_})});
  return relu_grad;
}

const AnfNodePtr ReluV2Pass::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
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
  auto manage = graph->manager();
  MS_EXCEPTION_IF_NULL(manage);
  manage->Replace(relu, relu_v2_node_outputs[0]);
  return relu_grad_v2;
}
}  // namespace opt
}  // namespace mindspore
