/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/gpu/batch_norm_relu_grad_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/gpu/kernel_info_setter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
const BaseRef BatchNormReluGradFusion::DefinePattern() const {
  VectorRef relu_grad = VectorRef({prim::kPrimReluGrad, dy_, y_});
  VectorRef batch_norm_grad =
    VectorRef({prim::kPrimBatchNormGrad, relu_grad, x_, scale_, save_mean_, save_var_, reserve_});
  return batch_norm_grad;
}

const AnfNodePtr BatchNormReluGradFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto is_train = AnfAlgo::GetCNodePrimitive(node)->GetAttr("is_training");
  MS_EXCEPTION_IF_NULL(is_train);
  if (!GetValue<bool>(is_train)) {
    return nullptr;
  }
  auto format_attr = AnfAlgo::GetCNodePrimitive(node)->GetAttr("format");
  MS_EXCEPTION_IF_NULL(format_attr);
  auto format = GetValue<std::string>(format_attr);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return nullptr;
  }
  if (AnfAlgo::GetInputFormat(node, 0) != kOpFormat_NHWC && format != "NHWC") {
    return nullptr;
  }
  auto shape = AnfAlgo::GetInputDeviceShape(node, 0);
  if (shape.back() % kBNChannelMultipleFactor != 0) {
    return nullptr;
  }

  auto relu_grad = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(relu_grad);

  auto outlist = GetRealNodeUsedList(graph, relu_grad);
  if (outlist->size() >= 2) {
    return nullptr;
  }

  auto dy = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(relu_grad), 0);
  MS_EXCEPTION_IF_NULL(dy);
  auto y = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(relu_grad), 1);
  MS_EXCEPTION_IF_NULL(y);
  auto x = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 1);
  MS_EXCEPTION_IF_NULL(x);
  auto scale = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 2);
  MS_EXCEPTION_IF_NULL(scale);
  auto save_mean = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 3);
  MS_EXCEPTION_IF_NULL(save_mean);
  auto save_var = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 4);
  MS_EXCEPTION_IF_NULL(save_var);
  auto reserve = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 5);
  MS_EXCEPTION_IF_NULL(reserve);
  auto batch_norm = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(save_mean), 0);
  MS_EXCEPTION_IF_NULL(batch_norm);
  auto bias = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), 2);
  MS_EXCEPTION_IF_NULL(bias);

  auto prim = std::make_shared<Primitive>(kBatchNormGradWithActivation);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), dy, x, scale, save_mean, save_var, reserve, bias, y};
  auto fused_batch_norm_grad_with_relu = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fused_batch_norm_grad_with_relu);

  std::vector<TypeId> outputs_type;
  std::vector<std::vector<size_t>> outputs_shape;
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; i++) {
    outputs_type.push_back(AnfAlgo::GetOutputInferDataType(node, i));
    outputs_shape.push_back(AnfAlgo::GetOutputInferShape(node, i));
  }
  AnfAlgo::SetOutputInferTypeAndShape(outputs_type, outputs_shape, fused_batch_norm_grad_with_relu.get());
  AnfAlgo::CopyNodeAttrs(node, fused_batch_norm_grad_with_relu);
  device::gpu::SetKernelInfo(fused_batch_norm_grad_with_relu);
  return fused_batch_norm_grad_with_relu;
}
}  // namespace opt
}  // namespace mindspore
