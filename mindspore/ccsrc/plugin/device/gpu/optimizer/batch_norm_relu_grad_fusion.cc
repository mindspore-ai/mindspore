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
#include "plugin/device/gpu/optimizer/batch_norm_relu_grad_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "utils/ms_context.h"
#include "kernel/graph_kernel_info.h"

namespace mindspore {
namespace opt {
const BaseRef BatchNormReluGradFusion::DefinePattern() const {
  VectorRef relu_grad = VectorRef({prim::kPrimReluGrad, dy_, y_});
  VectorRef batch_norm_grad =
    VectorRef({prim::kPrimBatchNormGrad, relu_grad, x_, scale_, save_mean_, save_var_, reserve_});
  return batch_norm_grad;
}

const AnfNodePtr BatchNormReluGradFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto is_train = common::AnfAlgo::GetCNodePrimitive(node)->GetAttr("is_training");
  MS_EXCEPTION_IF_NULL(is_train);
  if (!GetValue<bool>(is_train)) {
    return nullptr;
  }
  auto format_attr = common::AnfAlgo::GetCNodePrimitive(node)->GetAttr("format");
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

  auto relu_grad = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(relu_grad);

  auto outlist = GetRealNodeUsedList(graph, relu_grad);
  const size_t node_user_num_upper_bound = 2;
  if (outlist->size() >= node_user_num_upper_bound) {
    return nullptr;
  }

  auto dy = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(relu_grad), kIndex0);
  MS_EXCEPTION_IF_NULL(dy);
  auto y = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(relu_grad), kIndex1);
  MS_EXCEPTION_IF_NULL(y);
  auto x = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex1);
  MS_EXCEPTION_IF_NULL(x);
  auto scale = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex2);
  MS_EXCEPTION_IF_NULL(scale);
  auto save_mean = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex3);
  MS_EXCEPTION_IF_NULL(save_mean);
  auto save_var = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex4);
  MS_EXCEPTION_IF_NULL(save_var);
  auto reserve = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex5);
  MS_EXCEPTION_IF_NULL(reserve);
  auto batch_norm = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(save_mean), kIndex0);
  MS_EXCEPTION_IF_NULL(batch_norm);
  auto bias = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex2);
  MS_EXCEPTION_IF_NULL(bias);

  auto prim = std::make_shared<Primitive>(kBatchNormGradWithActivation);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), dy, x, scale, save_mean, save_var, reserve, bias, y};
  auto fused_batch_norm_grad_with_relu = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fused_batch_norm_grad_with_relu);

  std::vector<TypeId> outputs_type;
  std::vector<BaseShapePtr> outputs_shape;
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; i++) {
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
    outputs_shape.push_back(AnfAlgo::GetOutputDetailShape(node, i));
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape(outputs_type, outputs_shape, fused_batch_norm_grad_with_relu.get());
  common::AnfAlgo::CopyNodeAttrs(node, fused_batch_norm_grad_with_relu);
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kGPUDevice);
  kernel_info_setter->SetKernelInfo(fused_batch_norm_grad_with_relu, KernelType::UNKNOWN_KERNEL_TYPE);
  return fused_batch_norm_grad_with_relu;
}
}  // namespace opt
}  // namespace mindspore
