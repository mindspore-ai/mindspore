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
#include "plugin/device/gpu/optimizer/post_batch_norm_add_relu_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "kernel/graph_kernel_info.h"

namespace mindspore {
namespace opt {
const BaseRef PostBatchNormAddReluFusion::DefinePattern() const {
  VectorRef batch_norm = VectorRef({prim::kPrimBatchNorm, x_, scale_, bias_, mean_, var_, umonad_});
  VectorRef tuple_get_item = VectorRef({prim::kPrimTupleGetItem, batch_norm, index_});
  VectorRef tensor_add = VectorRef({prim::kPrimAdd, z_, tuple_get_item});
  VectorRef relu = VectorRef({prim::kPrimReLU, tensor_add});
  return relu;
}

const AnfNodePtr PostBatchNormAddReluFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(tensor_add);
  auto tuple_get_item = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tensor_add), 1);
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  auto batch_norm = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_get_item), 0);
  MS_EXCEPTION_IF_NULL(batch_norm);
  auto is_train = common::AnfAlgo::GetCNodePrimitive(batch_norm)->GetAttr("is_training");
  MS_EXCEPTION_IF_NULL(is_train);
  if (!GetValue<bool>(is_train)) {
    return nullptr;
  }
  auto format_attr = common::AnfAlgo::GetCNodePrimitive(batch_norm)->GetAttr("format");
  MS_EXCEPTION_IF_NULL(format_attr);
  auto format = GetValue<std::string>(format_attr);
  if (AnfAlgo::GetInputFormat(batch_norm, 0) != kOpFormat_NHWC && format != "NHWC") {
    return nullptr;
  }
  auto shape = AnfAlgo::GetInputDeviceShape(batch_norm, 0);
  if (shape.back() % kBNChannelMultipleFactor != 0) {
    return nullptr;
  }

  auto x = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex0);
  auto scale = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex1);
  auto bias = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex2);
  auto mean = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex3);
  auto var = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex4);
  auto umonad = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex5);
  auto z = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tensor_add), kIndex0);

  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(scale);
  MS_EXCEPTION_IF_NULL(bias);
  MS_EXCEPTION_IF_NULL(mean);
  MS_EXCEPTION_IF_NULL(var);
  MS_EXCEPTION_IF_NULL(umonad);
  MS_EXCEPTION_IF_NULL(z);

  auto prim = std::make_shared<Primitive>(kBatchNormWithAddAndActivation);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x, scale, bias, mean, var, z, umonad};
  auto fused_batch_norm_with_add_relu = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fused_batch_norm_with_add_relu);

  std::vector<TypeId> outputs_type;
  std::vector<BaseShapePtr> outputs_shape;
  auto output_num = AnfAlgo::GetOutputTensorNum(batch_norm);
  for (size_t i = 0; i < output_num; i++) {
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(batch_norm, i));
    outputs_shape.push_back(AnfAlgo::GetOutputDetailShape(batch_norm, i));
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape(outputs_type, outputs_shape, fused_batch_norm_with_add_relu.get());
  common::AnfAlgo::CopyNodeAttrs(batch_norm, fused_batch_norm_with_add_relu);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->Replace(batch_norm, fused_batch_norm_with_add_relu);
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kGPUDevice);
  MS_EXCEPTION_IF_NULL(kernel_info_setter);
  kernel_info_setter->SetKernelInfo(fused_batch_norm_with_add_relu, KernelType::UNKNOWN_KERNEL_TYPE);
  return tuple_get_item;
}
}  // namespace opt
}  // namespace mindspore
