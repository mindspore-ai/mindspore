/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/optimizer/batch_norm_silu_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "ops/sequence_ops.h"
#include "ops/nn_optimizer_ops.h"
#include "ops/nn_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "kernel/graph_kernel_info.h"
#include "ops/op_name.h"

namespace mindspore {
namespace opt {
const BaseRef BatchNormSiluFusion::DefinePattern() const {
  VectorRef batch_norm = VectorRef({prim::kPrimBatchNorm, x_, scale_, bias_, mean_, var_, umonad_});
  VectorRef tuple_get = VectorRef({prim::kPrimTupleGetItem, batch_norm, index_});
  VectorRef silu = VectorRef({prim::kPrimSiLU, tuple_get});
  return silu;
}

const AnfNodePtr BatchNormSiluFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto tuple_get_item = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(tuple_get_item);

  // Only fuse output[0] of BatchNorm with SiLU
  size_t output_index = common::AnfAlgo::GetTupleGetItemOutIndex(utils::cast<CNodePtr>(tuple_get_item));
  if (output_index != 0) {
    return nullptr;
  }

  auto batch_norm = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_get_item), 0);
  MS_EXCEPTION_IF_NULL(batch_norm);

  auto x = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex0);
  auto scale = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex1);
  auto bias = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex2);
  auto mean = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex3);
  auto var = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex4);
  auto umonad = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex5);

  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(scale);
  MS_EXCEPTION_IF_NULL(bias);
  MS_EXCEPTION_IF_NULL(mean);
  MS_EXCEPTION_IF_NULL(var);
  MS_EXCEPTION_IF_NULL(umonad);

  auto prim = std::make_shared<Primitive>(kBatchNormWithActivationOpName);
  MS_EXCEPTION_IF_NULL(prim);
  prim->AddAttr(mindspore::ops::kActivationType, MakeValue(static_cast<int64_t>(mindspore::ActivationType::SWISH)));
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x, scale, bias, mean, var, umonad};
  auto fused_batch_norm_with_silu = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fused_batch_norm_with_silu);

  std::vector<TypeId> outputs_type;
  std::vector<BaseShapePtr> outputs_shape;
  auto output_num = AnfAlgo::GetOutputTensorNum(batch_norm);
  for (size_t i = 0; i < output_num; i++) {
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(batch_norm, i));
    outputs_shape.push_back(AnfAlgo::GetOutputDetailShape(batch_norm, i));
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape(outputs_type, outputs_shape, fused_batch_norm_with_silu.get());
  common::AnfAlgo::CopyNodeAttrs(batch_norm, fused_batch_norm_with_silu);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (!manager->Replace(batch_norm, fused_batch_norm_with_silu)) {
    MS_LOG(EXCEPTION) << "manager replace node failed in batchnorm silu fusion.";
  }
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kGPUDevice);
  MS_EXCEPTION_IF_NULL(kernel_info_setter);
  kernel_info_setter->SetKernelInfo(fused_batch_norm_with_silu, KernelType::UNKNOWN_KERNEL_TYPE);
  MS_LOG(INFO) << "Batch norm and silu operation fusion is finish.";
  return tuple_get_item;
}
}  // namespace opt
}  // namespace mindspore
