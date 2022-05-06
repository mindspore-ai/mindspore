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
#include "plugin/device/ascend/optimizer/ir_fusion/deformable_offsets_grad_fusion.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAxisH = 2;
constexpr size_t kAxisW = 3;
constexpr size_t kAxisC = 1;
constexpr size_t kDeformableOffsetsGradInputNum = 4;
constexpr size_t kChannel = 3;
}  // namespace

ValueNodePtr DeformableOffsetsGradFusion::CreateHelperNode(
  const FuncGraphPtr &func_graph, const AnfNodePtr &node, const ShapeVector &offset_shape,
  const std::vector<int64_t> &kernel_sizes, const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
  const std::vector<int64_t> &dilations, const size_t axis_h, const size_t axis_w, const size_t axis_c) const {
  int64_t h_out = offset_shape[axis_h];
  int64_t w_out = offset_shape[axis_w];
  int64_t kernel_size_h = kernel_sizes[0];
  int64_t kernel_size_w = kernel_sizes[1];
  int64_t stride_h = strides[axis_h];
  int64_t stride_w = strides[axis_w];
  int64_t dilation_h = dilations[axis_h];
  int64_t dilation_w = dilations[axis_w];
  size_t group = offset_shape[axis_c] / (kChannel * kernel_size_h * kernel_size_w);
  int64_t pad_top = pads[0];
  int64_t pad_left = pads[axis_w];
  int64_t h_index;
  int64_t w_index;
  ShapeVector out_shape = {1, offset_shape[1], offset_shape[2], offset_shape[3]};
  tensor::TensorPtr helper_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, out_shape);
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kFloat32);
  tensor::DeviceInfo device_info{kOpFormat_NHWC, tensor_type, kOpFormat_NHWC};
  helper_tensor->set_device_info(device_info);
  auto tensor_data = reinterpret_cast<float *>(helper_tensor->data_c());
  for (int64_t h = 0; h < h_out; ++h) {
    for (int64_t w = 0; w < w_out; ++w) {
      for (size_t g = 0; g < group; ++g) {
        for (int64_t k_h = 0; k_h < kernel_size_h; ++k_h) {
          for (int64_t k_w = 0; k_w < kernel_size_w; ++k_w) {
            w_index = static_cast<int64_t>(h * w_out * kChannel * group * kernel_size_h * kernel_size_w +
                                           w * kChannel * group * kernel_size_h * kernel_size_w +
                                           0 * group * kernel_size_h * kernel_size_w +
                                           g * kernel_size_h * kernel_size_w + k_h * kernel_size_w + k_w);
            h_index = static_cast<int64_t>(h * w_out * kChannel * group * kernel_size_h * kernel_size_w +
                                           w * kChannel * group * kernel_size_h * kernel_size_w +
                                           1 * group * kernel_size_h * kernel_size_w +
                                           g * kernel_size_h * kernel_size_w + k_h * kernel_size_w + k_w);
            float w_val = static_cast<float>(w * stride_w - pad_left + k_w * dilation_w);
            float h_val = static_cast<float>(h * stride_h - pad_top + k_h * dilation_h);
            tensor_data[w_index] = w_val;
            tensor_data[h_index] = h_val;
          }
        }
      }
    }
  }
  AbstractBasePtr x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat, out_shape);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto assist_value_node = kernel_graph->NewValueNode(x_abstract, helper_tensor);
  kernel_graph->AddValueNodeToGraph(assist_value_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {out_shape}, assist_value_node.get());
  return assist_value_node;
}

const BaseRef DeformableOffsetsGradFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimDeformableOffsetsGrad, Xs});
}

const AnfNodePtr DeformableOffsetsGradFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto deformable_offsets_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(deformable_offsets_grad_cnode);
  size_t origin_input_size = deformable_offsets_grad_cnode->inputs().size();
  if (origin_input_size <= kDeformableOffsetsGradInputNum) {
    MS_LOG(INFO) << "The node " << deformable_offsets_grad_cnode->DebugString() << " is not equal to "
                 << kDeformableOffsetsGradInputNum << " inputs";
  }
  auto pads = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(deformable_offsets_grad_cnode, kAttrPads);
  auto stride = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(deformable_offsets_grad_cnode, kAttrStrides);
  auto dialation = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(deformable_offsets_grad_cnode, kAttrDilations);
  auto kernel_size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(deformable_offsets_grad_cnode, kAttrKsize);
  auto offset_shape = common::AnfAlgo::GetOutputInferShape(deformable_offsets_grad_cnode->inputs()[kIndex3], 0);
  std::vector<AnfNodePtr> new_inputs{
    NewValueNode(std::make_shared<Primitive>(prim::kPrimDeformableOffsetsGrad->name()))};
  auto assist_const = CreateHelperNode(func_graph, deformable_offsets_grad_cnode, offset_shape, kernel_size, stride,
                                       pads, dialation, kAxisH, kAxisW, kAxisC);
  (void)new_inputs.insert(new_inputs.end(), deformable_offsets_grad_cnode->inputs().begin() + 1,
                          deformable_offsets_grad_cnode->inputs().end());
  new_inputs.push_back(assist_const);
  auto new_cnode = NewCNode(new_inputs, func_graph);
  new_cnode->set_abstract(deformable_offsets_grad_cnode->abstract());
  new_cnode->set_scope(deformable_offsets_grad_cnode->scope());
  common::AnfAlgo::CopyNodeAttrs(deformable_offsets_grad_cnode, new_cnode);
  common::AnfAlgo::SetNodeAttr(kAttrDataFormat, MakeValue("NHWC"), new_cnode);
  if (kernel_graph != nullptr) {
    kernel_graph->AddValueNodeToGraph(assist_const);
    MS_LOG(INFO) << "Add assist tensor for DeformableOffsets op success.";
  }
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
