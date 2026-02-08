/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/dense_aclnn_kernel.h"
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "view/transpose_ext_view_strides_calc.h"
#include "kernel/host/view/view_utils.h"

namespace mindspore {
namespace kernel {
namespace dense {
void DenseAclnnKernelMod::SetFlatternNdLinearTensorStorageInfo(const KernelTensorPtr &new_tensor,
                                                               const int &new_shape_first, const ShapeVector &shape) {
  auto new_shape_second = shape[shape.size() - 1];
  auto new_shape = ShapeVector{new_shape_first, new_shape_second};

  new_tensor->SetShapeVector(new_shape);

  size_t offset = 0;
  auto shape_ori = new_shape;
  auto shape_new = new_shape;
  const std::vector<int64_t> strides_new = {new_shape_second, 1};
  const std::vector<int64_t> strides_ori = strides_new;
  TensorStorageInfoPtr tensor_storage_info =
    std::make_shared<TensorStorageInfo>(shape_new, strides_new, offset, shape_ori, strides_ori, true);

  new_tensor->set_tensor_storage_info(tensor_storage_info);
}

std::vector<int64_t> DenseAclnnKernelMod::TransposeWeight(const KernelTensor *w_tensor,
                                                          const std::vector<int64_t> &w_shape,
                                                          const KernelTensorPtr &w_t_tensor) {
  ShapeVector w_t_shape{w_shape};
  if (w_t_shape.size() >= kIndex2) {
    // set shape
    auto w_rank = w_shape.size();
    w_t_shape[w_rank - kIndex1] = w_shape[w_rank - kIndex2];
    w_t_shape[w_rank - kIndex2] = w_shape[w_rank - kIndex1];
    w_t_tensor->SetShape(std::make_shared<abstract::TensorShape>(w_t_shape));
    // calculate new storage infoF
    ops::OldTensorInfoPtr old_info = GetOldTensorInfo(w_tensor);
    auto new_info = ops::TransposeExtViewStridesCalc(old_info->old_shape, old_info->old_strides,
                                                     w_tensor->tensor_storage_info(), -1, -2);
    w_t_tensor->set_tensor_storage_info(new_info[0]);
  }
  return w_t_shape;
}

void DenseAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  const auto &x_shape = inputs[kIndex0]->GetShapeVector();
  const auto &w_shape = inputs[kIndex1]->GetShapeVector();
  auto w_rank = w_shape.size();
  x_rank_ = x_shape.size();

  // 1. Generate Transpose weight, the weight'rank greater than 1
  w_t_tensor_ = inputs[kIndex1]->CloneKernelTensor();
  auto w_t_shape = TransposeWeight(inputs[kIndex1], w_shape, w_t_tensor_);
  auto w_t = w_t_tensor_.get();

  // 2. Go through Different scenarios based on input'rank and bias'rank.
  auto bias = inputs[kIndex2];
  is_bias_none_ = bias->GetType()->type_id() == kMetaTypeNone;

  cube_math_type_ = OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowMatmulHF32());

  if (is_bias_none_) {
    GetWorkspaceForResizeMatmul(inputs[kIndex0], w_t, outputs[kIndex0], cube_math_type_);
  } else {
    const auto &bias_shape = bias->GetShapeVector();
    bias_rank_ = bias_shape.size();

    if (x_rank_ == kIndex2) {
      GetWorkspaceForResizeAddmm(inputs[kIndex2], inputs[kIndex0], w_t, one_, one_, outputs[kIndex0], cube_math_type_);
    } else if (bias_rank_ == kIndex1 || x_rank_ == kIndex3) {
      input_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();
      output_kernel_tensor_ = outputs[kIndex0]->CloneKernelTensor();

      int input_reshape_size = 1;
      if (x_rank_ > kIndex1) {
        input_reshape_size =
          std::accumulate(x_shape.begin(), x_shape.end() - 1, static_cast<int64_t>(1), std::multiplies<int64_t>());
      }
      // Generate reshape Input 2D(Input_shape_i0 * Input_shape_i1 * ... Input_shape_ix-1, Input_shape_ix)
      SetFlatternNdLinearTensorStorageInfo(input_kernel_tensor_, input_reshape_size, x_shape);
      // Generate reshape Output 2D(Input_shape_i0 * Input_shape_i1 * ... Input_shape_ix-1, W_t_shape_last)
      SetFlatternNdLinearTensorStorageInfo(output_kernel_tensor_, input_reshape_size, w_t_shape);

      GetWorkspaceForResizeAddmm(inputs[kIndex2], input_kernel_tensor_.get(), w_t, one_, one_,
                                 output_kernel_tensor_.get(), cube_math_type_);
    } else {
      ShapeVector matmul_shape{x_shape};
      matmul_shape[x_rank_ - 1] = w_t_shape[w_rank - 1];
      // When x and w are 1D, a scalar tensor(0D) is generated.
      if (x_rank_ == 1 && w_rank == 1) {
        matmul_shape = {};
      }

      matmul_tensor_.SetType(inputs[kIndex0]->GetType());
      matmul_tensor_.SetShape(std::make_shared<abstract::TensorShape>(matmul_shape));

      GetWorkspaceForResizeMatmul(inputs[kIndex0], w_t, &matmul_tensor_, cube_math_type_);
      GetWorkspaceForResizeAdd(&matmul_tensor_, bias, one_, outputs[kIndex0]);

      const auto &matmul_output_size =
        ops::CalOutputSize(matmul_tensor_.GetShapeVector(), mindspore::abstract::TypeIdSize(matmul_tensor_.dtype_id()));
      workspace_size_list_.emplace_back(matmul_output_size);
    }
  }
}

bool DenseAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  // 1. Generate Transpose weight, the weight'rank greater than 1
  w_t_tensor_->set_device_ptr(inputs[kIndex1]->device_ptr());
  auto w_t = w_t_tensor_.get();

  // 2. Go through Different scenarios based on input'rank and bias'rank.
  if (is_bias_none_) {
    RunOpMatmul(stream_ptr, workspace, inputs[kIndex0], w_t, outputs[kIndex0], cube_math_type_);
  } else {
    if (x_rank_ == kIndex2) {
      RunOpAddmm(stream_ptr, workspace, inputs[kIndex2], inputs[kIndex0], w_t, one_, one_, outputs[kIndex0],
                 cube_math_type_);
    } else if (bias_rank_ == kIndex1 || x_rank_ == kIndex3) {
      input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
      output_kernel_tensor_->set_device_ptr(outputs[kIndex0]->device_ptr());
      RunOpAddmm(stream_ptr, workspace, inputs[kIndex2], input_kernel_tensor_.get(), w_t, one_, one_,
                 output_kernel_tensor_.get(), cube_math_type_);
    } else {
      // If matmul_tensor_ has value, it's the penultimate one.
      auto bias = inputs[kIndex2];
      matmul_tensor_.set_device_ptr(workspace[workspace.size() - kIndex1]->device_ptr());
      RunOpMatmul(stream_ptr, workspace, inputs[kIndex0], w_t, &matmul_tensor_, cube_math_type_);
      RunOpAdd(stream_ptr, workspace, &matmul_tensor_, bias, one_, outputs[kIndex0]);
    }
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Dense, DenseAclnnKernelMod);
}  // namespace dense
}  // namespace kernel
}  // namespace mindspore
