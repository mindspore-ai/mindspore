/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/conv_transpose2d_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <memory>
#include <functional>

#include "include/utils/utils.h"
#include "ir/tensor.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace conv_transpose2d {
namespace {
void ExpandParamIfNeeded(std::vector<int64_t> *const param, size_t expect_dim) {
  if (param->size() == kIndex1) {
    param->insert(param->end(), expect_dim - kIndex1, param->at(kIndex0));
  }
}
}  // namespace

TensorStorageInfoPtr ConvTranspose2DAscend::CreateTensorStorageInfo(const KernelTensor *ori_tensor,
                                                                    const std::vector<int64_t> &new_shape) {
  // fetch origin tensor info
  const auto &old_tensor_storage_info = ori_tensor->tensor_storage_info();
  if (old_tensor_storage_info != nullptr) {
    const auto &shape = old_tensor_storage_info->shape;
    const auto &strides = old_tensor_storage_info->strides;
    // create new tensor info
    auto new_strides(strides);
    new_strides.insert(new_strides.begin(), shape[0] * strides[0]);
    return std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_tensor_storage_info->storage_offset,
                                               old_tensor_storage_info->ori_shape, old_tensor_storage_info->ori_strides,
                                               old_tensor_storage_info->is_contiguous);
  }
  std::vector<int64_t> new_strides(new_shape.size(), 1);
  for (size_t i = new_shape.size() - 1; i > 0; --i) {
    new_strides[i - 1] = new_strides[i] * new_shape[i];
  }
  return std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_shape, new_strides, true);
}

void ConvTranspose2DAscend::SetTensorStorageInfo(const KernelTensorPtr &new_tensor, const KernelTensor *ori_tensor) {
  auto shape = ori_tensor->GetShapeVector();
  shape.insert(shape.begin(), 1, 1);
  new_tensor->SetShapeVector(shape);
  auto tensor_storage_info = CreateTensorStorageInfo(ori_tensor, shape);
  new_tensor->set_tensor_storage_info(tensor_storage_info);
}

void ConvTranspose2DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  const auto &weight_shape = inputs[kIndex1]->GetShapeVector();
  auto spatial_len = weight_shape.size() - kIndex2;
  stride_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex3]);
  ExpandParamIfNeeded(&stride_, spatial_len);
  padding_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex4]);
  ExpandParamIfNeeded(&padding_, spatial_len);
  output_padding_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex5]);
  ExpandParamIfNeeded(&output_padding_, spatial_len);
  groups_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex6]);
  dilation_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex7]);
  ExpandParamIfNeeded(&dilation_, spatial_len);
  cube_math_type_ = OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32());
  const auto &input_shape = inputs[kIndex0]->GetShapeVector();
  is_batchify_ = (input_shape.size() == kIndex4);
  // batchfy the input and output tensor
  auto input_tensor = inputs[kIndex0];
  auto output_tensor = outputs[kIndex0];
  if (!is_batchify_) {
    // clone input 0 and expand dim 0
    input_tensor_ = inputs[kIndex0]->CloneKernelTensor();
    SetTensorStorageInfo(input_tensor_, inputs[kIndex0]);
    input_tensor = input_tensor_.get();
    // clone output 0 and expand dim 0
    output_tensor_ = outputs[kIndex0]->CloneKernelTensor();
    SetTensorStorageInfo(output_tensor_, outputs[kIndex0]);
    output_tensor = output_tensor_.get();
  }
  // aclnn phase oen
  GetWorkspaceForResize(input_tensor, inputs[kIndex1], inputs[kIndex2], stride_, padding_, dilation_, transposed_,
                        output_padding_, groups_, output_tensor, cube_math_type_);
}

bool ConvTranspose2DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  // batchfy the input and output tensor
  auto input_tensor = inputs[kIndex0];
  auto output_tensor = outputs[kIndex0];
  if (!is_batchify_) {
    // set input device address
    input_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
    input_tensor = input_tensor_.get();
    // set output device address
    output_tensor_->set_device_ptr(outputs[kIndex0]->device_ptr());
    output_tensor = output_tensor_.get();
  }
  // aclnn phase two
  RunOp(stream_ptr, workspace, input_tensor, inputs[kIndex1], inputs[kIndex2], stride_, padding_, dilation_,
        transposed_, output_padding_, groups_, output_tensor, cube_math_type_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ConvTranspose2D, ConvTranspose2DAscend);
}  // namespace conv_transpose2d
}  // namespace kernel
}  // namespace mindspore
