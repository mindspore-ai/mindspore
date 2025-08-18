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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/conv3d_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace conv3d_ext {
namespace {
void ExpandParamIfNeeded(std::vector<int64_t> *const param, size_t expect_dim) {
  if (param->size() == kIndex1) {
    param->insert(param->end(), expect_dim - kIndex1, param->at(kIndex0));
  }
}
}  // namespace
std::vector<int64_t> Conv3DExtAscend::GetOriStrides(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> ret(shape.size(), 1);
  int64_t strides = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides *= shape[i];
    ret[i - 1] = strides;
  }
  return ret;
}

TensorStorageInfoPtr Conv3DExtAscend::CreateTensorStorageInfoPtr(const std::vector<int64_t> &new_shape,
                                                                 const TensorStorageInfoPtr &old_tensor_storage_info) {
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

  size_t offset = 0;
  const std::vector<int64_t> expand_shape_ori = new_shape;
  const std::vector<int64_t> expand_shape_new = new_shape;
  auto expand_stride_ori = GetOriStrides(expand_shape_ori);
  auto expand_stride_new = expand_stride_ori;
  return std::make_shared<TensorStorageInfo>(expand_shape_new, expand_stride_new, offset, expand_shape_ori,
                                             expand_stride_ori, true);
}

template <typename T>
void Conv3DExtAscend::SetTensorStorageInfo(T kernel_tensor, ShapeVector new_shape) {
  const auto &old_storage_info = kernel_tensor->tensor_storage_info();
  TensorStorageInfoPtr tensor_storage_info = CreateTensorStorageInfoPtr(new_shape, old_storage_info);
  kernel_tensor->SetShapeVector(new_shape);
  kernel_tensor->set_tensor_storage_info(tensor_storage_info);
}

template <typename T>
void Conv3DExtAscend::SetBackTensorStorageInfo(T kernel_tensor, ShapeVector new_shape) {
  kernel_tensor->SetShapeVector(new_shape);
  kernel_tensor->set_tensor_storage_info(nullptr);
}

void Conv3DExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  const auto &weight_shape = inputs[kIndex1]->GetShapeVector();
  auto spatial_len = weight_shape.size() - kIndex2;
  stride_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex3]);
  ExpandParamIfNeeded(&stride_, spatial_len);
  padding_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex4]);
  ExpandParamIfNeeded(&padding_, spatial_len);
  dilation_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex5]);
  ExpandParamIfNeeded(&dilation_, spatial_len);
  groups_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex6]);

  auto in_shape = inputs[kIndex0]->GetShapeVector();
  _is_batchify = (in_shape.size() == weight_shape.size());
  if (_is_batchify) {
    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], stride_, padding_, dilation_, transposed_,
                          output_padding_, groups_, outputs[kIndex0],
                          OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));
  } else {
    in_shape.insert(in_shape.begin(), 1);
    input_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();
    SetTensorStorageInfo<std::shared_ptr<KernelTensor>>(input_kernel_tensor_, in_shape);

    auto out_shape = outputs[kIndex0]->GetShapeVector();
    expand_out_shape_ = out_shape;
    expand_out_shape_.insert(expand_out_shape_.begin(), 1);
    output_kernel_tensor_ = outputs[kIndex0]->CloneKernelTensor();
    output_kernel_tensor_->SetShapeVector(expand_out_shape_);
    GetWorkspaceForResize(input_kernel_tensor_.get(), inputs[kIndex1], inputs[kIndex2], stride_, padding_, dilation_,
                          transposed_, output_padding_, groups_, output_kernel_tensor_.get(),
                          OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));
  }
}

bool Conv3DExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (_is_batchify) {
    RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], stride_, padding_, dilation_,
          transposed_, output_padding_, groups_, outputs[kIndex0],
          OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));
  } else {
    input_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
    output_kernel_tensor_->set_device_ptr(outputs[kIndex0]->device_ptr());
    RunOp(stream_ptr, workspace, input_kernel_tensor_.get(), inputs[kIndex1], inputs[kIndex2], stride_, padding_,
          dilation_, transposed_, output_padding_, groups_, output_kernel_tensor_.get(),
          OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32()));
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Conv3DExt, Conv3DExtAscend);
}  // namespace conv3d_ext
}  // namespace kernel
}  // namespace mindspore
