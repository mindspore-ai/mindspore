/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/cdist_aclnn_kernel.h"

#include <cmath>
#include <limits>
#include <vector>

#include "view/broadcast_to_strides_calc.h"

namespace mindspore {
namespace kernel {
namespace cdist {
namespace {
constexpr float kDefaultPValue = 2.0f;
constexpr int64_t kDefaultComputeMode = 0;
constexpr int64_t kEuclideanComputeMode = 2;
constexpr size_t kCdistInputRank = 3;
constexpr size_t kBatchAxis = 0;

std::vector<int64_t> GetTensorStrideForView(const KernelTensor *tensor) {
  auto storage = tensor->tensor_storage_info();
  if (storage != nullptr) {
    return storage->strides;
  }
  const auto &shape = tensor->GetShapeVector();
  if (shape.empty()) {
    return {};
  }
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    stride *= shape[i];
    strides[i - 1] = stride;
  }
  return strides;
}

void SetBatchBroadcastView(const KernelTensorPtr &kernel_tensor, const ShapeVector &target_shape) {
  auto tensor_storage_info =
    ops::BroadCastToStrideCalc(kernel_tensor->GetShapeVector(), GetTensorStrideForView(kernel_tensor.get()),
                               kernel_tensor->tensor_storage_info(), target_shape)[0];
  kernel_tensor->SetShapeVector(target_shape);
  kernel_tensor->set_tensor_storage_info(tensor_storage_info);
}
}  // namespace

void CdistAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto p_dtype_id = inputs[kIndex2]->dtype_id();
  if (p_dtype_id == kMetaTypeNone) {
    p_value_ = kDefaultPValue;
  } else if (p_dtype_id == kNumberTypeFloat32) {
    p_value_ = inputs[kIndex2]->GetValueWithCheck<float>();
  } else {
    const auto p_value = inputs[kIndex2]->GetValueWithCheck<double>();
    if (p_value > static_cast<double>(std::numeric_limits<float>::max())) {
      MS_LOG(EXCEPTION) << "For Cdist, p should not exceed float32 range, but got " << p_value;
    }
    p_value_ = static_cast<float>(p_value);
  }
  compute_mode_ = (p_value_ == kDefaultPValue) ? kEuclideanComputeMode : kDefaultComputeMode;

  auto input_x = inputs[kIndex0];
  auto input_y = inputs[kIndex1];
  input_x_kernel_tensor_ = nullptr;
  input_y_kernel_tensor_ = nullptr;
  const auto &x_shape = inputs[kIndex0]->GetShapeVector();
  const auto &y_shape = inputs[kIndex1]->GetShapeVector();
  if (x_shape.size() == kCdistInputRank && y_shape.size() == kCdistInputRank &&
      x_shape[kBatchAxis] != y_shape[kBatchAxis]) {
    if (x_shape[kBatchAxis] == 1) {
      input_x_kernel_tensor_ = inputs[kIndex0]->CloneKernelTensor();
      SetBatchBroadcastView(input_x_kernel_tensor_,
                            ShapeVector{y_shape[kBatchAxis], x_shape[kIndex1], x_shape[kIndex2]});
      input_x = input_x_kernel_tensor_.get();
    } else if (y_shape[kBatchAxis] == 1) {
      input_y_kernel_tensor_ = inputs[kIndex1]->CloneKernelTensor();
      SetBatchBroadcastView(input_y_kernel_tensor_,
                            ShapeVector{x_shape[kBatchAxis], y_shape[kIndex1], y_shape[kIndex2]});
      input_y = input_y_kernel_tensor_.get();
    }
  }
  GetWorkspaceForResize(input_x, input_y, p_value_, compute_mode_, outputs[kIndex0]);
}

bool CdistAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto input_x = inputs[kIndex0];
  auto input_y = inputs[kIndex1];
  if (input_x_kernel_tensor_ != nullptr) {
    input_x_kernel_tensor_->set_device_ptr(inputs[kIndex0]->device_ptr());
    input_x = input_x_kernel_tensor_.get();
  }
  if (input_y_kernel_tensor_ != nullptr) {
    input_y_kernel_tensor_->set_device_ptr(inputs[kIndex1]->device_ptr());
    input_y = input_y_kernel_tensor_.get();
  }
  RunOp(stream_ptr, workspace, input_x, input_y, p_value_, compute_mode_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Cdist, CdistAscend);
}  // namespace cdist
}  // namespace kernel
}  // namespace mindspore
