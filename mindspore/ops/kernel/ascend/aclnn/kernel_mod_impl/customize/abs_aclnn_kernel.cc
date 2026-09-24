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

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/abs_aclnn_kernel.h"

#include <memory>
#include <vector>

#include "kernel/host/view/view_utils.h"
#include "view/real_view_strides_calc.h"

namespace mindspore::kernel::abs {
namespace {
constexpr size_t kAbsIntermediateTensorNum = 3;
}  // namespace

void AbsAscend::SetRealImagView(const KernelTensor *input, const KernelTensorPtr &view, bool is_real) const {
  auto [storage_info, data_type] = ops::RealImagViewStridesCalc(GetOldTensorInfo(input), input->dtype_id(), is_real);
  view->SetType(std::make_shared<TensorType>(TypeIdToType(data_type)));
  view->SetShape(std::make_shared<abstract::TensorShape>(storage_info->shape));
  view->set_tensor_storage_info(storage_info);
}

void AbsAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  const auto input_type = inputs[kIndex0]->dtype_id();
  use_complex_fallback_ = input_type == kNumberTypeComplex64 || input_type == kNumberTypeComplex128;
  if (!use_complex_fallback_) {
    GetWorkspaceForResize(inputs[kIndex0], outputs[kIndex0]);
    return;
  }

  real_view_ = inputs[kIndex0]->CloneKernelTensor();
  imag_view_ = inputs[kIndex0]->CloneKernelTensor();
  SetRealImagView(inputs[kIndex0], real_view_, true);
  SetRealImagView(inputs[kIndex0], imag_view_, false);

  real_square_ = outputs[kIndex0]->CloneKernelTensor();
  imag_square_ = outputs[kIndex0]->CloneKernelTensor();
  square_sum_ = outputs[kIndex0]->CloneKernelTensor();
  MAKE_SCALAR(1, kNumberTypeInt64, one_);

  GetWorkspaceForResizeSquareReal(real_view_.get(), real_view_.get(), real_square_.get());
  GetWorkspaceForResizeSquareImag(imag_view_.get(), imag_view_.get(), imag_square_.get());
  GetWorkspaceForResizeAdd(real_square_.get(), imag_square_.get(), one_, square_sum_.get());
  GetWorkspaceForResizeSqrt(square_sum_.get(), outputs[kIndex0]);

  const auto intermediate_size =
    ops::CalOutputSize(outputs[kIndex0]->GetShapeVector(), abstract::TypeIdSize(outputs[kIndex0]->dtype_id()));
  for (size_t i = 0; i < kAbsIntermediateTensorNum; ++i) {
    workspace_size_list_.emplace_back(intermediate_size);
  }
}

bool AbsAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (!use_complex_fallback_) {
    RunOp(stream_ptr, workspace, inputs[kIndex0], outputs[kIndex0]);
    return true;
  }

  real_view_->set_device_ptr(inputs[kIndex0]->device_ptr());
  imag_view_->set_device_ptr(inputs[kIndex0]->device_ptr());
  const auto intermediate_offset = workspace.size() - kAbsIntermediateTensorNum;
  real_square_->set_device_ptr(workspace[intermediate_offset]->device_ptr());
  imag_square_->set_device_ptr(workspace[intermediate_offset + 1]->device_ptr());
  square_sum_->set_device_ptr(workspace[intermediate_offset + 2]->device_ptr());

  RunOpSquareReal(stream_ptr, workspace, real_view_.get(), real_view_.get(), real_square_.get());
  RunOpSquareImag(stream_ptr, workspace, imag_view_.get(), imag_view_.get(), imag_square_.get());
  RunOpAdd(stream_ptr, workspace, real_square_.get(), imag_square_.get(), one_, square_sum_.get());
  RunOpSqrt(stream_ptr, workspace, square_sum_.get(), outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Abs, AbsAscend);
}  // namespace mindspore::kernel::abs
