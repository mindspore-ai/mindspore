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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/addmm_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "view/broadcast_to_strides_calc.h"

namespace mindspore {
namespace kernel {
namespace addmm {
namespace {
void SetBroadcastView(const KernelTensorPtr &kernel_tensor, const ShapeVector &target_shape) {
  const auto &storage_info = kernel_tensor->tensor_storage_info();
  const auto &strides =
    storage_info == nullptr ? ops::GetOriStrides(kernel_tensor->GetShapeVector()) : storage_info->strides;
  auto broadcast_storage =
    ops::BroadCastToStrideCalc(kernel_tensor->GetShapeVector(), strides, storage_info, target_shape)[0];
  kernel_tensor->SetShapeVector(target_shape);
  kernel_tensor->set_tensor_storage_info(broadcast_storage);
}
}  // namespace

void AddmmAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  cube_math_type_ = OpApiUtil::GetCubeMathTypeForAddOps(OpApiUtil::IsAllowMatmulHF32());
  beta_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex3]);
  alpha_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex4]);

  auto input = inputs[kIndex0];
  input_broadcast_ = nullptr;
  const auto &output_shape = outputs[kIndex0]->GetShapeVector();
  if (cube_math_type_ == device::ascend::USE_FP32_ADD && input->GetShapeVector() != output_shape) {
    input_broadcast_ = input->CloneKernelTensor();
    SetBroadcastView(input_broadcast_, output_shape);
    input = input_broadcast_.get();
  }
  GetWorkspaceForResize(input, inputs[kIndex1], inputs[kIndex2], beta_, alpha_, outputs[kIndex0], cube_math_type_);
}

bool AddmmAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto input = inputs[kIndex0];
  if (input_broadcast_ != nullptr) {
    input_broadcast_->set_device_ptr(input->device_ptr());
    input = input_broadcast_.get();
  }
  RunOp(stream_ptr, workspace, input, inputs[kIndex1], inputs[kIndex2], beta_, alpha_, outputs[kIndex0],
        cube_math_type_);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(Addmm, AddmmAclnnKernelMod);
}  // namespace addmm
}  // namespace kernel
}  // namespace mindspore
