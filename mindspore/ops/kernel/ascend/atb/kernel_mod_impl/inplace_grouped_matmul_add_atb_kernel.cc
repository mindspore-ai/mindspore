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

#include "kernel/ascend/atb/kernel_mod_impl/inplace_grouped_matmul_add_atb_kernel.h"

#include <functional>
#include <numeric>
#include <memory>

#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore {
namespace kernel {
TensorStorageInfoPtr InplaceGroupedMatmulAddATBKernelMod::CreateTensorStorageInfo(
  const KernelTensor *ori_tensor, const std::vector<int64_t> &new_shape) {
  std::vector<int64_t> new_strides(new_shape.size(), 1);
  for (size_t i = new_shape.size() - 1; i > 0; --i) {
    new_strides[i - 1] = new_strides[i] * new_shape[i];
  }
  return std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_shape, new_strides, true);
}

void InplaceGroupedMatmulAddATBKernelMod::SetTensorStorageInfo(const KernelTensorPtr &new_tensor,
                                                               const KernelTensor *ori_tensor) {
  const auto &ori_shape = ori_tensor->GetShapeVector();
  auto size = std::accumulate(ori_shape.begin(), ori_shape.end(), int64_t(1), std::multiplies<int64_t>());
  std::vector<int64_t> new_shape{size / ori_shape.back(), ori_shape.back()};
  new_tensor->SetShapeVector(new_shape);
  auto tensor_storage_info = CreateTensorStorageInfo(ori_tensor, new_shape);
  new_tensor->set_tensor_storage_info(tensor_storage_info);
}

void InplaceGroupedMatmulAddATBKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
#ifndef EXPERIMENT_A5
  atb::infer::GroupedMatmulInplaceAddParam param;
  param.transposeA = true;
  param.transposeB = false;
  uint64_t hash_id = device::ascend::AtbHash(param, op_name_);
  if (hash_id != hash_id_) {
    atb::CreateOperation(param, &op_);
    hash_id_ = hash_id;
  }

  out_tensor_ = inputs[kIndex3]->CloneKernelTensor();
  SetTensorStorageInfo(out_tensor_, inputs[kIndex3]);
  const auto &out_tensor = out_tensor_.get();

  param_setter_.SetIndex({0, 1, 2, 3}, {0})
    .Input(inputs[kIndex0])
    .Input(inputs[kIndex1])
    .Input(inputs[kIndex2])
    .Input(out_tensor)
    .Output(out_tensor);
  UpdateWorkspace(device::ascend::GetWorkSpaceSize(op_, param_setter_.variant_pack, param_setter_.stream));
#endif
}

bool InplaceGroupedMatmulAddATBKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &workspace,
                                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
#ifndef EXPERIMENT_A5
  out_tensor_->set_device_ptr(inputs[kIndex3]->device_ptr());
  const auto &out_tensor = out_tensor_.get();

  const std::vector<KernelTensor *> real_inputs{inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], out_tensor};
  const std::vector<KernelTensor *> real_outputs{out_tensor};

  param_setter_.Update(real_inputs, real_outputs);
  device::ascend::Launch(op_, param_setter_.variant_pack, workspace[0]->device_ptr(), workspace_size_list_, stream_ptr);
#endif
  return true;
}

MS_ATB_KERNEL_FACTORY_REG(InplaceGroupedMatmulAdd, InplaceGroupedMatmulAddATBKernelMod);
}  // namespace kernel
}  // namespace mindspore
