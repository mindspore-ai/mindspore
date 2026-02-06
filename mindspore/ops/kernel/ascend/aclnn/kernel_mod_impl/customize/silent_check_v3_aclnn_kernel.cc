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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/silent_check_v3_aclnn_kernel.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "include/utils/utils.h"
#include "ir/tensor.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "mindapi/base/shape_vector.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
namespace silent_check_v3 {
namespace {
std::vector<int64_t> GetTensorStride(const KernelTensor *tensor) {
  auto storage = tensor->tensor_storage_info();
  if (storage != nullptr) {
    return storage->strides;
  }
  auto &shape = tensor->GetShapeVector();
  if (shape.empty()) {
    return {};
  }
  std::vector<int64_t> ret(shape.size(), 1);
  int64_t stride = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    stride *= shape[i];
    ret[i - 1] = stride;
  }
  return ret;
}

int64_t GetTensorOffset(const KernelTensor *tensor) {
  auto storage = tensor->tensor_storage_info();
  return storage == nullptr ? 0 : SizeToLong(storage->storage_offset);
}
}  // namespace

void SilentCheckV3Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  c_thresh_l1_ = inputs[kIndex5]->GetValueWithCheck<pyfloat>();
  c_thresh_l2_ = inputs[kIndex6]->GetValueWithCheck<pyfloat>();
  beta1_ = inputs[kIndex7]->GetValueWithCheck<pyfloat>();
  npu_asd_detect_ = inputs[kIndex8]->GetValueWithCheck<int64_t>();
  auto input_grad = inputs[kIndex3];
  dst_size_ = input_grad->GetShapeVector();
  dst_stride_ = GetTensorStride(input_grad);
  dst_offset_ = ShapeVector({GetTensorOffset(input_grad)});
  ClearOpsWorkSpaceList();
  GetWorkspaceForResizeSilentCheckV3(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], input_grad, inputs[kIndex4],
                                     dst_size_, dst_stride_, dst_offset_, c_thresh_l1_, c_thresh_l2_, beta1_,
                                     npu_asd_detect_, outputs[kIndex3]);
  GetWorkspaceForResizeInputGradCopy(outputs[kIndex1], inputs[kIndex3]);
}

bool SilentCheckV3Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOpSilentCheckV3(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3],
                     inputs[kIndex4], dst_size_, dst_stride_, dst_offset_, c_thresh_l1_, c_thresh_l2_, beta1_,
                     npu_asd_detect_, outputs[kIndex3]);
  RunOpInputGradCopy(stream_ptr, workspace, outputs[kIndex1], inputs[kIndex3]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SilentCheckV3, SilentCheckV3Ascend);
}  // namespace silent_check_v3
}  // namespace kernel
}  // namespace mindspore
