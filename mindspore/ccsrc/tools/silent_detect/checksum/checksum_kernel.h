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
#ifndef MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_CHECKSUM_CHECKSUM_KERNEL_H_
#define MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_CHECKSUM_CHECKSUM_KERNEL_H_

#include <set>
#include <vector>
#include "utils/ms_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace checksum {
using kernel::KernelTensor;
using kernel::KernelTensorPtr;
using mindspore::device::DeviceContext;

class CheckSumKernel {
 public:
  explicit CheckSumKernel(const DeviceContext *device_context) : device_context_(device_context) {
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context_->device_res_manager_);
  }
  static bool IsCheckSumSupported(const std::vector<KernelTensor *> &matmul_inputs,
                                  const std::vector<KernelTensor *> &matmul_outputs);
  KernelTensorPtr LaunchKernelAsync(std::vector<KernelTensor *> matmul_inputs,
                                    std::vector<KernelTensor *> matmul_outputs, const uint32_t stream_id);

 private:
  KernelTensorPtr CalculateError();
  KernelTensorPtr CalculateErrorTotal();
  KernelTensorPtr CalculateC1Trans();
  // std::vector<KernelTensorPtr> GetCErrors();
  // std::vector<KernelTensorPtr> GetDeltas();
  static const std::set<TypeId> supported_dtype_;
  const DeviceContext *device_context_{nullptr};
  uint32_t stream_id_ = kDefaultStreamIndex;
  KernelTensor *tensor_a_{nullptr};
  KernelTensor *tensor_b_{nullptr};
  KernelTensor *tensor_trans_a_{nullptr};
  KernelTensor *tensor_trans_b_{nullptr};
  KernelTensor *tensor_c_{nullptr};
  ShapeVector shape_a_;
  ShapeVector shape_b_;
  ShapeVector shape_c_;
};
}  // namespace checksum
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_SILENT_DETECT_CHECKSUM_CHECKSUM_KERNEL_H_
