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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_CHECK_OVERFLOW_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_CHECK_OVERFLOW_H_
#include <map>
#include <set>
#include <string>
#include <vector>

#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "primitive/nn_op_name.h"
#include "tools/data_dump/device_statistic/statistic_kernel.h"
#include "tools/visible.h"

namespace mindspore {

namespace datadump {

inline const std::set<TypeId> overflow_supported_dtype{kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};

class TOOLS_EXPORT CheckOverflowKernel : public StatisticKernel {
 public:
  explicit CheckOverflowKernel(const DeviceContext *device_context)
      : StatisticKernel(device_context, kAllFiniteOpName, overflow_supported_dtype) {}

  std::vector<KernelTensorPtr> GetWorkSpaceDeviceAddressList(const std::vector<KernelTensor *> &inputs,
                                                             const std::vector<KernelTensor *> &outputs);
  std::vector<KernelTensor *> CheckInputs(std::vector<KernelTensor *> inputs);
  DeviceAddressPtr LaunchKernelAsync(KernelTensor *input, const std::uint32_t stream_id) = delete;
  KernelTensorPtr LaunchKernelAsync(std::vector<KernelTensor *> inputs, const std::uint32_t stream_id) override;
};

}  // namespace datadump
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_CHECK_OVERFLOW_H_
