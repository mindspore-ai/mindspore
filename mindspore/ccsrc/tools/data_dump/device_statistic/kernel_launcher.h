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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_KERNEL_LAUNCHER_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_KERNEL_LAUNCHER_H_
#include <string>
#include <vector>

#include "tools/data_dump/device_statistic/kernel_factory.h"

namespace mindspore {

namespace datadump {
TensorPtr CalStatistic(const std::string &, const DeviceContext *, KernelTensor *, const std::uint32_t);

std::vector<KernelTensorPtr> CalStatisticAsync(const std::string &, const DeviceContext *, KernelTensor *,
                                               const std::uint32_t);

bool CalCheckOverflow(const DeviceContext *, const std::vector<KernelTensor *> &, const std::uint32_t);

KernelTensorPtr CalCheckOverflowAsync(const DeviceContext *, std::vector<KernelTensor *>, const std::uint32_t);
}  // namespace datadump

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_KERNEL_LAUNCHER_H_
