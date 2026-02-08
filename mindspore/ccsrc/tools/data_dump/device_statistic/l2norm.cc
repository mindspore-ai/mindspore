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
#include "tools/data_dump/device_statistic/l2norm.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tools/data_dump/device_statistic/kernel_factory.h"

namespace mindspore {
namespace datadump {

std::vector<KernelTensorPtr> NormStatisticKernel::GetExtraInputsDeviceAddress(KernelTensor *input) {
  MS_EXCEPTION_IF_NULL(input);

  auto scalar = std::make_shared<KernelTensor>(nullptr, kTypeNone, kNone);
  MS_EXCEPTION_IF_NULL(scalar);
  std::vector<KernelTensorPtr> extra_inputs{scalar};

  auto other_extra_inputs = MeanStatisticKernel::GetExtraInputsDeviceAddress(input);
  extra_inputs.insert(extra_inputs.end(), other_extra_inputs.begin(), other_extra_inputs.end());
  return extra_inputs;
}

REGISTER_KERNEL(KStatL2Norm, NormStatisticKernel);

}  // namespace datadump
}  // namespace mindspore
