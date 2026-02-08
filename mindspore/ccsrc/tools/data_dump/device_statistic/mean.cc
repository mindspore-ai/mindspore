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
#include "tools/data_dump/device_statistic/mean.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tools/data_dump/device_statistic/kernel_factory.h"

namespace mindspore {
namespace datadump {

std::vector<KernelTensorPtr> MeanStatisticKernel::GetExtraInputsDeviceAddress(KernelTensor *input) {
  MS_EXCEPTION_IF_NULL(input);

  auto dim = input->GetShapeVector().size();
  std::vector<int64_t> axes(dim);
  std::iota(axes.begin(), axes.end(), 0LL);
  auto axis = std::make_shared<KernelTensor>(
    std::make_shared<abstract::TensorShape>(ShapeVector(static_cast<int64_t>(dim))), kInt64, MakeValue(axes));
  MS_EXCEPTION_IF_NULL(axis);
  auto keepdims = std::make_shared<KernelTensor>(nullptr, kBool, MakeValue(false));
  MS_EXCEPTION_IF_NULL(keepdims);
  auto dtype = std::make_shared<KernelTensor>(nullptr, kTypeNone, kNone);
  MS_EXCEPTION_IF_NULL(dtype);

  return {axis, keepdims, dtype};
}

KernelTensorPtr MeanStatisticKernel::GetOutputDeviceAddress(TypeId type_id) {
  if (DumpJsonParser::GetInstance().IsDeviceStatHighPrecisionMode()) {
    type_id = kNumberTypeFloat32;
  }
  return StatisticKernel::GetOutputDeviceAddress(type_id);
}

REGISTER_KERNEL(KStatMean, MeanStatisticKernel);

}  // namespace datadump
}  // namespace mindspore
