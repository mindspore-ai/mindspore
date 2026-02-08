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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_L2NORM_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_L2NORM_H_
#include <set>
#include <string>
#include <vector>

#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "tools/data_dump/device_statistic/mean.h"

namespace mindspore {

namespace datadump {

inline const std::set<TypeId> norm_supported_dtype{kNumberTypeBFloat16, kNumberTypeFloat16, kNumberTypeFloat32};

class NormStatisticKernel : public MeanStatisticKernel {
 public:
  explicit NormStatisticKernel(const DeviceContext *device_context)
      : MeanStatisticKernel(device_context, ops::kNameNorm, norm_supported_dtype) {}

 protected:
  std::vector<KernelTensorPtr> GetExtraInputsDeviceAddress(KernelTensor *input) override;
};

}  // namespace datadump

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_L2NORM_H_
