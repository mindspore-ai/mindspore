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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_MIN_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_MIN_H_
#include <set>
#include <string>
#include <vector>

#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "tools/data_dump/device_statistic/statistic_kernel.h"

namespace mindspore {

namespace datadump {

inline const std::set<TypeId> GetMinSupportedDtype() {
  static auto is_low_precison_mode = !DumpJsonParser::GetInstance().IsDeviceStatHighPrecisionMode();
  // In low-precision mode, memory multiplexing is not used.
  // Workspace mem is not released until the callback is complete.
  // Operators of the bool type are converted into those of the int32 type for calculation.
  // As a result, the memory usage is too high. Therefore, the support for the bool type is removed.
  if (is_low_precison_mode) {
    return {kNumberTypeBFloat16, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64,
            kNumberTypeFloat,    kNumberTypeDouble,  kNumberTypeInt,     kNumberTypeInt8,
            kNumberTypeUInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64};
  } else {
    return {kNumberTypeBFloat16, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeFloat,
            kNumberTypeDouble,   kNumberTypeInt,     kNumberTypeInt8,    kNumberTypeUInt8,   kNumberTypeInt16,
            kNumberTypeInt32,    kNumberTypeInt64,   kNumberTypeBool};
  }
}

class MinStatisticKernel : public StatisticKernel {
 public:
  explicit MinStatisticKernel(const DeviceContext *device_context)
      : StatisticKernel(device_context, ops::kNameMin, GetMinSupportedDtype()) {}
};

}  // namespace datadump

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_MIN_H_
