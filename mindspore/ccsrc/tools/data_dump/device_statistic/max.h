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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_MAX_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_MAX_H_
#include <set>
#include <string>
#include <vector>

#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "tools/data_dump/device_statistic/statistic_kernel.h"

namespace mindspore {

namespace datadump {

inline const std::set<TypeId> GetMaxSupportedDtype() {
  static auto is_low_precison_mode = !DumpJsonParser::GetInstance().IsDeviceStatHighPrecisionMode();
  // In low-precision mode, memory multiplexing is not used for the workspace.
  // The workspace memory is released only after the callback is fully executed.
  // Operators of the bool type are internally converted to int32 for calculation,
  // leading to excessive memory usage. Therefore, support for the bool type is removed.
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

class MaxStatisticKernel : public StatisticKernel {
 public:
  explicit MaxStatisticKernel(const DeviceContext *device_context)
      : StatisticKernel(device_context, ops::kNameMax, GetMaxSupportedDtype()) {}
};

}  // namespace datadump

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_MAX_H_
