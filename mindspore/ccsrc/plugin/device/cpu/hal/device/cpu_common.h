/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_COMMON_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_COMMON_H_

#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace cpu {
#define CHECK_RET_WITH_EXCEPT(expression, status, message) \
  do {                                                     \
    auto ret = (expression);                               \
    if (ret != (status)) {                                 \
      MS_LOG(EXCEPTION) << (message);                      \
    }                                                      \
  } while (0);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_CPU_COMMON_H_
