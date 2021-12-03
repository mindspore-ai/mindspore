/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef AICPU_OPS_AICPU_PULSE_H_
#define AICPU_OPS_AICPU_PULSE_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*PulseNotifyFunc)(void);

/**
 * aicpu pulse notify.
 * timer will call this method per second.
 */
void AicpuPulseNotify(void);

/**
 * Register kernel pulse notify func.
 * @param name name of kernel lib, must end with '\0' and unique.
 * @param func pulse notify function.
 * @return 0:success, other:failed.
 */
int32_t RegisterPulseNotifyFunc(const char *name, PulseNotifyFunc func);

#ifdef __cplusplus
}
#endif

#endif  // AICPU_OPS_AICPU_PULSE_H_
