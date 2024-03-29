/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_BISHENG_IMPL_ADD_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_BISHENG_IMPL_ADD_H

#include <stdint.h>
#include "plugin/device/ascend/kernel/bisheng/impl/visible.h"

namespace mindspore::kernel::bisheng {
template <typename T>
BISHENG_LIB_EXPORT void Add(void *x1, void *x2, void *y, void *size, void *stream);
}  // namespace mindspore::kernel::bisheng
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_BISHENG_IMPL_ADD_H
