/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_KERNEL_INFO_ADAPTER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_KERNEL_INFO_ADAPTER_H_

#include <memory>
#include <vector>
#include <string>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "ir/value.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
${kernel_info_adapter_list}
#define MS_KERNEL_INFO_ADAPTER_REG(NAME, DERIVE, BASE) MS_KERNEL_FACTORY_REG(BASE, NAME, DERIVE)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_KERNEL_INFO_ADAPTER_H_
