/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_CPU_KERNEL_EXECUTOR_KERNEL_SELECT_KERNEL_SELECT_CPU_H_
#define MINDSPORE_CCSRC_PLUGIN_CPU_KERNEL_EXECUTOR_KERNEL_SELECT_KERNEL_SELECT_CPU_H_

#include <utility>
#include <string>
#include <vector>

#include "ir/anf.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "plugin/cpu/visible.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_info.h"

namespace mindspore {
namespace device {
namespace cpu {
using kernel::DataType;
bool IsVmapNotSupported(const CNodePtr &node);
CPU_EXPORT std::pair<std::string, ExceptionType> SetKernelInfoWithMsg(const CNodePtr &apply_kernel_ptr);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_CPU_KERNEL_EXECUTOR_KERNEL_SELECT_KERNEL_SELECT_CPU_H_
