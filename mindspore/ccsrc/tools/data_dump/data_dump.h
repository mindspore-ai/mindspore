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
#include <memory>
#include <string>
#include <vector>

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "ir/anf.h"
#include "tools/visible.h"
#include "utils/ms_context.h"

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DATA_DUMP_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DATA_DUMP_H_

namespace mindspore::datadump {

#ifdef ENABLE_DEBUGGER
void AscendDataDump(const CNodePtr &cnode, const std::vector<kernel::KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &, const device::DeviceContext *device_context);
void GPUDataDump(const CNodePtr &cnode, std::vector<kernel::KernelTensor *>, std::vector<kernel::KernelTensor *>,
                 const device::DeviceContext *device_context);
#endif

void CPUDataDump(const CNodePtr &cnode);

BACKEND_COMMON_EXPORT void DataDump(const CNodePtr &cnode,
                                    const std::vector<kernel::KernelTensor *> &input_kernel_tensors,
                                    const std::vector<kernel::KernelTensor *> &output_kernel_tensors,
                                    const device::DeviceContext *device_context);
}  // namespace mindspore::datadump

#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DATA_DUMP_H_
