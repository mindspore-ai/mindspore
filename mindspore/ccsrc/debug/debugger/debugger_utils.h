/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_UTILS_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_UTILS_H_

#include <iostream>
#include <vector>
#include <string>
#include "debug/debugger/debugger.h"
#include "kernel/kernel.h"
#include "runtime/hardware/device_context.h"
using mindspore::device::DeviceContext;
using mindspore::kernel::KernelLaunchInfo;

namespace mindspore {

std::vector<size_t> CheckRealOutput(const std::string &node_name, const size_t &output_size);

void LoadInputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order, uint32_t root_graph_id,
                const DeviceContext *device_context, const bool trans_flag);

void LoadOutputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order,
                 uint32_t root_graph_id, const DeviceContext *device_context, const bool trans_flag);

bool CheckReadData(const CNodePtr &cnode);

void ReadDataAndDump(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order,
                     const DeviceContext *device_context);

std::string CheckDatasetSinkMode(const KernelGraphPtr &graph_ptr);

void LoadDataForDebugger(const KernelGraphPtr &graph_ptr);

void SuperKernelE2eDump(const KernelGraphPtr &graph);
}  // namespace mindspore
#endif
