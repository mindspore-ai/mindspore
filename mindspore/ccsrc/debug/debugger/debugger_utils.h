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

#include <iostream>
#include <vector>
#include <string>
#include "debug/debugger/debugger.h"
#include "backend/kernel_compiler/kernel.h"
#include "runtime/hardware/device_context.h"
#ifdef ENABLE_D
#include "toolchain/adx_datadump_callback.h"

using Adx::DumpChunk;
#endif
using mindspore::device::DeviceContext;
using mindspore::kernel::KernelLaunchInfo;

namespace mindspore {

std::vector<size_t> CheckRealOutput(const std::string &node_name, const size_t &output_size);

void LoadInputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order, uint32_t root_graph_id,
                const DeviceContext *device_context);

void LoadOutputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order,
                 uint32_t root_graph_id, const DeviceContext *device_context);

bool CheckReadData(const CNodePtr &cnode);

void ReadDataAndDump(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order,
                     const DeviceContext *device_context);

std::string CheckDatasetSinkMode(const KernelGraphPtr &graph_ptr);

void LoadDataForDebugger(const KernelGraphPtr &graph_ptr);

void SuperKernelE2eDump(const KernelGraphPtr &graph);

#ifdef ENABLE_D
// Callback function to dump ascend async mode
int32_t DumpDataCallBack(const DumpChunk *dump_chunk, int32_t size);
#endif
}  // namespace mindspore
