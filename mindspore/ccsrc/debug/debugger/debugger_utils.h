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

#include <iostream>
#include <vector>
#include <string>
#include "debug/debugger/debugger.h"
#include "backend/kernel_compiler/kernel.h"

using mindspore::kernel::KernelLaunchInfo;

namespace mindspore {

std::vector<size_t> CheckRealOutput(const std::string &node_name, const size_t &output_size);

void LoadInputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info_, uint32_t exec_order_);

void LoadOutputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info_, uint32_t exec_order_);

bool CheckReadData(const CNodePtr &cnode);

void ReadDataAndDump(const CNodePtr &cnode, const KernelLaunchInfo *launch_info_, uint32_t exec_order_);

}  // namespace mindspore
