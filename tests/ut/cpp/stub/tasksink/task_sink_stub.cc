/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/tasksink/task_generator.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/dump/data_dumper.h"
#include "plugin/device/ascend/hal/device/dump/kernel_dumper.h"
#include "mindspore/ccsrc/kernel/kernel.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
bool TaskGenerator::GenTasks(const std::vector<CNodePtr> &anf_node_list, std::vector<TaskInfoPtr> *const task_info_list,
                             uint32_t graph_id) {
  return true;
}

AddressPtrList TaskGenerator::GetTaskInput(const CNodePtr &node) { return {}; }
AddressPtrList TaskGenerator::GetTaskOutput(const CNodePtr &node) { return {}; }
AddressPtrList TaskGenerator::GetTaskWorkspace(const CNodePtr &node) { return {}; }
}  // namespace tasksink
#ifndef ENABLE_SECURITY
void DataDumper::LoadDumpInfo() {}
void DataDumper::UnloadDumpInfo() {}
void DataDumper::OpDebugRegister() {}
void DataDumper::OpDebugUnregister() {}
DataDumper::~DataDumper() {}
std::map<std::string, std::string> KernelDumper::stream_task_graphs;
void KernelDumper::OpLoadDumpInfo(const CNodePtr &kernel) {}
void KernelDumper::DumpHcclOutput(const std::shared_ptr<HcclTaskInfo> &task_info, const rtStream_t stream) {}
void KernelDumper::Init() {}
void KernelDumper::OpDebugRegisterForStream(const CNodePtr &kernel) {}
void KernelDumper::OpDebugUnregisterForStream() {}
KernelDumper::~KernelDumper() {}
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore