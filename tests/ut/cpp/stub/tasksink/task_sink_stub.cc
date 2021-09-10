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

#include "runtime/device/ascend/tasksink/task_generator.h"
#ifndef ENABLE_SECURITY
#include "runtime/device/ascend/dump/data_dumper.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
bool TaskGenerator::GenTasks(const std::vector<CNodePtr> &anf_node_list, std::vector<TaskInfoPtr> *const task_info_list,
                             uint32_t graph_id) {
  return true;
}
}  // namespace tasksink
#ifndef ENABLE_SECURITY
void DataDumper::LoadDumpInfo() {}
void DataDumper::UnloadDumpInfo() {}
void DataDumper::OpDebugRegister() {}
void DataDumper::OpDebugUnregister() {}
DataDumper::~DataDumper() {}
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore