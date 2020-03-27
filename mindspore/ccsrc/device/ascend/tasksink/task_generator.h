/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_TASK_TASK_BUILD_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_TASK_TASK_BUILD_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "cce/aicpu_engine_struct.h"
#include "cce/taskdown_api.h"
#include "cce/fwk_adpt_struct.h"
#include "device/kernel_runtime.h"
#include "ir/anf.h"
#include "kernel/kernel.h"
#include "framework/ge_runtime/task_info.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;
using ge::model_runner::TaskInfo;
using TaskInfoPtr = std::shared_ptr<TaskInfo>;
class TaskGenerator {
 public:
  TaskGenerator() = default;
  ~TaskGenerator() = default;
  TaskGenerator(const TaskGenerator &in) = delete;
  TaskGenerator &operator=(const TaskGenerator &in) = delete;

  static bool GenTasks(const std::vector<CNodePtr> &anf_node_list, std::vector<TaskInfoPtr> *task_info_list,
                       uint32_t graph_id);

 private:
  static void LaunchAddrCleanKernel(const CNodePtr &anf_node_ptr, AddressPtrList *kernel_inputs);
  static bool LaunchKernel(const CNodePtr &anf_node_ptr, uint32_t stream_id, std::vector<TaskInfoPtr> *task_info_list);
  static bool LaunchAllKernel(const std::vector<CNodePtr> &anf_node_list, std::vector<TaskInfoPtr> *task_info_list,
                              uint32_t graph_id);
};
}  // namespace tasksink
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_TASK_TASK_BUILD_H_
