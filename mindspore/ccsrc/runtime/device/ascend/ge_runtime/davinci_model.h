/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_DAVINCI_MODEL_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_DAVINCI_MODEL_H_

#include <memory>
#include <vector>
#include "runtime/device/ascend/ge_runtime/task_info.h"

namespace mindspore::ge::model_runner {
class DavinciModel {
 public:
  DavinciModel(const std::vector<std::shared_ptr<TaskInfo>> &task_info_list,
               const std::vector<uint32_t> &wait_active_stream_list,
               const std::vector<uint32_t> &force_copy_stream_list, uint64_t mem_size = 0, uint64_t weight_size = 0,
               uint64_t var_size = 0, uintptr_t logic_mem_base = 0, uintptr_t logic_weight_base = 0,
               uintptr_t logic_var_base = 0, uint32_t stream_num = 0, uint32_t batch_num = 0, uint32_t event_num = 0,
               int32_t priority = 0)
      : task_info_list_(task_info_list),
        wait_active_stream_list_(wait_active_stream_list),
        force_copy_stream_list_(force_copy_stream_list),
        mem_size_(mem_size),
        weight_size_(weight_size),
        var_size_(var_size),
        logic_mem_base_(logic_mem_base),
        logic_weight_base_(logic_weight_base),
        logic_var_base_(logic_var_base),
        stream_num_(stream_num),
        batch_num_(batch_num),
        event_num_(event_num),
        priority_(priority) {}
  ~DavinciModel() {}

  uint64_t GetMemSize() const { return mem_size_; }
  uint64_t GetWeightSize() const { return weight_size_; }
  uint64_t GetVarSize() const { return var_size_; }

  uintptr_t GetLogicMemBase() const { return logic_mem_base_; }
  uintptr_t GetLogicWeightBase() const { return logic_weight_base_; }
  uintptr_t GetLogicVarBase() const { return logic_var_base_; }

  uint32_t GetStreamNum() const { return stream_num_; }
  uint32_t GetBatchNum() const { return batch_num_; }
  uint32_t GetEventNum() const { return event_num_; }

  const std::vector<uint32_t> &GetWaitActiveStreams() const { return wait_active_stream_list_; }
  const std::vector<uint32_t> &GetForceCopyStreams() const { return force_copy_stream_list_; }

  int32_t GetPriority() const { return priority_; }

  const std::vector<std::shared_ptr<TaskInfo>> &GetTaskInfoList() const { return task_info_list_; }

 private:
  std::vector<std::shared_ptr<TaskInfo>> task_info_list_;

  std::vector<uint32_t> wait_active_stream_list_;
  std::vector<uint32_t> force_copy_stream_list_;

  uint64_t mem_size_;
  uint64_t weight_size_;
  uint64_t var_size_;

  uintptr_t logic_mem_base_;
  uintptr_t logic_weight_base_;
  uintptr_t logic_var_base_;

  uint32_t stream_num_;
  uint32_t batch_num_;
  uint32_t event_num_;

  int32_t priority_;

  // Disable to copy constructor and assignment operator
  DavinciModel &operator=(const DavinciModel &) = delete;
  DavinciModel(const DavinciModel &) = delete;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_DAVINCI_MODEL_H_
