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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_AICPU_TASK_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_AICPU_TASK_H_

#include <vector>
#include <memory>
#include <string>
#include "runtime/device/ascend/ge_runtime/task/task.h"

namespace mindspore::ge::model_runner {
class AicpuTask : public TaskRepeater<AicpuTaskInfo> {
 public:
  AicpuTask(const ModelContext &model_context, const std::shared_ptr<AicpuTaskInfo> &task_info);

  ~AicpuTask() override;

  void Distribute() override;

  void *Args() override { return input_output_addr_; }

  std::string task_name() const override { return task_info_->op_name(); }

 private:
  static void ReleaseRtMem(void **ptr) noexcept;
  void SetAicpuParamHead(uint32_t args_size, uint32_t io_addrs_num);
  void SetInputOutputAddrs(const std::vector<void *> &io_addrs, uint32_t io_addr_offset);
  void SetNodeDef(uint32_t node_def_len_offset, uint32_t node_def_addr_offset);

  std::shared_ptr<AicpuTaskInfo> task_info_;
  void *stream_;
  void *args_;
  void *ext_info_;
  void *input_output_addr_;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_AICPU_TASK_H_
