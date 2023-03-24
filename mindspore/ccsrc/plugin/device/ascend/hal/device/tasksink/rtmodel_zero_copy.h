/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TASKSINK_RTMODEL_ZERO_COPY_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TASKSINK_RTMODEL_ZERO_COPY_H_

#include <memory>
#include <utility>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "include/backend/device_address.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
class ZeroCopyTask {
 public:
  ZeroCopyTask(AnfNodeWeakPtr anf_node, void *args_base, size_t args_offset, std::string task_name)
      : anf_node_(std::move(anf_node)),
        args_base_(args_base),
        args_offset_(args_offset),
        task_name_(std::move(task_name)) {}
  virtual ~ZeroCopyTask() = default;

  // Update the address in task args
  bool UpdateArgs(void *stream);

  virtual void *GetAddressPtr() = 0;

 protected:
  // Parameter or ValueNode
  AnfNodeWeakPtr anf_node_;

 private:
  void *args_base_;
  size_t args_offset_;
  std::string task_name_;
  void *device_ptr_{nullptr};
  void *previous_ptr_{nullptr};
};
using ZeroCopyTaskPtr = std::shared_ptr<ZeroCopyTask>;

class ParameterZeroCopyTask : public ZeroCopyTask {
 public:
  ParameterZeroCopyTask(const AnfNodeWeakPtr &anf_node, void *args_base, size_t args_offset,
                        const std::string &task_name)
      : ZeroCopyTask(anf_node, args_base, args_offset, task_name) {}
  ~ParameterZeroCopyTask() override = default;
  void *GetAddressPtr() override;
};

class ValueNodeZeroCopyTask : public ZeroCopyTask {
 public:
  ValueNodeZeroCopyTask(const AnfNodeWeakPtr &anf_node, void *args_base, size_t args_offset,
                        const std::string &task_name)
      : ZeroCopyTask(anf_node, args_base, args_offset, task_name) {}
  ~ValueNodeZeroCopyTask() override = default;
  void *GetAddressPtr() override;
};

class CNodeZeroCopyTask : public ZeroCopyTask {
 public:
  CNodeZeroCopyTask(const AnfNodeWeakPtr &anf_node, size_t output_index, void *args_base, size_t args_offset,
                    const std::string &task_name)
      : ZeroCopyTask(anf_node, args_base, args_offset, task_name) {
    output_index_ = output_index;
  }
  ~CNodeZeroCopyTask() override = default;
  void *GetAddressPtr() override;

 private:
  size_t output_index_;
};

// Update the device address in task without copying data.
// Usually we assume that the address in the task is constant.
// If the address of graph input changed, we need to copy data of graph input tensor to the address of the task.
// In fact, when the operator is executed, the real address is obtained from the task args.
// Task args is a secondary pointer, which stores the input address or output address of the operator.
// If we can update the input address in task args, we can avoid data copying.
class RtModelZeroCopy {
 public:
  RtModelZeroCopy() = default;
  ~RtModelZeroCopy() = default;

  // Generate ZeroCopyTasks after the tasks of rtModel is Distributed. (Need to get task args address)
  bool GenerateZeroCopyTasks(const session::KernelGraph &graph);
  bool GenerateZeroCopyTaskForSubGraphSink(const session::KernelGraph &graph);
  // Copy device address ptr to task args if the ptr changed.
  bool UpdateTaskArgs(const session::KernelGraph &graph, void *stream) const;
  // Check rtModel after update task args. The process of checking is consistent with the process of generating tasks.
  static bool CheckRtModelValid(const session::KernelGraph &graph);
  // Release resource after the graph is destructed.
  void Release(uint32_t graph_id);

 private:
  mindspore::HashMap<uint32_t, std::vector<ZeroCopyTaskPtr>> graph_zero_copy_tasks_;
};
}  // namespace tasksink
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TASKSINK_RTMODEL_ZERO_COPY_H_
