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

#include "plugin/device/ascend/hal/device/tasksink/rtmodel_zero_copy.h"

#include <vector>
#include <map>
#include <algorithm>
#include "runtime/rt.h"
#include "external/acl/acl_rt.h"
#include "ir/tensor.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_info.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "plugin/device/ascend/hal/device/ge_runtime/model_runner.h"
#include "plugin/device/ascend/hal/device/tasksink/task_generator.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
using TaskPtr = std::shared_ptr<ge::model_runner::Task>;
namespace {
bool IsForwardOutputValueNode(const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->isa<ValueNode>()) {
    auto value_node = input->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor->is_forward_output()) {
        return true;
      }
    }
  }
  return false;
}

bool CheckTaskValid(const CNodePtr &node, const std::vector<void *> &args_datas) {
  MS_EXCEPTION_IF_NULL(node);
  bool task_valid = true;
  // Check input/output/workspace
  auto input_addrs = TaskGenerator::GetTaskInput(node);
  auto output_addrs = TaskGenerator::GetTaskOutput(node);
  auto workspace_addrs = TaskGenerator::GetTaskWorkspace(node);

  std::vector<AddressPtr> node_addresses;
  (void)std::move(input_addrs.begin(), input_addrs.end(), std::back_inserter(node_addresses));
  (void)std::move(output_addrs.begin(), output_addrs.end(), std::back_inserter(node_addresses));
  (void)std::move(workspace_addrs.begin(), workspace_addrs.end(), std::back_inserter(node_addresses));

  if (node_addresses.size() != args_datas.size()) {
    MS_LOG(ERROR) << "Check failed, Node " << node->UniqueName() << " total addr size " << node_addresses.size()
                  << " is not equal to " << args_datas.size();
    return false;
  }

  for (size_t i = 0; i < node_addresses.size(); ++i) {
    auto node_address = node_addresses[i];
    MS_EXCEPTION_IF_NULL(node_address);
    if (node_address->addr != args_datas[i]) {
      MS_LOG(ERROR) << "Node " << node->UniqueName() << " addr " << node_address->addr << " not equal to addr of task "
                    << args_datas[i];
      task_valid = false;
    }
  }

  return task_valid;
}

bool NeedSkipZeroCopy(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::IsNonTaskOp(node)) {
    MS_LOG(INFO) << "Skip generate ZeroCopyTask for NonTaskOp " << node->fullname_with_scope();
    return true;
  }
  auto kernel_type = AnfAlgo::GetKernelType(node);
  if (kernel_type != KernelType::TBE_KERNEL && kernel_type != KernelType::AICPU_KERNEL) {
    MS_LOG(INFO) << "Skip generate ZeroCopyTask for " << node->fullname_with_scope();
    return true;
  }
  return false;
}
}  // namespace

void *ParameterZeroCopyTask::GetAddressPtr() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Not a parameter node " << node->DebugString();
  }
  auto kernel_info = dynamic_cast<KernelInfo *>(node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto parameter_address = kernel_info->GetOutputAddr(0);
  MS_EXCEPTION_IF_NULL(parameter_address);
  return parameter_address->GetMutablePtr();
}

void *ValueNodeZeroCopyTask::GetAddressPtr() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    MS_LOG(EXCEPTION) << "Not a ValueNode " << node->DebugString();
  }

  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  auto tensor = value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto value_node_address = tensor->device_address();
  MS_EXCEPTION_IF_NULL(value_node_address);
  return value_node_address->GetMutablePtr();
}

bool ZeroCopyTask::UpdateArgs(void *stream) {
  device_ptr_ = GetAddressPtr();
  if (device_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Device address ptr is null, task " << task_name_;
    return false;
  }

  if (device_ptr_ == previous_ptr_) {
    MS_LOG(DEBUG) << "No need to update task of " << task_name_;
    return true;
  }

  auto ret = aclrtMemcpyAsync(static_cast<uint8_t *>(args_base_) + args_offset_, sizeof(void *), &device_ptr_,
                              sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Update task " << task_name_ << " aclrtMemcpy failed, ret " << ret;
    return false;
  }

  previous_ptr_ = device_ptr_;
  MS_LOG(INFO) << "Update task " << task_name_ << " args_offset " << args_offset_ << " device_ptr " << device_ptr_
               << " success";
  return true;
}

bool RtModelZeroCopy::GenerateZeroCopyTasks(const session::KernelGraph &graph) {
  if (!graph.has_flag(kFlagPyNativeRunInGraph)) {
    MS_LOG(INFO) << "RtModelZeroCopy is not enabled";
    return true;
  }

  std::vector<ZeroCopyTaskPtr> zero_copy_tasks;
  auto task_lists = ge::model_runner::ModelRunner::Instance().GetTaskList(graph.graph_id());
  std::map<std::string, TaskPtr> op_name_to_task;
  (void)std::transform(task_lists.begin(), task_lists.end(), std::inserter(op_name_to_task, op_name_to_task.end()),
                       [](const TaskPtr &task) { return std::make_pair(task->task_name(), task); });

  auto nodes = graph.execution_order();
  for (const auto &node : nodes) {
    if (NeedSkipZeroCopy(node)) {
      continue;
    }

    MS_EXCEPTION_IF_NULL(node);
    auto op_name = node->UniqueName();
    auto iter = op_name_to_task.find(op_name);
    if (iter == op_name_to_task.end()) {
      MS_LOG(EXCEPTION) << "Cannot found task of op " << op_name;
    }

    auto task = iter->second;
    MS_EXCEPTION_IF_NULL(task);
    auto input_num = common::AnfAlgo::GetInputTensorNum(node);
    for (size_t i = 0; i < input_num; ++i) {
      auto input_index_in_graph = AnfAlgo::GetInputIndexInGraph(node, i);
      auto input = common::AnfAlgo::GetPrevNodeOutput(node, input_index_in_graph, true).first;
      MS_EXCEPTION_IF_NULL(input);
      if (input->isa<Parameter>()) {
        (void)zero_copy_tasks.emplace_back(std::make_shared<tasksink::ParameterZeroCopyTask>(
          input, task->Args(), i * sizeof(void *), task->task_name()));
        MS_LOG(INFO) << "Generate ZeroCopyTask for Node " << node->fullname_with_scope() << " Parameter "
                     << input->DebugString();
      } else if (IsForwardOutputValueNode(input)) {
        (void)zero_copy_tasks.emplace_back(std::make_shared<tasksink::ValueNodeZeroCopyTask>(
          input, task->Args(), i * sizeof(void *), task->task_name()));
        MS_LOG(INFO) << "Generate ZeroCopyTask for Node " << node->fullname_with_scope() << " ValueNode "
                     << input->DebugString();
      }
    }
  }

  auto iter = graph_zero_copy_tasks_.try_emplace(graph.graph_id(), zero_copy_tasks);
  if (!iter.second) {
    MS_LOG(ERROR) << "Generate ZeroCopyTask failed, Duplicate graph id " << graph.graph_id();
    return false;
  }
  return true;
}

bool RtModelZeroCopy::UpdateTaskArgs(const session::KernelGraph &graph, void *stream) const {
  if (!graph.has_flag(kFlagPyNativeRunInGraph)) {
    MS_LOG(INFO) << "RtModelZeroCopy is not enabled, no need to update task args.";
    return true;
  }

  auto iter = graph_zero_copy_tasks_.find(graph.graph_id());
  if (iter == graph_zero_copy_tasks_.end()) {
    MS_LOG(ERROR) << "No zero copy tasks found. graph id " << graph.graph_id();
    return false;
  }

  auto zero_copy_tasks = iter->second;
  if (std::any_of(zero_copy_tasks.begin(), zero_copy_tasks.end(),
                  [stream](const ZeroCopyTaskPtr &task) { return !task->UpdateArgs(stream); })) {
    MS_LOG(ERROR) << "Update task args failed";
    return false;
  }

  MS_LOG(INFO) << "Check rtMode valid " << ((rtStreamSynchronize(stream) == RT_ERROR_NONE) && CheckRtModelValid(graph));
  return true;
}

bool RtModelZeroCopy::CheckRtModelValid(const session::KernelGraph &graph) {
  auto graph_id = graph.graph_id();
  auto tasks = ge::model_runner::ModelRunner::Instance().GetTaskList(graph_id);
  std::map<std::string, TaskPtr> op_name_to_task;
  (void)std::transform(tasks.begin(), tasks.end(), std::inserter(op_name_to_task, op_name_to_task.end()),
                       [](const TaskPtr &task) { return std::make_pair(task->task_name(), task); });

  auto nodes = graph.execution_order();
  bool task_valid = true;
  for (const auto &node : nodes) {
    if (NeedSkipZeroCopy(node)) {
      continue;
    }

    MS_EXCEPTION_IF_NULL(node);
    auto unique_name = node->UniqueName();
    auto iter = op_name_to_task.find(unique_name);
    if (iter == op_name_to_task.end()) {
      MS_LOG(ERROR) << "Cannot found task of op " << unique_name;
      task_valid = false;
      continue;
    }

    auto task = iter->second;
    MS_EXCEPTION_IF_NULL(task);
    auto task_args = task->Args();
    auto task_size = task->ArgsSize();
    if (task_size == 0) {
      // For example InitDataSet (AiCpu kernel).
      MS_LOG(INFO) << "task name " << task->task_name() << " task_size is 0";
      continue;
    }
    std::vector<void *> args_datas(task_size / sizeof(void *), nullptr);
    if (aclrtMemcpy(args_datas.data(), task_size, task_args, task_size, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrtMemcpy failed, task " << task->task_name() << " task_size " << task_size;
      return false;
    }

    if (!CheckTaskValid(node, args_datas)) {
      task_valid = false;
    }
  }
  return task_valid;
}

void RtModelZeroCopy::Release(uint32_t graph_id) { (void)graph_zero_copy_tasks_.erase(graph_id); }
}  // namespace tasksink
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
