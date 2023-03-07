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
#include <set>
#include <algorithm>
#include "runtime/rt.h"
#include "external/acl/acl_rt.h"
#include "ir/tensor.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
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
  std::move(input_addrs.begin(), input_addrs.end(), std::back_inserter(node_addresses));
  std::move(output_addrs.begin(), output_addrs.end(), std::back_inserter(node_addresses));
  std::move(workspace_addrs.begin(), workspace_addrs.end(), std::back_inserter(node_addresses));

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
                    << args_datas[i] << " index:" << i;
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
  if (kernel_type != KernelType::TBE_KERNEL && kernel_type != KernelType::AICPU_KERNEL &&
      kernel_type != KernelType::AKG_KERNEL) {
    MS_LOG(INFO) << "Skip generate ZeroCopyTask for " << node->fullname_with_scope();
    return true;
  }
  return false;
}

bool EnableZeroCopyForSubgraphSink(const session::KernelGraph &graph) {
  if (!graph.has_flag(kFlagEnableZeroCopyInGraph)) {
    return false;
  }
  auto ms_ctx = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_ctx);
  if (ms_ctx->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode &&
      ms_ctx->get_param<bool>(MS_CTX_ENABLE_TASK_SINK) == true &&
      ms_ctx->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK) == false) {
    return true;
  }
  return false;
}

size_t FetchInputNumByInputNode(const AnfNodePtr &node, const KernelWithIndex &input_with_index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_with_index.first);
  auto input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    size_t input_index_in_graph = AnfAlgo::GetInputGraphIdxByKernelIdx(node, i);
    const auto &node_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_index_in_graph, true);
    if (node_with_index == input_with_index) {
      return i;
    }
  }
  MS_LOG(EXCEPTION) << "Invalid input node:" << input_with_index.first->DebugString()
                    << " index:" << input_with_index.second << " for node:" << node->DebugString();
  return 0;
}

// If the output node is output of kernel graph or the input of output ref node, the output should be replaced.
bool IsOutputZeroCopy(const KernelWithIndex &node, const std::vector<KernelWithIndex> &graph_outputs,
                      const std::set<KernelWithIndex> &zero_copy_ref_nodes) {
  return ((std::find_if(graph_outputs.begin(), graph_outputs.end(),
                        [&node](const KernelWithIndex &output) {
                          const auto &real_output = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
                          return real_output == node;
                        }) != graph_outputs.end()) ||
          (zero_copy_ref_nodes.find(node) != zero_copy_ref_nodes.end()));
}

// Check if the node with index has an input of parameter.
bool IsParameterInputRefNode(const KernelWithIndex &node_with_index,
                             const std::map<KernelWithIndex, KernelWithIndex> &ref_map, AnfNodePtr *parameter) {
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  MS_EXCEPTION_IF_NULL(parameter);
  if (node_with_index.first->isa<Parameter>()) {
    *parameter = node_with_index.first;
    return true;
  }
  const auto &iter = ref_map.find(node_with_index);
  if (iter == ref_map.end()) {
    return false;
  }
  return IsParameterInputRefNode(iter->second, ref_map, parameter);
}

void DumpDeviceAddressInGraph(const session::KernelGraph &graph) {
  auto graph_id = graph.graph_id();
  auto tasks = ge::model_runner::ModelRunner::Instance().GetTaskList(graph_id);
  std::map<std::string, TaskPtr> op_name_to_task;
  std::transform(tasks.begin(), tasks.end(), std::inserter(op_name_to_task, op_name_to_task.end()),
                 [](const TaskPtr &task) { return std::make_pair(task->task_name(), task); });
  MS_LOG(WARNING) << "Start dump device address for graph:" << graph.ToString();
  auto nodes = graph.execution_order();
  for (const auto &node : nodes) {
    if (NeedSkipZeroCopy(node)) {
      continue;
    }

    MS_EXCEPTION_IF_NULL(node);
    auto unique_name = node->UniqueName();
    auto iter = op_name_to_task.find(unique_name);
    if (iter == op_name_to_task.end()) {
      MS_LOG(ERROR) << "Cannot found task of op " << unique_name;
      continue;
    }

    auto task = iter->second;
    MS_EXCEPTION_IF_NULL(task);
    auto task_args = task->Args();
    auto task_size = task->ArgsSize();
    if (task_size == 0) {
      // For example InitDataSet (AiCpu kernel).
      MS_LOG(WARNING) << "task name " << task->task_name() << " task_size is 0";
      continue;
    }
    std::vector<void *> args_datas(task_size / sizeof(void *), nullptr);
    if (aclrtMemcpy(args_datas.data(), task_size, task_args, task_size, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrtMemcpy failed, task " << task->task_name() << " task_size " << task_size;
      continue;
    }
    for (size_t i = 0; i < args_datas.size(); ++i) {
      MS_LOG(WARNING) << "Graph:" << graph.ToString() << " kernel:" << node->fullname_with_scope()
                      << " task:" << task->task_name() << " arg index:" << i << " ptr:" << args_datas[i];
    }
  }

  for (const auto &input_node : graph.input_nodes()) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (AnfAlgo::OutputAddrExist(input_node, 0, false)) {
      const auto &device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_address);
      MS_LOG(WARNING) << "Graph:" << graph.ToString() << " input node:" << input_node->DebugString()
                      << " device address:" << device_address << " ptr:" << device_address->GetPtr();
    } else {
      MS_LOG(WARNING) << "Graph:" << graph.ToString() << " input node:" << input_node->DebugString()
                      << " device address:0";
    }
  }
  for (const auto &kernel : graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto output_num = AnfAlgo::GetOutputTensorNum(kernel);
    for (size_t i = 0; i < output_num; ++i) {
      MS_LOG(WARNING) << "Graph:" << graph.ToString() << " kernel:" << kernel->fullname_with_scope()
                      << " output index:" << i << " device address:" << AnfAlgo::GetMutableOutputAddr(kernel, i, false)
                      << " ptr:" << AnfAlgo::GetMutableOutputAddr(kernel, i, false)->GetPtr();
    }
  }
  for (const auto &node_pair : graph.GetRefMap()) {
    MS_EXCEPTION_IF_NULL(node_pair.first.first);
    MS_EXCEPTION_IF_NULL(node_pair.second.first);
    MS_LOG(WARNING) << "Ref output node:" << node_pair.first.first->fullname_with_scope()
                    << " index:" << node_pair.first.second
                    << " input node:" << node_pair.second.first->fullname_with_scope()
                    << " index:" << node_pair.second.second;
  }
}

// Check if all of the empty ptr has a corresponding  zero task.
void CheckZeroCopyTaskValid(const session::KernelGraph &graph,
                            const std::set<std::pair<AnfNodePtr, size_t>> &node_to_offset) {
  for (const auto &kernel : graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    auto output_num = AnfAlgo::GetOutputTensorNum(kernel);
    for (size_t i = 0; i < output_num; ++i) {
      const auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
      if (device_address != nullptr && device_address->GetPtr() == nullptr &&
          node_to_offset.find(std::pair(kernel, i + input_num)) == node_to_offset.end()) {
        DumpDeviceAddressInGraph(graph);
        MS_LOG(EXCEPTION) << "Failed to generate zero copy task for kernel:" << kernel->fullname_with_scope()
                          << " output index:" << i << " in graph:" << graph.ToString();
      }
    }

    for (size_t i = 0; i < input_num; ++i) {
      size_t input_index_in_graph = AnfAlgo::GetInputGraphIdxByKernelIdx(kernel, i);
      const auto &input_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, input_index_in_graph, true);
      if (common::AnfAlgo::IsNoneInput(kernel, i)) {
        continue;
      }
      const auto device_address = AnfAlgo::GetMutableOutputAddr(input_with_index.first, input_with_index.second, false);
      if (device_address != nullptr && device_address->GetPtr() == nullptr &&
          node_to_offset.find(std::pair(kernel, i)) == node_to_offset.end()) {
        DumpDeviceAddressInGraph(graph);
        MS_LOG(EXCEPTION) << "Failed to generate zero copy task for kernel:" << kernel->fullname_with_scope()
                          << " input index:" << i << " in graph:" << graph.ToString();
      }
    }
  }
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
  MS_LOG(DEBUG) << "Get ptr:" << parameter_address->GetMutablePtr() << " for device address:" << parameter_address
                << " in node:" << node->fullname_with_scope();
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

void *CNodeZeroCopyTask::GetAddressPtr() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Not a CNode " << node->DebugString();
  }
  auto node_device_address = AnfAlgo::GetMutableOutputAddr(node, output_index_, false);
  MS_EXCEPTION_IF_NULL(node_device_address);
  if (node_device_address->GetMutablePtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Empty ptr in device address:" << node_device_address
                      << " for node:" << node->fullname_with_scope() << " index:" << output_index_;
  }
  MS_LOG(DEBUG) << "Get ptr:" << node_device_address->GetMutablePtr() << " for device address:" << node_device_address
                << " in node:" << node->fullname_with_scope();
  return node_device_address->GetMutablePtr();
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

namespace {
std::vector<KernelWithIndex> GetInputNodeWithIndex(const CNodePtr &node, const TaskPtr &task,
                                                   const std::vector<KernelWithIndex> &output_with_indexs,
                                                   std::set<std::pair<AnfNodePtr, size_t>> *node_to_offset) {
  std::vector<KernelWithIndex> input_node_with_indexs;
  auto input_num = common::AnfAlgo::GetInputTensorNum(node);
  if (common::AnfAlgo::GetCNodeName(node) == kAtomicAddrCleanOpName) {
    // For atomic addr clean op, the args in task is not the input node of kernel, we should get the real input index
    // from the input node. The output and workspace addr should be reset, and the output addr should be collect.
    size_t workspace_size = 0;
    for (size_t i = 0; i < input_num; ++i) {
      const auto &input = node->input(i + 1);
      MS_EXCEPTION_IF_NULL(input);
      if (input->isa<CNode>()) {
        if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, input->cast<CNodePtr>())) {
          auto clean_output_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(input, kAttrAtomicOutputIndexs);
          for (auto index : clean_output_indexs) {
            MS_LOG(DEBUG) << "atomic addr clean index:" << index << " for node:" << input->fullname_with_scope();
            input_node_with_indexs.emplace_back(input, index);
          }
        } else if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, input->cast<CNodePtr>())) {
          workspace_size += common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(input, kAttrAtomicWorkspaceIndexs).size();
        }
      }
    }
    if (input_node_with_indexs.size() + workspace_size != (task->ArgsSize() / sizeof(void *))) {
      MS_LOG(WARNING) << "Invalid input size:" << input_node_with_indexs.size()
                      << " task size:" << (task->ArgsSize() / sizeof(void *)) << " for node:" << node->DebugString();
    }
  } else {
    for (size_t i = 0; i < input_num; ++i) {
      if (node_to_offset->find(std::make_pair(node, i)) != node_to_offset->end()) {
        input_node_with_indexs.emplace_back(nullptr, i);
        continue;
      }

      size_t input_index_in_graph = AnfAlgo::GetInputGraphIdxByKernelIdx(node, i);
      KernelWithIndex input_with_index{node, input_index_in_graph};
      do {
        input_with_index = common::AnfAlgo::GetPrevNodeOutput(input_with_index.first, input_with_index.second, false);
        if (std::find_if(output_with_indexs.begin(), output_with_indexs.end(),
                         [input_with_index](const KernelWithIndex &output) {
                           const auto &real_output = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
                           return real_output == input_with_index;
                         }) != output_with_indexs.end()) {
          break;
        }
      } while (input_with_index.first != nullptr && common::AnfAlgo::IsNopNode(input_with_index.first));
      MS_LOG(DEBUG) << "Add input node:" << input_with_index.first->fullname_with_scope()
                    << " index:" << input_with_index.second << " for node:" << node->fullname_with_scope();
      input_node_with_indexs.emplace_back(input_with_index);
    }
  }
  return input_node_with_indexs;
}

void GenerateZeroCopyTaskForInput(const CNodePtr &node, const TaskPtr &task, const session::KernelGraph &graph,
                                  std::vector<ZeroCopyTaskPtr> *zero_copy_tasks,
                                  std::set<std::pair<AnfNodePtr, size_t>> *node_to_offset) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(zero_copy_tasks);
  MS_EXCEPTION_IF_NULL(node_to_offset);

  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph.output());
  const auto &ref_node_map = graph.GetRefMap();

  std::vector<KernelWithIndex> input_node_with_indexs =
    GetInputNodeWithIndex(node, task, output_with_indexs, node_to_offset);

  for (size_t i = 0; i < input_node_with_indexs.size(); ++i) {
    KernelWithIndex input_with_index = input_node_with_indexs[i];
    const auto input = input_with_index.first;
    if (input == nullptr || node_to_offset->find(std::make_pair(node, i)) != node_to_offset->end()) {
      continue;
    }

    if (input->isa<Parameter>()) {
      // 1. Input parameter.
      zero_copy_tasks->emplace_back(
        std::make_shared<tasksink::ParameterZeroCopyTask>(input, task->Args(), i * sizeof(void *), task->task_name()));
      node_to_offset->emplace(node, i);
      MS_LOG(DEBUG) << "Add zero copy task for node:" << node->fullname_with_scope() << " input index:" << i
                    << " ptr from parameter input:" << input->fullname_with_scope();
    } else if (input->isa<CNode>()) {
      // 2. Input which is graph output.
      if (std::find_if(output_with_indexs.begin(), output_with_indexs.end(),
                       [&input_with_index](const KernelWithIndex &output) {
                         const auto &real_output = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
                         return real_output == input_with_index;
                       }) != output_with_indexs.end()) {
        zero_copy_tasks->emplace_back(std::make_shared<tasksink::CNodeZeroCopyTask>(
          input, input_with_index.second, task->Args(), i * sizeof(void *), task->task_name()));
        node_to_offset->emplace(node, i);
        MS_LOG(DEBUG) << "Add zero copy task for node:" << node->fullname_with_scope() << " input index:" << i
                      << " ptr from cnode input:" << input->fullname_with_scope()
                      << " cnode index:" << input_with_index.second;
      } else {
        // 3. Input which is a ref node whose input is a parameter, like:
        // refnode(parameter, node1)
        // node2(refnode)
        // the input of node2 should be replaced.
        AnfNodePtr parameter = nullptr;
        bool is_parameter_ref_input = IsParameterInputRefNode(input_with_index, ref_node_map, &parameter);
        if (is_parameter_ref_input && parameter != nullptr) {
          zero_copy_tasks->emplace_back(std::make_shared<tasksink::ParameterZeroCopyTask>(
            parameter, task->Args(), i * sizeof(void *), task->task_name()));
          MS_LOG(DEBUG) << "Add zero copy task for node:" << node->fullname_with_scope() << " input index:" << i
                        << " ptr from parameter input:" << parameter->fullname_with_scope();
          node_to_offset->emplace(node, i);
        }
      }
    }
  }
}

void GenerateZeroCopyTaskForOutput(const AnfNodePtr &node, const TaskPtr &task, const session::KernelGraph &graph,
                                   std::vector<ZeroCopyTaskPtr> *zero_copy_tasks,
                                   std::set<std::pair<AnfNodePtr, size_t>> *node_to_offset,
                                   std::set<KernelWithIndex> *zero_copy_ref_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(zero_copy_tasks);
  MS_EXCEPTION_IF_NULL(node_to_offset);

  auto input_num = common::AnfAlgo::GetInputTensorNum(node);
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph.output());
  const auto &ref_node_map = graph.GetRefMap();

  for (size_t i = 0; i < output_num; ++i) {
    bool is_output_zero_copy = IsOutputZeroCopy(KernelWithIndex(node, i), output_with_indexs, *zero_copy_ref_nodes);
    if (is_output_zero_copy) {
      // 4. Output of graph.
      // 5. Output which is input of ref node.
      zero_copy_tasks->emplace_back(std::make_shared<tasksink::CNodeZeroCopyTask>(
        node, i, task->Args(), (input_num + i) * sizeof(void *), task->task_name()));
      MS_LOG(DEBUG) << "Add zero copy task for node:" << node->fullname_with_scope() << " output index:" << i
                    << " output index:" << i;
      node_to_offset->emplace(node, i + input_num);
    }

    const auto ref_iter = ref_node_map.find(KernelWithIndex(node, i));
    if (ref_iter != ref_node_map.end() && ref_iter->second.first != nullptr) {
      if (is_output_zero_copy && ref_iter->second.first->isa<CNode>()) {
        // 6. Input of ref output node.
        size_t input_index = FetchInputNumByInputNode(node, ref_iter->second);
        zero_copy_tasks->emplace_back(
          std::make_shared<tasksink::CNodeZeroCopyTask>(ref_iter->second.first, ref_iter->second.second, task->Args(),
                                                        input_index * sizeof(void *), task->task_name()));
        MS_LOG(DEBUG) << "Add zero copy task for node:" << node->fullname_with_scope() << " input index:" << i
                      << " ptr from cnode input:" << ref_iter->second.first->fullname_with_scope()
                      << " cnode index:" << ref_iter->second.second;
        node_to_offset->emplace(node, input_index);
        zero_copy_ref_nodes->emplace(ref_iter->second);
      } else if (ref_iter->second.first->isa<Parameter>()) {
        // 7. Ref output of Parameter input.
        zero_copy_tasks->emplace_back(std::make_shared<tasksink::ParameterZeroCopyTask>(
          ref_iter->second.first, task->Args(), (input_num + i) * sizeof(void *), task->task_name()));
        MS_LOG(DEBUG) << "Add zero copy task for node:" << node->fullname_with_scope() << " output index:" << i
                      << " ptr from parameter input:" << ref_iter->second.first->fullname_with_scope();
        node_to_offset->emplace(node, input_num + i);
      }
    }
  }
}
}  // namespace

bool RtModelZeroCopy::GenerateZeroCopyTaskForSubGraphSink(const session::KernelGraph &graph) {
  std::vector<ZeroCopyTaskPtr> zero_copy_tasks;
  auto task_lists = ge::model_runner::ModelRunner::Instance().GetTaskList(graph.graph_id());
  std::map<std::string, TaskPtr> op_name_to_task;
  std::transform(task_lists.begin(), task_lists.end(), std::inserter(op_name_to_task, op_name_to_task.end()),
                 [](const TaskPtr &task) { return std::make_pair(task->task_name(), task); });

  const auto &nodes = graph.execution_order();
  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph.output());
  const auto &ref_node_map = graph.GetRefMap();
  // Collect all the zero task node with its offset, if the task is an input copy the offset is the index of input,
  // if is output, it is the index of output add its input num.
  std::set<std::pair<AnfNodePtr, size_t>> node_to_offset;
  // Record the node as an input of ref node, whose output should be replaced.
  std::set<KernelWithIndex> zero_copy_ref_nodes;
  for (auto iter = nodes.rbegin(); iter != nodes.rend(); ++iter) {
    const auto &node = *iter;
    MS_EXCEPTION_IF_NULL(node);
    if (NeedSkipZeroCopy(node)) {
      continue;
    }

    MS_EXCEPTION_IF_NULL(node);
    auto op_name = node->UniqueName();
    auto task_iter = op_name_to_task.find(op_name);
    if (task_iter == op_name_to_task.end()) {
      MS_LOG(EXCEPTION) << "Cannot found task of op " << op_name;
    }
    auto task = task_iter->second;
    MS_EXCEPTION_IF_NULL(task);
    GenerateZeroCopyTaskForInput(node, task, graph, &zero_copy_tasks, &node_to_offset);
    GenerateZeroCopyTaskForOutput(node, task, graph, &zero_copy_tasks, &node_to_offset, &zero_copy_ref_nodes);
  }
  MS_LOG(INFO) << "Generate zero copy task num:" << zero_copy_tasks.size() << " for graph:" << graph.ToString();
  CheckZeroCopyTaskValid(graph, node_to_offset);
  auto iter = graph_zero_copy_tasks_.try_emplace(graph.graph_id(), zero_copy_tasks);
  if (!iter.second) {
    MS_LOG(ERROR) << "Generate ZeroCopyTask failed, Duplicate graph id " << graph.graph_id();
    return false;
  }
  return true;
}

bool RtModelZeroCopy::GenerateZeroCopyTasks(const session::KernelGraph &graph) {
  if (EnableZeroCopyForSubgraphSink(graph)) {
    return GenerateZeroCopyTaskForSubGraphSink(graph);
  }
  if (!graph.has_flag(kFlagPyNativeRunInGraph)) {
    MS_LOG(INFO) << "RtModelZeroCopy is not enabled";
    return true;
  }

  std::vector<ZeroCopyTaskPtr> zero_copy_tasks;
  auto task_lists = ge::model_runner::ModelRunner::Instance().GetTaskList(graph.graph_id());
  std::map<std::string, TaskPtr> op_name_to_task;
  std::transform(task_lists.begin(), task_lists.end(), std::inserter(op_name_to_task, op_name_to_task.end()),
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
      auto input_index_in_graph = AnfAlgo::GetInputGraphIdxByKernelIdx(node, i);
      auto input = common::AnfAlgo::GetPrevNodeOutput(node, input_index_in_graph, true).first;
      MS_EXCEPTION_IF_NULL(input);
      if (input->isa<Parameter>()) {
        zero_copy_tasks.emplace_back(std::make_shared<tasksink::ParameterZeroCopyTask>(
          input, task->Args(), i * sizeof(void *), task->task_name()));
        MS_LOG(INFO) << "Generate ZeroCopyTask for Node " << node->fullname_with_scope() << " Parameter "
                     << input->DebugString();
      } else if (IsForwardOutputValueNode(input)) {
        zero_copy_tasks.emplace_back(std::make_shared<tasksink::ValueNodeZeroCopyTask>(
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
  if (!graph.has_flag(kFlagPyNativeRunInGraph) && !EnableZeroCopyForSubgraphSink(graph)) {
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

  if (rtStreamSynchronize(stream) != RT_ERROR_NONE) {
    MS_LOG(WARNING) << "Sync stream for graph:" << graph.ToString() << " failed.";
    return true;
  }

  // If the zero copy in graph mode is enabled, the input and output addr in task may not be same as addr in graph,
  // so skip the addr check.
  if (!graph.has_flag(kFlagEnableZeroCopyInGraph)) {
    MS_LOG(INFO) << "Check rtMode valid " << (CheckRtModelValid(graph));
  }
  return true;
}

bool RtModelZeroCopy::CheckRtModelValid(const session::KernelGraph &graph) {
  auto graph_id = graph.graph_id();
  auto tasks = ge::model_runner::ModelRunner::Instance().GetTaskList(graph_id);
  std::map<std::string, TaskPtr> op_name_to_task;
  std::transform(tasks.begin(), tasks.end(), std::inserter(op_name_to_task, op_name_to_task.end()),
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
