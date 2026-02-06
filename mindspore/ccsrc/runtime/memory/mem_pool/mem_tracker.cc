/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "include/runtime/memory/mem_pool/mem_tracker.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>

#include "runtime/memory/mem_pool/tracker_graph.h"
#include "include/runtime/memory/mem_pool/dynamic_mem_pool.h"
#include "ir/dtype.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/distributed_meta.h"
#include "device_address/device_type.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace device {
namespace tracker {
namespace {
constexpr int64_t kUserTaskNumThreshold = 1e6;
constexpr size_t kLogThreshold = 10;
constexpr size_t kLogPersentage = 100;
}  // namespace

void MemoryTrackerEnabled::AddTask(const std::string &task_name, const std::string &node_name,
                                   const std::string &graph_name, const bool to_graph, const std::string &file_name,
                                   size_t line_num) {
  std::string python_stack = GetPythonStackStr();

  std::lock_guard lock(mutex_);
  if (!is_init_enable_hccl_) {
    // MS_CTX_ENABLE_HCCL will be reset when the process is destroyed.
    // Therefore, record the enable_hccl when AddTask for the first time.
    auto ms_context = MsContext::GetInstance();
    enable_hccl_ = ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL);
    is_init_enable_hccl_ = true;
  }

  auto task_info = std::make_shared<TaskInfo>();
  MS_EXCEPTION_IF_NULL(task_info);
  task_info->task_name = task_name;
  task_info->node_name = node_name;
  task_info->graph_name = graph_name;
  task_info->file_name = file_name;
  task_info->line_num = line_num;
  task_info->time_stamp = time_stamp_;
  task_info->python_stack = python_stack;
  task_info->attrs[kStreamId] = "0";
  task_map_[task_name] = task_info;
  if (to_graph) {
    time_stamp_++;
    task_list_.push_back(task_info);
    graph::TrackerGraph::getInstance().AddOperator(task_info);
  }
}
void MemoryTrackerEnabled::AddTask(const std::string &task_name, const std::string &node_name,
                                   const std::string &graph_name, const std::string &file_name, size_t line_num) {
  AddTask(task_name, node_name, graph_name, true, file_name, line_num);
}

void MemoryTrackerEnabled::UpdateTask(const std::string &task_name,
                                      const std::unordered_map<std::string, std::string> &attrs) {
  std::lock_guard lock(mutex_);
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker UpdateTask failed, task_name:" << task_name << " not found";
    return;
  }
  for (const auto &attr : attrs) {
    iter->second->attrs[attr.first] = attr.second;
  }
}

void MemoryTrackerEnabled::CacheLastTask() {
  std::lock_guard lock(mutex_);
  if (task_list_.empty()) {
    cache = nullptr;
    return;
  }
  cache = task_list_.back();
  task_list_.pop_back();
  graph::TrackerGraph::getInstance().CacheLastTask();
}

void MemoryTrackerEnabled::EmptyCache() {
  std::lock_guard lock(mutex_);
  if (cache == nullptr) {
    return;
  }
  task_list_.push_back(cache);
  cache = nullptr;
  graph::TrackerGraph::getInstance().EmptyCache();
}

void MemoryTrackerEnabled::AddNestedTask(const std::string &task_name, const std::string &node_name,
                                         const std::string &graph_name, const std::string &file_name, size_t line_num) {
  {
    // lock scope
    std::lock_guard lock(mutex_);
    nested_num_++;
    if (nested_num_ != 1) {
      return;
    }
  }
  AddTask(task_name, node_name, graph_name, file_name, line_num);
}

void MemoryTrackerEnabled::DelNestedTask() {
  std::lock_guard lock(mutex_);
  nested_num_--;
}

void MemoryTrackerEnabled::MarkTensorAsInput(const std::string &task_name, const std::string &device_name,
                                             DeviceMemPtr device_ptr, TypeId dtype, const ShapeVector &shape,
                                             TensorStorageInfoPtr tensor_info, const std::string &file_name,
                                             size_t line_num) {
  if (device_name == "CPU" || device_ptr == nullptr) {
    return;
  }
  UseMemBlock(task_name, device_ptr, file_name, line_num);
  std::lock_guard lock(mutex_);
  if (nested_num_ > 1) {
    return;
  }
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsInput failed, task_name:" << task_name << " not found";
    return;
  }
  auto task_info = iter->second;
  auto mem_block_iter = FindMemBlock(device_ptr, file_name, line_num);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsInput failed, device_ptr:" << device_ptr << " not found";
    return;
  }
  auto mem_block = mem_block_iter->second;
  auto input_tensor = graph::TrackerGraph::getInstance().AddTensor(mem_block, device_ptr, dtype, shape, tensor_info);
  graph::TrackerGraph::getInstance().AddOperatorInput(task_info, input_tensor);
}

void MemoryTrackerEnabled::MarkTensorAsOutput(const std::string &task_name, const std::string &device_name,
                                              DeviceMemPtr device_ptr, TypeId dtype, const ShapeVector &shape,
                                              TensorStorageInfoPtr tensor_info, const std::string &file_name,
                                              size_t line_num) {
  std::lock_guard lock(mutex_);
  if (device_name == "CPU" || device_ptr == nullptr) {
    return;
  }
  if (nested_num_ > 1) {
    return;
  }
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsOutput failed, task_name:" << task_name << " not found";
    return;
  }
  auto task_info = iter->second;
  auto mem_block_iter = FindMemBlock(device_ptr, file_name, line_num);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsOutput failed, device_ptr:" << device_ptr << " not found";
    return;
  }
  auto mem_block = mem_block_iter->second;
  auto output_tensor = graph::TrackerGraph::getInstance().AddTensor(mem_block, device_ptr, dtype, shape, tensor_info);
  graph::TrackerGraph::getInstance().AddOperatorOutput(task_info, output_tensor);
}

MemInfoPtr MemoryTrackerEnabled::NewMemInfo(const std::string &task_name, MemType type, size_t size,
                                            const void *kernel_tensor, const std::string &file_name, size_t line_num) {
  auto mem_info = std::make_shared<MemInfo>();
  MS_EXCEPTION_IF_NULL(mem_info);
  mem_info->type = type;
  mem_info->size = size;
  mem_info->kernel_tensor = kernel_tensor;
  mem_info->file_name = file_name;
  mem_info->line_num = line_num;
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker AddMemInfo failed, task_name:" << task_name << " not found, " << file_name << ":"
                  << line_num;
    return nullptr;
  }

  const auto &node_name = iter->second->node_name;
  DynamicMemAllocatorDebugInfo::SetDebugInfo(node_name, type);

  mem_info->producer_task = iter->second;
  mem_info_list_.push_back(mem_info);
  return mem_info;
}

void MemoryTrackerEnabled::AddMemInfo(const std::string &task_name, MemType type, size_t size,
                                      DeviceAddress *device_address, const std::string &file_name, size_t line_num) {
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetDeviceType() == DeviceType::kCPU) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);

  auto mem_info = NewMemInfo(task_name, type, size, nullptr, file_name, line_num);
  device_address_mem_map[device_address] = mem_info;
  if (MS_UNLIKELY(enable_memory_debug_info_)) {
    auto &&iter = task_map_.find(task_name);
    if (iter != task_map_.end()) {
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(iter->second->node_name, type);
    } else {
      MS_LOG(WARNING) << "Find task : " << task_name << " failed, file name : " << file_name
                      << ", line num : " << line_num << ".";
    }
  }
}

void MemoryTrackerEnabled::AddCompileTimeMemInfo(const std::string &task_name, size_t size, DeviceMemPtr device_ptr,
                                                 MemType mem_type, const std::string &file_name, size_t line_num) {
  std::lock_guard lock(mutex_);
  auto mem_info = std::make_shared<MemInfo>();
  MS_EXCEPTION_IF_NULL(mem_info);
  mem_info->type = mem_type;
  mem_info->size = size;
  mem_info->file_name = file_name;
  mem_info->line_num = line_num;
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker AddCompileTimeMemInfo failed, task_name:" << task_name << " not found, "
                  << file_name << ":" << line_num;
    return;
  }
  mem_info->producer_task = iter->second;
  auto mem_block_iter = FindMemBlock(device_ptr, file_name, line_num);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker AddCompileTimeMemInfo failed, device_ptr:" << device_ptr << " not found, "
                  << file_name << ":" << line_num;
    return;
  }
  mem_info->mem_block = mem_block_iter->second;
  mem_info->mem_block->is_bind = true;
  mem_info->mem_block->mem_info = mem_info;
  mem_info_list_.push_back(mem_info);
}

void MemoryTrackerEnabled::BindDevicePtr(DeviceAddress *device_address, DeviceMemPtr device_ptr,
                                         const std::string &file_name, size_t line_num) {
  if (device_address == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_address->GetDeviceType() == DeviceType::kCPU) {
    return;
  }
  MemInfoPtr mem_info{nullptr};
  auto iter = device_address_mem_map.find(device_address);
  if (iter == device_address_mem_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, device_address:" << device_address << " not found, "
                  << file_name << ":" << line_num;
    return;
  }
  mem_info = iter->second;
  if (mem_info == nullptr) {
    MS_LOG(ERROR) << "BindDevicePtr failed, mem_info is nullptr, " << file_name << ":" << line_num << ".";
    return;
  }

  auto mem_block_iter = FindMemBlock(device_ptr, file_name, line_num);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, device_addr:" << device_ptr << " not found, " << file_name
                  << ":" << line_num;
    return;
  }
  mem_info->mem_block = mem_block_iter->second;
  mem_info->mem_block->is_bind = true;
  mem_info->mem_block->mem_info = mem_info;
}

void MemoryTrackerEnabled::AllocMemBlock(DeviceMemPtr device_addr, size_t size, const std::string &pool_name,
                                         size_t actual_peak_memory, size_t in_used_size, uint32_t stream_id,
                                         bool is_persistent, bool is_small_pool) {
  std::lock_guard lock(mutex_);
  time_stamp_++;
  auto mem_block = std::make_shared<MemBlockInfo>();
  MS_EXCEPTION_IF_NULL(mem_block);
  mem_block->device_addr = device_addr;
  mem_block->start_time_stamp = time_stamp_;
  mem_block->actual_peak_memory = actual_peak_memory;
  mem_block->size = size;
  mem_block->used_size = in_used_size;
  mem_block->pool_name = pool_name;
  mem_block->stream_id = stream_id;
  mem_block->is_persistent = is_persistent;
  mem_block->is_small = is_small_pool;
  device_mem_block_map[device_addr] = mem_block;
  mem_block_list_.emplace_back(mem_block);
  // mem_block need to dump again, after mem_block_list_ changed
  has_dump = false;
}

std::map<DeviceMemPtr, MemBlockInfoPtr>::iterator MemoryTrackerEnabled::FindMemBlock(DeviceMemPtr device_ptr,
                                                                                     const std::string &file_name,
                                                                                     size_t line_num) {
  auto it = device_mem_block_map.upper_bound(device_ptr);
  if (it == device_mem_block_map.begin()) {
    return device_mem_block_map.end();
  }
  --it;
  DeviceMemPtr right_border = static_cast<const uint8_t *>(it->first) + it->second->size;
  if (device_ptr < right_border) {
    return it;
  }
  MS_LOG(ERROR) << "MemoryTracker FindMemBlock failed, device_ptr:" << device_ptr
                << " not found, right border: " << right_border << ", " << file_name << ":" << line_num;
  return device_mem_block_map.end();
}

void MemoryTrackerEnabled::FreeMemBlock(DeviceMemPtr device_addr, size_t in_used_size, size_t total_size) {
  std::lock_guard lock(mutex_);
  time_stamp_++;
  auto iter = device_mem_block_map.find(device_addr);
  if (iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker FreeMemBlock failed, device_addr:" << device_addr << " not found";
    return;
  }
  iter->second->end_time_stamp = time_stamp_;
  device_mem_block_map.erase(iter);
}

void MemoryTrackerEnabled::UseMemBlock(const std::string &task_name, DeviceMemPtr device_addr,
                                       const std::string &file_name, size_t line_num) {
  std::lock_guard lock(mutex_);
  auto iter = FindMemBlock(device_addr, file_name, line_num);
  if (iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, device_addr:" << device_addr << " not found, " << file_name
                  << ":" << line_num;
    return;
  }
  if (iter->second->pool_name == "CPU") {
    return;
  }
  auto task_iter = task_map_.find(task_name);
  if (task_iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, task_name:" << task_name << " not found, " << file_name << ":"
                  << line_num;
    return;
  }
  auto mem_info = iter->second->mem_info.lock();
  if (mem_info == nullptr) {
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, mem_info is null, " << file_name << ":" << line_num
                  << ", addr:" << device_addr;
    return;
  }
  mem_info->user_tasks.push_back(task_iter->second);
}

namespace {
auto task_list_to_str = [](const std::vector<TaskInfoPtr> &task_list) -> std::string {
  std::stringstream ss;
  ss << "{";
  std::string result = "";
  if (!task_list.empty()) {
    result = std::accumulate(
      std::next(task_list.begin()), task_list.end(), std::to_string(task_list.front()->time_stamp),
      [](const std::string &ss, TaskInfoPtr task) { return ss + "-" + std::to_string(task->time_stamp); });
  }
  ss << result;
  ss << "}";
  return ss.str();
};

const std::vector<std::pair<std::string, std::function<void(const MemBlockInfoPtr &, std::ofstream &)>>> block_csv = {
  {"start_time_stamp",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->start_time_stamp; }},
  {"end_time_stamp", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->end_time_stamp; }},
  {"device_addr", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->device_addr; }},
  {"stream_id", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->stream_id; }},
  {"pool_type", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->pool_name; }},
  {"size", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->size; }},
  {"actual_used_memory", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->used_size; }},
  {"actual_peak_memory",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->actual_peak_memory; }},
  {"file_name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << mem_info->file_name;
     }
   }},
  {"line_num",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << mem_info->line_num;
     }
   }},
  {"type",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << MemTypeToStr(mem_info->type);
     }
   }},
  {"producer_task",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->time_stamp;
     }
   }},
  {"task_name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->task_name;
     }
   }},
  {"node_name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->node_name;
     }
   }},
  {"graph_name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->graph_name;
     }
   }},
  {"user_tasks",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     static bool is_simple_tracker = memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocSimpleTracker);
     if (is_simple_tracker) {
       return;
     }
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << task_list_to_str(mem_info->user_tasks);
     }
   }},
  {"last_user_task",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       std::string result = "";
       if (!mem_info->user_tasks.empty()) {
         result = std::to_string(mem_info->user_tasks.back()->time_stamp);
       }
       oss << result;
     }
   }},
  {"python_stack",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->python_stack;
     }
   }},
  {"is_persistent", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->is_persistent; }},
  {"is_small", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->is_small; }},
};

const std::vector<std::pair<std::string, std::function<void(const TaskInfoPtr &, std::ofstream &)>>> task_csv = {
  {"time_stamp", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->time_stamp; }},
  {"task_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->task_name; }},
  {"node_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->node_name; }},
  {"graph_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->graph_name; }},
  {"file_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->file_name; }},
  {"line_num", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->line_num; }},
  {"python_stack", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->python_stack; }},
};
}  // namespace

void MemoryTrackerEnabled::Dump(size_t rank_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (has_dump) {
    return;
  }
  has_dump = true;

  // Check if need dump
  if (task_list_.empty() && !graph::TrackerGraph::getInstance().NeedDump()) {
    MS_LOG(WARNING) << "MemoryTracker skip Dump, since no data has been collected";
    return;
  }

  auto graph_path = memory::mem_pool::GeneratePath(rank_id, "tracker_graph", "ir");
  if (graph_path.empty()) {
    MS_LOG(ERROR) << "Generate graph path failed, rank_id : " << rank_id << ".";
    return;
  }

  int64_t user_task_num = 0;
  for (auto &mem_block : mem_block_list_) {
    MS_EXCEPTION_IF_NULL(mem_block);
    if (mem_block->pool_name == "CPU") {
      continue;
    }
    auto mem_info = mem_block->mem_info.lock();
    if (mem_info) {
      user_task_num += static_cast<int64_t>(mem_info->user_tasks.size());
    }
  }
  if (user_task_num >= kUserTaskNumThreshold &&
      !memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocSimpleTracker)) {
    MS_LOG(WARNING)
      << "The number of user tasks is too large: " << user_task_num
      << ", the speed of dump will be slow, please set MS_ALLOC_CONF=\"simple_tracker:True\" to speed up the dump";
  }
  MS_LOG(WARNING) << "MemoryTracker Dump start, task num: " << task_list_.size()
                  << ", mem block num: " << mem_block_list_.size() << ", user task num: " << user_task_num;
  graph::TrackerGraph::getInstance().Dump(graph_path);

  MS_LOG(INFO) << "MemoryTracker Dump start";
  DumpMemoryBlock(rank_id);
  DumpTaskFile(rank_id);
  MS_LOG(INFO) << "MemoryTracker Dump end";
}

void MemoryTrackerEnabled::DumpMemoryBlock(size_t rank_id) {
  std::string memory_block_csv_path = memory::mem_pool::GeneratePath(rank_id, "memory_block", "csv");
  if (memory_block_csv_path.empty()) {
    MS_LOG(ERROR) << "Generate memory block csv path failed, rank_id : " << rank_id << ".";
    return;
  }

  MS_LOG(WARNING) << "memory block csv path : " << memory_block_csv_path << ".";
  std::ofstream block_file(memory_block_csv_path);
  if (!block_file) {
    MS_LOG(EXCEPTION) << "Open file " << memory_block_csv_path << " failed.";
  }
  size_t not_bind_size = 0;
  for (const auto &csv : block_csv) {
    block_file << csv.first << ",";
  }
  block_file << "\n";
  size_t log_threshold = mem_block_list_.size() / kLogThreshold;
  size_t i = 0;
  for (auto &mem_block : mem_block_list_) {
    i++;
    if (mem_block->pool_name == "CPU") {
      continue;
    }
    for (const auto &csv : block_csv) {
      csv.second(mem_block, block_file);
      block_file << ",";
    }
    if (!mem_block->is_bind) {
      not_bind_size += mem_block->size;
    }
    block_file << "\n";
    // print log
    if (i > log_threshold) {
      MS_LOG(WARNING) << "MemoryTracker MemBlock Dump progress: " << (i + 1) * kLogPersentage / mem_block_list_.size()
                      << "%";
      log_threshold += mem_block_list_.size() / kLogThreshold;
    }
  }
  block_file.close();
  MS_LOG(INFO) << "Not bind size : " << not_bind_size << ".";
}

void MemoryTrackerEnabled::DumpTaskFile(size_t rank_id) {
  std::string task_csv_path = memory::mem_pool::GeneratePath(rank_id, "task", "csv");
  if (task_csv_path.empty()) {
    MS_LOG(ERROR) << "Generate task csv path failed, rank_id : " << rank_id << ".";
    return;
  }

  MS_LOG(WARNING) << "task csv path: " << task_csv_path << ".";
  std::ofstream task_file(task_csv_path);
  if (!task_file) {
    MS_LOG(EXCEPTION) << "Open file " << task_csv_path << " failed.";
  }
  for (const auto &csv : task_csv) {
    task_file << csv.first << ",";
  }
  task_file << "\n";
  size_t log_threshold = task_list_.size() / kLogThreshold;
  size_t i = 0;
  for (auto &task : task_list_) {
    i++;
    for (const auto &csv : task_csv) {
      csv.second(task, task_file);
      task_file << ",";
    }
    task_file << "\n";
    // print log
    if (i > log_threshold) {
      MS_LOG(WARNING) << "MemoryTracker Task Dump progress: " << (i + 1) * kLogPersentage / task_list_.size() << "%";
      log_threshold += task_list_.size() / kLogThreshold;
    }
  }
  task_file.close();
}

void MemoryTrackerEnabled::UpdateProfilingPos() {
  std::lock_guard<std::mutex> lock(mutex_);
  last_profiling_pos_ = mem_block_list_.size();
}

void MemoryTrackerDisabled::AddTask(const std::string &task_name, const std::string &node_name,
                                    const std::string &graph_name, const std::string &file_name, size_t line_num) {
  AddTask(task_name, node_name, graph_name, false, file_name, line_num);
}

void MemoryTrackerDisabled::AddTask(const std::string &task_name, const std::string &node_name,
                                    const std::string &graph_name, const bool to_graph, const std::string &file_name,
                                    size_t line_num) {
  if (MS_UNLIKELY(enable_memory_debug_info_)) {
    LockGuard lock(lock_);
    task_map_[task_name] = node_name;
  }
}

void MemoryTrackerDisabled::AddNestedTask(const std::string &task_name, const std::string &node_name,
                                          const std::string &graph_name, const std::string &file_name,
                                          size_t line_num) {
  AddTask(task_name, node_name, graph_name, false, file_name, line_num);
}

void MemoryTrackerDisabled::AddMemInfo(const std::string &task_name, MemType type, size_t size,
                                       DeviceAddress *device_address, const std::string &file_name,
                                       const size_t line_num) {
  if (MS_UNLIKELY(enable_memory_debug_info_)) {
    LockGuard lock(lock_);
    auto &&iter = task_map_.find(task_name);
    if (iter == task_map_.end()) {
      MS_LOG(WARNING) << "Find task : " << task_name << " failed, file name : " << file_name
                      << ", line num : " << line_num << ".";
    } else {
      DynamicMemAllocatorDebugInfo::SetDebugInfo(iter->second, type);
    }
  }
}
}  // namespace tracker
}  // namespace device
}  // namespace mindspore
