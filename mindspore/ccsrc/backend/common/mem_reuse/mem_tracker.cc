/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "include/backend/mem_reuse/mem_tracker.h"
#include <fstream>
#include "frontend/parallel/group_manager.h"
#include "utils/ms_context.h"
#include "include/common/debug/common.h"
#include "include/common/utils/comm_manager.h"

namespace mindspore {
namespace device {
namespace tracker {
void MemoryTrackerEnabled::AddTask(const std::string &task_name, const std::string &node_name,
                                   const std::string &graph_name, const std::string &file_name, size_t line_num) {
  std::lock_guard<std::mutex> lock(mutex_);
  time_stamp_++;
  auto task_info = std::make_shared<TaskInfo>();
  MS_EXCEPTION_IF_NULL(task_info);
  task_info->task_name = task_name;
  task_info->node_name = node_name;
  task_info->graph_name = graph_name;
  task_info->file_name = file_name;
  task_info->line_num = line_num;
  task_info->time_stamp = time_stamp_;
  task_map_[task_name] = task_info;
  task_list_.push_back(task_info);
}

void MemoryTrackerEnabled::AddMemInfo(const std::string &task_name, MemType type, size_t size,
                                      KernelTensorPtr kernel_tensor, const std::string &file_name, size_t line_num) {
  std::lock_guard<std::mutex> lock(mutex_);
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
    return;
  }
  mem_info->producer_task = iter->second;
  mem_info_list_.push_back(mem_info);
  kernel_tensor_mem_map[kernel_tensor] = mem_info;
}

void MemoryTrackerEnabled::UpdateMemInfo(KernelTensorPtr kernel_tensor, MemType mem_type, const std::string &file_name,
                                         size_t line_num) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = kernel_tensor_mem_map.find(kernel_tensor);
  if (iter == kernel_tensor_mem_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker UpdateMemInfoMemType failed, kernel_tensor:" << kernel_tensor << " not found";
    return;
  }
  iter->second->type = mem_type;
  iter->second->file_name = file_name;
  iter->second->line_num = line_num;
}

void MemoryTrackerEnabled::AddCompileTimeMemInfo(const std::string &task_name, size_t size, DeviceMemPtr device_ptr,
                                                 MemType mem_type, const std::string &file_name, size_t line_num) {
  std::lock_guard<std::mutex> lock(mutex_);
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
  auto mem_block_iter = device_mem_block_map.find(device_ptr);
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

void MemoryTrackerEnabled::BindDevicePtr(KernelTensorPtr kernel_tensor, DeviceMemPtr device_ptr,
                                         const std::string &file_name, size_t line_num) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = kernel_tensor_mem_map.find(kernel_tensor);
  if (iter == kernel_tensor_mem_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, kernel_tensor:" << kernel_tensor << " not found, "
                  << file_name << ":" << line_num;
    return;
  }
  if (iter->second->type == MemType::kInSideSomas) {
    auto mem_block_info = std::make_shared<MemBlockInfo>();
    MS_EXCEPTION_IF_NULL(mem_block_info);
    mem_block_info->device_addr = device_ptr;
    mem_block_info->size = iter->second->size;
    mem_block_info->start_time_stamp = -1;
    mem_block_info->end_time_stamp = -1;
    mem_block_info->is_bind = true;
    mem_block_info->mem_info = iter->second;
    iter->second->mem_block = mem_block_info;
    device_mem_block_map[device_ptr] = mem_block_info;
    mem_block_list_.push_back(mem_block_info);
    return;
  }
  auto mem_block_iter = device_mem_block_map.find(device_ptr);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, device_ptr:" << device_ptr << " not found, " << file_name
                  << ":" << line_num;
    return;
  }
  iter->second->mem_block = mem_block_iter->second;
  iter->second->mem_block->is_bind = true;
  iter->second->mem_block->mem_info = iter->second;
}

void MemoryTrackerEnabled::AllocMemBlock(DeviceMemPtr device_addr, size_t size, const std::string &pool_name,
                                         uint32_t stream_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  time_stamp_++;
  auto mem_block = std::make_shared<MemBlockInfo>();
  MS_EXCEPTION_IF_NULL(mem_block);
  mem_block->device_addr = device_addr;
  mem_block->start_time_stamp = time_stamp_;
  mem_block->size = size;
  mem_block->pool_name = pool_name;
  mem_block->stream_id = stream_id;
  device_mem_block_map[device_addr] = mem_block;
  real_device_mem_block_map[device_addr] = mem_block;
  mem_block_list_.emplace_back(mem_block);
}

void MemoryTrackerEnabled::FreeMemBlock(DeviceMemPtr device_addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  time_stamp_++;
  auto iter = real_device_mem_block_map.find(device_addr);
  if (iter == real_device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker FreeMemBlock failed, device_addr:" << device_addr << " not found";
    return;
  }
  iter->second->end_time_stamp = time_stamp_;
}

void MemoryTrackerEnabled::UseMemBlock(const std::string &task_name, DeviceMemPtr device_addr,
                                       const std::string &file_name, size_t line_num) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = device_mem_block_map.find(device_addr);
  if (iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, device_addr:" << device_addr << " not found, " << file_name
                  << ":" << line_num;
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
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, mem_info is null, " << file_name << ":" << line_num;
    return;
  }
  mem_info->user_tasks.push_back(task_iter->second);
}

namespace {
auto task_list_to_str = [](const std::vector<TaskInfoPtr> &task_list) -> std::string {
  std::stringstream ss;
  ss << "{";
  for (auto &task : task_list) {
    ss << task->time_stamp << "-";
  }
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
       oss << MemTypeToStr.at(mem_info->type);
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
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << task_list_to_str(mem_info->user_tasks);
     }
   }},
};

const std::vector<std::pair<std::string, std::function<void(const TaskInfoPtr &, std::ofstream &)>>> task_csv = {
  {"time_stamp", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->time_stamp; }},
  {"task_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->task_name; }},
  {"node_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->node_name; }},
  {"graph_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->graph_name; }},
  {"file_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->file_name; }},
  {"line_num", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->line_num; }},
};

std::string GetWorldGroup() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string world_group;
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == kAscendDevice) {
    world_group = parallel::HCCL_WORLD_GROUP;
  } else if (backend == kGPUDevice) {
    world_group = parallel::NCCL_WORLD_GROUP;
  } else {
    MS_LOG(EXCEPTION) << "Invalid backend: " << backend;
  }
  return world_group;
}

std::string GetRankID() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto world_group = GetWorldGroup();
  uint32_t rank_id = 0;
  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
      MS_LOG(INFO) << "Failed to get rank id.";
    }
  }
  return std::to_string(rank_id);
}
}  // namespace

void MemoryTrackerEnabled::Dump() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (has_dump) {
    return;
  }
  has_dump = true;

  static const char kMemoryTracePath[] = "MS_MEMORY_TRACE_PATH";
  auto trace_path = common::GetEnv(kMemoryTracePath);

  if (trace_path.empty()) {
    trace_path = "./";
  }

  std::string block_csv_path = trace_path + "/rank_" + GetRankID() + "/memory_block.csv";
  std::string task_csv_path = trace_path + "/rank_" + GetRankID() + "/task.csv";

  Common::CreatePrefixPath(block_csv_path);
  Common::CreatePrefixPath(task_csv_path);

  MS_LOG(INFO) << "MemoryTracker Dump start";

  std::ofstream block_file(block_csv_path);
  if (!block_file) {
    MS_LOG(EXCEPTION) << "Open file " << block_csv_path << " failed.";
  }
  size_t not_bind_size = 0;
  for (const auto &csv : block_csv) {
    block_file << csv.first << ",";
  }
  block_file << "\n";
  for (auto &mem_block : mem_block_list_) {
    for (const auto &csv : block_csv) {
      csv.second(mem_block, block_file);
      block_file << ",";
    }
    if (!mem_block->is_bind) {
      not_bind_size += mem_block->size;
    }
    block_file << "\n";
  }

  std::ofstream task_file(task_csv_path);
  if (!task_file) {
    MS_LOG(EXCEPTION) << "Open file " << task_csv_path << " failed.";
  }
  for (const auto &csv : task_csv) {
    task_file << csv.first << ",";
  }
  task_file << "\n";
  for (auto &task : task_list_) {
    for (const auto &csv : task_csv) {
      csv.second(task, task_file);
      task_file << ",";
    }
    task_file << "\n";
  }

  block_file.close();
  task_file.close();
  MS_LOG(INFO) << "Not bind size, " << not_bind_size;
  MS_LOG(INFO) << "MemoryTracker Dump end";
}

}  // namespace tracker
}  // namespace device
}  // namespace mindspore
