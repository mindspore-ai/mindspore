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
#include "include/backend/device_type.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "include/common/utils/utils.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace device {
namespace tracker {
constexpr int64_t kIllegalStartTimeStamp = -1L;
namespace {
std::string GetRankID() {
  uint32_t rank_id = 0;
#if !defined(BUILD_LITE)
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
    rank_id = CommManager::GetInstance().GetRank();
  }
#endif
  return std::to_string(rank_id);
}

AllocatorType GetAllocatorType(MemType mem_type) {
  static std::map<MemType, device::AllocatorType> mem_allocator_type_map = {
    {MemType::kWeight, AllocatorType::kWeight},
    {MemType::kConstantValue, AllocatorType::kConstantValue},
    {MemType::kKernel, AllocatorType::kConstantValue},
    {MemType::kGraphOutput, AllocatorType::kGraphOutput},
    {MemType::kSomas, AllocatorType::kConstantValue},
    {MemType::kInSideSomas, AllocatorType::kConstantValue},
    {MemType::kSomasOutput, AllocatorType::kKernelOutput},
    {MemType::kGeConst, AllocatorType::kConstantValue},
    {MemType::kBatchMemory, AllocatorType::kConstantValue},
    {MemType::kContinuousMemory, AllocatorType::kConstantValue},
    {MemType::kPyNativeInput, AllocatorType::kConstantValue},
    {MemType::kPyNativeOutput, AllocatorType::kKernelOutput},
    {MemType::kGeFeatureMemory, AllocatorType::kConstantValue},
    {MemType::kWorkSpace, AllocatorType::kWorkspace},
    {MemType::kOther, AllocatorType::kOther}};

  auto iter = mem_allocator_type_map.find(mem_type);
  if (iter == mem_allocator_type_map.end()) {
    MS_LOG(WARNING) << "Not found mem_type:" << mem_type << " in mem_allocator_type_map.";
    return AllocatorType::kOther;
  }
  return iter->second;
}
}  // namespace

std::pair<std::string, std::string> MemoryTrackerEnabled::GetPath() {
  std::string block_csv_path;
  std::string task_csv_path;

  auto ms_context = MsContext::GetInstance();
  auto trace_path = ms_context->get_param<std::string>(MS_CTX_PROF_MEM_OUTPUT_PATH);
  if (trace_path.empty()) {
    trace_path = "./";
  }

  if (enable_hccl_) {
    block_csv_path = trace_path + "/rank_" + GetRankID() + "/memory_block.csv";
    task_csv_path = trace_path + "/rank_" + GetRankID() + "/task.csv";
  } else {
    block_csv_path = trace_path + "/memory_block.csv";
    task_csv_path = trace_path + "/task.csv";
  }
  return std::make_pair(block_csv_path, task_csv_path);
}

void MemoryTrackerEnabled::AddTask(const std::string &task_name, const std::string &node_name,
                                   const std::string &graph_name, const std::string &file_name, size_t line_num) {
  std::string python_stack;
  if (WithPythonStack()) {
    python_stack = GetPythonStackStr();
  }

  std::lock_guard lock(mutex_);
  if (!is_init_enable_hccl_) {
    // MS_CTX_ENABLE_HCCL will be reset when the process is destroyed.
    // Therefore, record the enable_hccl when AddTask for the first time.
    auto ms_context = MsContext::GetInstance();
    enable_hccl_ = ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL);
    is_init_enable_hccl_ = true;
  }

  time_stamp_++;
  auto task_info = std::make_shared<TaskInfo>();
  MS_EXCEPTION_IF_NULL(task_info);
  task_info->task_name = task_name;
  task_info->node_name = node_name;
  task_info->graph_name = graph_name;
  task_info->file_name = file_name;
  task_info->line_num = line_num;
  task_info->time_stamp = time_stamp_;
  task_info->python_stack = python_stack;
  task_map_[task_name] = task_info;
  task_list_.push_back(task_info);
}

MemInfoPtr MemoryTrackerEnabled::NewMemInfo(const std::string &task_name, MemType type, size_t size,
                                            KernelTensorPtr kernel_tensor, const std::string &file_name,
                                            size_t line_num) {
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
  DynamicMemAllocatorDebugInfo::SetDebugInfo(node_name, GetAllocatorType(type));

  mem_info->producer_task = iter->second;
  mem_info_list_.push_back(mem_info);
  return mem_info;
}

void MemoryTrackerEnabled::AddMemInfoForKernelTensor(const std::string &task_name, MemType type, size_t size,
                                                     KernelTensorPtr kernel_tensor, const std::string &file_name,
                                                     size_t line_num) {
  auto mem_info = NewMemInfo(task_name, type, size, kernel_tensor, file_name, line_num);
  if (mem_info != nullptr) {
    kernel_tensor_mem_map[kernel_tensor] = mem_info;
  }
}

void MemoryTrackerEnabled::AddMemInfo(const std::string &task_name, MemType type, size_t size,
                                      DeviceAddress *device_address, const std::string &file_name, size_t line_num) {
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetDeviceType() == DeviceType::kCPU) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);

  if (device_address->kernel_tensor() == nullptr) {
    auto mem_info = NewMemInfo(task_name, type, size, nullptr, file_name, line_num);
    device_address_mem_map[device_address] = mem_info;
  } else {
    AddMemInfoForKernelTensor(task_name, type, size, device_address->kernel_tensor().get(), file_name, line_num);
  }
}

void MemoryTrackerEnabled::UpdateMemInfo(const DeviceAddress *device_address, MemType mem_type,
                                         const std::string &file_name, size_t line_num) {
  std::lock_guard lock(mutex_);
  if (device_address->GetDeviceType() == DeviceType::kCPU) {
    return;
  }
  auto kernel_tensor = device_address->kernel_tensor().get();
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
  if (device_address->kernel_tensor() == nullptr) {
    auto iter = device_address_mem_map.find(device_address);
    if (iter == device_address_mem_map.end()) {
      MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, device_address:" << device_address << " not found, "
                    << file_name << ":" << line_num;
      return;
    }
    mem_info = iter->second;
  } else {
    auto iter = kernel_tensor_mem_map.find(device_address->kernel_tensor().get());
    if (iter == kernel_tensor_mem_map.end()) {
      MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, kernel_tensor:" << device_address->kernel_tensor().get()
                    << " not found, " << file_name << ":" << line_num;
      return;
    }
    mem_info = iter->second;
  }

  if (mem_info->type == MemType::kInSideSomas) {
    auto mem_block_info = std::make_shared<MemBlockInfo>();
    MS_EXCEPTION_IF_NULL(mem_block_info);
    mem_block_info->device_addr = device_ptr;
    mem_block_info->size = mem_info->size;
    mem_block_info->start_time_stamp = -1;
    mem_block_info->end_time_stamp = -1;
    mem_block_info->is_bind = true;
    mem_block_info->mem_info = mem_info;
    mem_info->mem_block = mem_block_info;
    device_mem_block_map[device_ptr] = mem_block_info;
    mem_block_list_.push_back(mem_block_info);
    // mem_block need to dump again, after mem_block_list_ changed
    has_dump = false;
    return;
  }
  auto mem_block_iter = device_mem_block_map.find(device_ptr);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, device_ptr:" << device_ptr << " not found, " << file_name
                  << ":" << line_num;
    return;
  }
  mem_info->mem_block = mem_block_iter->second;
  mem_info->mem_block->is_bind = true;
  mem_info->mem_block->mem_info = mem_info;
}

void MemoryTrackerEnabled::UpdateDevicePtrInfo(DeviceMemPtr device_ptr, MemType mem_type, const std::string &task_name,
                                               const std::string &file_name, size_t line_num) {
  std::lock_guard lock(mutex_);
  auto mem_block_iter = device_mem_block_map.find(device_ptr);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker AddCompileTimeMemInfo failed, device_ptr:" << device_ptr << " not found, "
                  << file_name << ":" << line_num;
    return;
  }
  auto mem_info = std::make_shared<MemInfo>();
  MS_EXCEPTION_IF_NULL(mem_info);
  auto task_info = std::make_shared<TaskInfo>();
  MS_EXCEPTION_IF_NULL(task_info);
  task_info->task_name = task_name;
  mem_info->producer_task = task_info;
  mem_info->file_name = file_name;
  mem_info->line_num = line_num;
  mem_info->type = mem_type;
  mem_info->mem_block = mem_block_iter->second;
  mem_info->mem_block->is_bind = true;
  mem_info->mem_block->mem_info = mem_info;
  mem_info_list_.push_back(mem_info);
}

void MemoryTrackerEnabled::AllocMemBlock(DeviceMemPtr device_addr, size_t size, const std::string &pool_name,
                                         size_t actual_peak_memory, size_t in_used_size, size_t total_size,
                                         uint32_t stream_id) {
  std::lock_guard lock(mutex_);
  time_stamp_++;
  auto mem_block = std::make_shared<MemBlockInfo>();
  MS_EXCEPTION_IF_NULL(mem_block);
  mem_block->device_addr = device_addr;
  mem_block->start_time_stamp = time_stamp_;
  mem_block->actual_peak_memory = actual_peak_memory;
  mem_block->size = size;
  mem_block->pool_name = pool_name;
  mem_block->stream_id = stream_id;
  mem_block->real_start_time = GetCurrentUSec();
  mem_block->alloc_in_used_size = in_used_size;
  mem_block->alloc_total_size = total_size;
  device_mem_block_map[device_addr] = mem_block;
  real_device_mem_block_map[device_addr] = mem_block;
  mem_block_list_.emplace_back(mem_block);
  // mem_block need to dump again, after mem_block_list_ changed
  has_dump = false;
}

void MemoryTrackerEnabled::FreeMemBlock(DeviceMemPtr device_addr, size_t in_used_size, size_t total_size) {
  std::lock_guard lock(mutex_);
  time_stamp_++;
  auto iter = real_device_mem_block_map.find(device_addr);
  if (iter == real_device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker FreeMemBlock failed, device_addr:" << device_addr << " not found";
    return;
  }
  iter->second->end_time_stamp = time_stamp_;
  iter->second->real_end_time = GetCurrentUSec();
  iter->second->release_in_used_size = in_used_size;
  iter->second->release_total_size = total_size;
}

void MemoryTrackerEnabled::UseMemBlock(const std::string &task_name, DeviceMemPtr device_addr,
                                       const std::string &file_name, size_t line_num) {
  std::lock_guard lock(mutex_);
  auto iter = device_mem_block_map.find(device_addr);
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
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, mem_info is null, " << file_name << ":" << line_num;
    return;
  }
  mem_info->user_tasks.push_back(task_iter->second);
}

namespace {
constexpr size_t kKBToByte = 1024;
constexpr size_t kMBToKB = 1024;
static const int kPrecisionDigits = 20;

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
  {"python_stack",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->python_stack;
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

const std::vector<std::pair<std::string, std::function<void(const MemBlockInfoPtr &, std::ofstream &)>>> prof_csv = {
  {"Name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->node_name;
     }
   }},
  {"Size(KB)", [](const MemBlockInfoPtr &mem_block,
                  std::ofstream &oss) { oss << (static_cast<float>(mem_block->size) / kKBToByte); }},
  {"Allocation Time(us)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->real_start_time; }},
  {"Duration(us)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     if (mem_block->real_end_time > 0) {
       oss << (mem_block->real_end_time - mem_block->real_start_time);
     }
   }},
  {"Allocation Total Allocated(MB)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     oss << (static_cast<float>(mem_block->alloc_in_used_size) / kKBToByte / kMBToKB);
   }},
  {"Allocation Total Reserved(MB)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     oss << (static_cast<float>(mem_block->alloc_total_size) / kKBToByte / kMBToKB);
   }},
  {"Release Total Allocated(MB)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     oss << (static_cast<float>(mem_block->release_in_used_size) / kKBToByte / kMBToKB);
   }},
  {"Release Total Reserved(MB)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     oss << (static_cast<float>(mem_block->release_total_size) / kKBToByte / kMBToKB);
   }},
  {"Device", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->pool_name; }},
};
}  // namespace

void MemoryTrackerEnabled::Dump() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (has_dump) {
    return;
  }
  has_dump = true;

  auto [block_csv_path, task_csv_path] = GetPath();
  auto block_csv_path_opt = Common::CreatePrefixPath(block_csv_path);
  auto task_csv_path_opt = Common::CreatePrefixPath(task_csv_path);
  if (!block_csv_path_opt.has_value() || !task_csv_path_opt.has_value()) {
    MS_LOG(ERROR) << "Get realpath failed, block_csv_path:" << block_csv_path << ", task_csv_path:" << task_csv_path;
    return;
  }

  MS_LOG(INFO) << "MemoryTracker Dump start";
  ChangeFileMode(block_csv_path_opt.value(), S_IWUSR | S_IRUSR);
  std::ofstream block_file(block_csv_path_opt.value());
  if (!block_file) {
    MS_LOG(EXCEPTION) << "Open file " << block_csv_path_opt.value() << " failed.";
  }
  size_t not_bind_size = 0;
  for (const auto &csv : block_csv) {
    block_file << csv.first << ",";
  }
  block_file << "\n";
  for (auto &mem_block : mem_block_list_) {
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
  }

  ChangeFileMode(task_csv_path_opt.value(), S_IWUSR | S_IRUSR);
  std::ofstream task_file(task_csv_path_opt.value());
  if (!task_file) {
    MS_LOG(EXCEPTION) << "Open file " << task_csv_path_opt.value() << " failed.";
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
  ChangeFileMode(block_csv_path_opt.value(), S_IWUSR | S_IRUSR);
  ChangeFileMode(task_csv_path_opt.value(), S_IWUSR | S_IRUSR);
  MS_LOG(INFO) << "Not bind size, " << not_bind_size;
  MS_LOG(INFO) << "MemoryTracker Dump end";
}

void MemoryTrackerEnabled::UpdateProfilingPos() {
  std::lock_guard<std::mutex> lock(mutex_);
  last_profiling_pos_ = mem_info_list_.size();
}

void MemoryTrackerEnabled::DumpProfilingMemInfo(const std::string &path, const std::string &file_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto csv_path = path + "/" + file_name + "_" + GetRankID() + ".csv";
  auto csv_path_opt = Common::CreatePrefixPath(csv_path);
  if (!csv_path_opt.has_value()) {
    MS_LOG(ERROR) << "Get realpath failed, csv_path:" << csv_path;
    return;
  }

  MS_LOG(INFO) << "MemoryTracker DumpProfilingMemInfo start, last_profiling_pos:" << last_profiling_pos_;
  ChangeFileMode(csv_path_opt.value(), S_IWUSR | S_IRUSR);
  std::ofstream block_file(csv_path_opt.value());
  auto old_file_flags = block_file.flags();
  auto old_precision = block_file.precision();
  block_file.unsetf(std::ios_base::floatfield);
  block_file.precision(kPrecisionDigits);
  for (const auto &csv : prof_csv) {
    block_file << csv.first << ",";
  }
  block_file << "\n";

  for (size_t i = 0; i < mem_block_list_.size(); i++) {
    const auto &mem_block = mem_block_list_[i];
    if (i < last_profiling_pos_) {
      continue;
    }

    if (mem_block->pool_name == "CPU") {
      continue;
    }

    if (mem_block->start_time_stamp == kIllegalStartTimeStamp) {
      MS_LOG(DEBUG) << "Mem block start time stamp is " << kIllegalStartTimeStamp << ".";
      continue;
    }

    for (const auto &csv : prof_csv) {
      csv.second(mem_block, block_file);
      block_file << ",";
    }
    block_file << "\n";
  }

  // Restore file flags and precision
  block_file.flags(old_file_flags);
  block_file.precision(old_precision);
  block_file.close();
  ChangeFileMode(csv_path_opt.value(), S_IWUSR | S_IRUSR);

  // record the last time stamp
  last_profiling_pos_ = mem_block_list_.size();
  MS_LOG(INFO) << "MemoryTracker DumpProfilingMemInfo end, last_profiling_pos:" << last_profiling_pos_;
}

}  // namespace tracker
}  // namespace device
}  // namespace mindspore
