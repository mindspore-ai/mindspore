/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "tools/data_dump/device_statistic/mem_manager.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/utils/common.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {

namespace datadump {
namespace {
constexpr auto kDump = "Dump";
constexpr auto kWorkspaceAddress = "WorkspaceAddress";
}  // namespace

void DumpMemManager::Initialize(const DeviceContext *device_context) {
  std::call_once(init_flag_, [this, device_context]() {
    MS_EXCEPTION_IF_NULL(device_context);
    size_t stream_size = device_context->device_res_manager_->QueryStreamSize();
    MS_VLOG(VL_DUMP) << "Dump start init memory cache, stream size is " << stream_size;
    workspace_cache_.reserve(stream_size);
    for (size_t stream_id = 0; stream_id < stream_size; ++stream_id) {
      workspace_cache_[stream_id] =
        CreateWorkspaceKernelTensor(device_context, max_workspace_size_, kDefaultStreamIndex);
      auto &stream_cache = output_cache_[stream_id];
      stream_cache.reserve(max_output_num_);
      for (size_t i = 0; i < max_output_num_; ++i) {
        stream_cache.emplace_back(CreateOutPutKernelTensor(device_context, kNumberTypeFloat64));
      }
    }
  });
}

void DumpMemManager::ClearCache() {
  std::scoped_lock lock_all(output_cache_mutex_, workspace_cache_mutex_);
  output_cache_.clear();
  output_index_.clear();
  workspace_cache_.clear();
  MS_VLOG(VL_DUMP) << "Clear dump memory cache";
}

void DumpMemManager::Reset() {
  std::lock_guard<std::mutex> lock(output_cache_mutex_);
  for (auto &item : output_index_) {
    item.second = 0;
  }
  MS_VLOG(VL_DUMP) << "Reset dump memory cache index";
}

KernelTensorPtr DumpMemManager::GetWorkSpaceTensor(const DeviceContext *device_context, size_t stream_id, size_t size) {
  MS_EXCEPTION_IF_NULL(device_context);
  Initialize(device_context);
  if (size > max_workspace_size_) {
    MS_VLOG(VL_DUMP) << "Workspace was not obtained from the cache, size exceeds cache maximum. Size is " << size
                     << ", maximum is " << max_workspace_size_;
    return CreateWorkspaceKernelTensor(device_context, size, stream_id);
  }

  std::lock_guard<std::mutex> lock(workspace_cache_mutex_);
  if (workspace_cache_.find(stream_id) == workspace_cache_.end()) {
    workspace_cache_[stream_id] = CreateWorkspaceKernelTensor(device_context, max_workspace_size_, kDefaultStreamIndex);
  }
  workspace_cache_[stream_id]->set_size(size);
  return workspace_cache_[stream_id];
}

KernelTensorPtr DumpMemManager::GetOutputTensor(const DeviceContext *device_context, size_t stream_id,
                                                TypeId dtype_id) {
  MS_EXCEPTION_IF_NULL(device_context);
  Initialize(device_context);
  std::lock_guard<std::mutex> lock(output_cache_mutex_);
  size_t idx = output_index_[stream_id]++;
  if (idx >= max_output_num_) {
    MS_VLOG(VL_DUMP) << "Get output without cache, idx exceeds cache length. Idx is " << idx << ", max length is "
                     << max_output_num_;
    return CreateOutPutKernelTensor(device_context, dtype_id);
  }

  auto &stream_cache = output_cache_[stream_id];
  while (stream_cache.size() < max_output_num_) {
    stream_cache.emplace_back(CreateOutPutKernelTensor(device_context, kNumberTypeFloat64));
  }

  auto output = stream_cache[idx];
  output->set_dtype_id(dtype_id);
  output->set_size(UnitSizeInBytes(dtype_id));
  return output;
}

KernelTensorPtr DumpMemManager::CreateOutPutKernelTensor(const DeviceContext *device_context, const TypeId &dtype_id) {
  MS_EXCEPTION_IF_NULL(device_context);

  const ShapeVector shape = {};
  auto shape_ptr = std::make_shared<abstract::Shape>(shape);
  auto type = std::make_shared<TensorType>(TypeIdToType(dtype_id));
  auto tensor =
    AnfAlgo::CreateKernelTensor(shape_ptr, type, nullptr, nullptr, UnitSizeInBytes(dtype_id),
                                kernel::GetFormatFromEnumToStr(Format::DEFAULT_FORMAT), dtype_id, shape,
                                device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                device_context->device_context_key().device_id_);
  tensor->set_stream_id(kDefaultStreamIndex);
  auto device_addr = tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, kDump, "OutputAddress", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, kDump, device::tracker::MemType::kOther,
                                                 device_addr->GetSize(), device_addr.get());
  if (!device_context->device_res_manager_->AllocateMemory(device_addr.get(), kDefaultStreamIndex)) {
    MS_LOG(EXCEPTION) << "Dump allocate outputs memory failed";
  } else {
    static std::string name = "Alloc memory";
    tensor->IncreaseNewRefCount(name);
  }
  return tensor;
}

KernelTensorPtr DumpMemManager::CreateWorkspaceKernelTensor(const DeviceContext *device_context,
                                                            const size_t &workspace_size, size_t stream_id) {
  MS_EXCEPTION_IF_NULL(device_context);

  auto kernel_tensor =
    AnfAlgo::CreateKernelTensor(nullptr, workspace_size, Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
                                device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(stream_id);

  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, kDump, kWorkspaceAddress, "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, kDump, device::tracker::MemType::kWorkSpace,
                                                 device_address->GetSize(), device_address.get());
  if (!device_context->device_res_manager_->AllocateMemory(device_address.get(), stream_id)) {
    MS_LOG(EXCEPTION) << "Allocate dynamic workspace memory failed";
  } else {
    static std::string name = "Alloc memory";
    kernel_tensor->IncreaseNewRefCount(name);
  }
  MS_LOG(DEBUG) << "Create workspace device address:" << device_address;
  return kernel_tensor;
}
}  // namespace datadump
}  // namespace mindspore
