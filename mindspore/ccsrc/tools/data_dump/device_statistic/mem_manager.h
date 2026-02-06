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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "tools/data_dump/tensor_info_collect.h"

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_MEM_MANAGER_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_MEM_MANAGER_H_
namespace mindspore {

namespace datadump {
using device::DeviceContext;

class BACKEND_COMMON_EXPORT DumpMemManager {
 public:
  static DumpMemManager &GetInstance() {
    static DumpMemManager instance;
    return instance;
  }

  ~DumpMemManager() = default;
  DumpMemManager(const DumpMemManager &) = delete;
  DumpMemManager &operator=(const DumpMemManager &) = delete;
  void ClearCache();
  void Reset();
  KernelTensorPtr GetOutputTensor(const DeviceContext *device_context, size_t stream_id, TypeId dtype_id);
  KernelTensorPtr GetWorkSpaceTensor(const DeviceContext *device_context, size_t stream_id, size_t size);

 private:
  void Initialize(const DeviceContext *device_context);
  KernelTensorPtr CreateOutPutKernelTensor(const DeviceContext *device_context, const TypeId &dtype_id);
  KernelTensorPtr CreateWorkspaceKernelTensor(const DeviceContext *device_context, const size_t &workspace_size,
                                              size_t stream_id);
  DumpMemManager() = default;

  std::unordered_map<size_t, std::vector<KernelTensorPtr>> output_cache_;
  std::unordered_map<size_t, KernelTensorPtr> workspace_cache_;
  std::unordered_map<size_t, size_t> output_index_;  // stream_id, index

  const size_t max_workspace_size_ = 128 * 1024;
  const size_t max_output_num_ = 128;

  std::once_flag init_flag_;
  std::mutex output_cache_mutex_;
  std::mutex workspace_cache_mutex_;
};
}  // namespace datadump
}  // namespace mindspore
#endif
