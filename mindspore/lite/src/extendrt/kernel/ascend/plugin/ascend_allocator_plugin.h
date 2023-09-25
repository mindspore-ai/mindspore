/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ASCEND_ALLOCATOR_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_ASCEND_ALLOCATOR_PLUGIN_H_
#include <string>
#include <memory>
#include "include/api/status.h"
namespace mindspore::kernel {
class AscendAllocatorPluginImpl {
 public:
  AscendAllocatorPluginImpl() = default;
  virtual ~AscendAllocatorPluginImpl() = default;

  virtual int GetCurrentDeviceId() = 0;
  virtual void *Malloc(size_t size, int device_id = -1) = 0;
  virtual void Free(void *device_data, int device_id) = 0;
  virtual void *MallocHost(size_t size) = 0;
  virtual void FreeHost(void *host_data) = 0;
  virtual Status CopyDeviceDataToHost(void *device_data, void *host_data, size_t data_size, int device_id) = 0;
  virtual Status CopyHostDataToDevice(void *host_data, void *device_data, size_t data_size) = 0;
  virtual Status CopyDeviceDataToDevice(void *src_device, void *dst_device, size_t src_data_size, size_t dst_data_size,
                                        int src_device_id, int dst_device_id) = 0;
};

class MS_API AscendAllocatorPlugin {
 public:
  static AscendAllocatorPlugin &GetInstance();
  bool Register();

  int GetCurrentDeviceId();
  void *Malloc(size_t size, int device_id = -1);
  void Free(void *device_data, int device_id);
  void *MallocHost(size_t size);
  void FreeHost(void *host_data);
  Status CopyDeviceDataToHost(void *device_data, void *host_data, size_t data_size, int device_id);
  Status CopyHostDataToDevice(void *host_data, void *device_data, size_t data_size);
  Status CopyDeviceDataToDevice(void *src_device, void *dst_device, size_t src_data_size, size_t dst_data_size,
                                int src_device_id, int dst_device_id);

 private:
  AscendAllocatorPlugin();
  ~AscendAllocatorPlugin();

  std::string plugin_path_;
  void *handle_ = nullptr;
  bool is_registered_ = false;
  std::shared_ptr<AscendAllocatorPluginImpl> ascend_allocator_plugin_impl_ = nullptr;
};
}  // namespace mindspore::kernel
#endif
