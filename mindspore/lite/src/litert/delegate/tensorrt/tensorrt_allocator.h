/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_TENSORRT_ALLOCATOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_TENSORRT_ALLOCATOR_H_
#include "src/litert/delegate/tensorrt/tensorrt_allocator.h"
#include <map>
#include <string>
#include <NvInfer.h>
#include "include/api/types.h"

namespace mindspore::lite {
struct CudaTensorParam {
  void *data = nullptr;
  bool is_valid_mem = false;
  size_t size = 0;
};
class TensorRTAllocator {
 public:
  TensorRTAllocator() = default;

  ~TensorRTAllocator() = default;

  void *MallocDeviceMem(const mindspore::MSTensor &host_tensor, size_t size);

  void *MallocDeviceMem(const std::string &name, size_t size, nvinfer1::DataType data_type);

  void *GetDevicePtr(const std::string &tensor_name);

  void SetCudaStream(cudaStream_t stream) { stream_ = stream; }

  std::map<std::string, CudaTensorParam> GetAllDevicePtr();

  int SyncMemInHostAndDevice(mindspore::MSTensor host_tensor, const std::string &device_tensor_name,
                             bool is_host2device, bool sync = true);

  int SyncMemInHostAndDevice(void *host_data, const std::string &device_tensor_name, size_t data_size,
                             bool is_host2device, bool sync = true);

  int ClearDeviceMem();

  void MarkMemValid(const std::string &name, bool isValid);

  bool GetMemIsValid(const std::string &name);

  void SetDeviceID(uint32_t device_id) { this->device_id_ = device_id; }

  uint32_t GetDeviceID() const { return this->device_id_; }

 private:
  std::map<std::string, CudaTensorParam> cuda_tensor_map_;
  cudaStream_t stream_;
  uint32_t device_id_{0};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_TENSORRT_ALLOCATOR_H_
