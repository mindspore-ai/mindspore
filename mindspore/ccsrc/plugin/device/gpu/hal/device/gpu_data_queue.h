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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DATA_QUEUE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DATA_QUEUE_H_

#include <cuda_runtime_api.h>
#include <unistd.h>
#include <memory>
#include <vector>
#include <functional>
#include "include/backend/data_queue/data_queue.h"
#include "runtime/hardware/device_context_manager.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
class BACKEND_EXPORT GpuDataQueueDynamic : public DataQueue {
 public:
  explicit GpuDataQueueDynamic(const size_t capacity);
  ~GpuDataQueueDynamic() override = default;

  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override;
  DataQueueStatus Pop() override;
  bool Destroy() override;

  void SetThreadDevice() override;

 private:
  struct NodeInfo {
    std::unique_ptr<cudaEvent_t> event_;
    std::vector<DataQueueItem> data_;
  };

  std::vector<size_t> shape_;

  cudaStream_t stream_;
  std::unique_ptr<NodeInfo[]> node_info_;
  uint32_t device_id_;
};

class BACKEND_EXPORT GpuQueue : public DataQueue {
 public:
  GpuQueue(size_t capacity, void *addr, const std::vector<size_t> &shape);
  ~GpuQueue() override;

  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override;
  DataQueueStatus Pop() override;
  bool Destroy() override;

  void SetThreadDevice() override;

 private:
  struct NodeInfo {
    std::unique_ptr<cudaEvent_t> event_;
    std::vector<DataQueueItem> data_;
  };

  void *buffer_;

  std::vector<size_t> shape_;
  size_t len_;
  cudaStream_t stream_;
  std::unique_ptr<NodeInfo[]> node_info_;
  bool ds_detected_{false};
  uint32_t device_id_;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_BLOCKING_QUEUE_H_
