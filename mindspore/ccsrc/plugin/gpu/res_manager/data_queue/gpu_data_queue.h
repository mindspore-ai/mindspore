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

#ifndef MINDSPORE_CCSRC_PLUGIN_GPU_RES_MANAGER_DATA_QUEUE_DATA_QUEUE_H_
#define MINDSPORE_CCSRC_PLUGIN_GPU_RES_MANAGER_DATA_QUEUE_DATA_QUEUE_H_

#include <cuda_runtime_api.h>
#include <memory>
#include <vector>
#include <functional>
#include <string>
#include "include/runtime/hardware_abstract/data_queue/data_queue.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"

namespace mindspore {
namespace device {
class GpuDataQueueDynamic : public DataQueue {
 public:
  explicit GpuDataQueueDynamic(const std::string &channel_name, const size_t capacity);
  ~GpuDataQueueDynamic() override = default;

  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override;
  DataQueueStatus Pop() override;

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

class GpuQueue : public DataQueue {
 public:
  GpuQueue(const std::string &channel_name, size_t capacity, const std::vector<size_t> &shape);
  ~GpuQueue() override;

  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override;
  DataQueueStatus Pop() override;

  void SetThreadDevice() override;

 private:
  struct NodeInfo {
    std::unique_ptr<cudaEvent_t> event_;
    std::vector<DataQueueItem> data_;
  };

  void *buffer_{nullptr};

  std::vector<size_t> shape_;
  size_t len_;
  cudaStream_t stream_;
  std::unique_ptr<NodeInfo[]> node_info_;
  uint32_t device_id_;
  bool ds_detected_{false};
};

class GpuDataQueue : public DataQueue {
 public:
  GpuDataQueue(const std::string &channel_name, size_t capacity, const std::vector<size_t> &shape);
  ~GpuDataQueue() override;

  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override;
  DataQueueStatus Pop() override;
  DataQueueStatus FrontAsync(std::vector<DataQueueItem> *data) const override;
  void SetThreadDevice() override;

 private:
  struct NodeInfo {
    std::unique_ptr<cudaEvent_t> event_;
    std::vector<DataQueueItem> data_;
    void *device_addr_{nullptr};
    size_t data_len_{0};
  };

  std::vector<size_t> shape_;
  size_t len_;
  cudaStream_t stream_;
  std::unique_ptr<NodeInfo[]> node_info_;
  uint32_t device_id_;
  bool ds_detected_{false};
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_GPU_RES_MANAGER_DATA_QUEUE_DATA_QUEUE_H_
