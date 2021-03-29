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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DATA_SOURCE_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DATA_SOURCE_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <queue>
#include <utility>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/memory_interface_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/device_tensor_store.h"
#include "runtime/framework/host_tensor_queue.h"
#include "base/base.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;

// The data source actor is used to fetch data from data source and process them into device tensors,
// and then send them to kernel actor. The processing flow is FetchData -> FillDataBuffer -> AllocateMemory
// -> OnMemoryAllocFinish -> SendOutput -> FreeMemory.
class DataSourceActor : public MemoryInterfaceActor {
 public:
  DataSourceActor(std::string name, size_t buffer_capacity, const DeviceContext *device_context,
                  const AID memory_manager_aid)
      : MemoryInterfaceActor(name),
        buffer_capacity_(buffer_capacity),
        device_context_(device_context),
        memory_manager_aid_(memory_manager_aid) {}
  virtual ~DataSourceActor() = default;

  // The process entry of data processing.
  void FetchData(OpContext<DeviceTensor> *context);

  // The memory related operation interface.
  void AllocateMemory(OpContext<DeviceTensor> *context) override;
  void FreeMemory(OpContext<DeviceTensor> *context) override;
  // Copy data from data source to the device tensor buffer of actor after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *context) override{};

 protected:
  // Construct the device tensors and fill to device tensor buffer from the member nodes during the data fetching.
  virtual void FillDataBuffer() = 0;

  // Send output to downstream actors to trigger computing after fetching data finished.
  void SendOutput(OpContext<DeviceTensor> *context);

  // To trigger kernel actors running by op arrows.
  std::vector<OpArrowPtr> output_op_arrows_;

  // The buffers store the device tensors.
  std::queue<std::vector<DeviceTensor *>> buffers_;
  size_t buffer_capacity_;

  // The device interface of data copy.
  const DeviceContext *device_context_;

  // The id of memory manager actor. Send message to it for alloc and free memory during the data processing.
  const AID memory_manager_aid_;
};

// The class represents that the data source is device queue.
class DeviceQueueDataSourceActor : public DataSourceActor {
 public:
  DeviceQueueDataSourceActor(std::string name, size_t buffer_capacity, const DeviceContext *device_context,
                             const AID memory_manager_aid)
      : DataSourceActor(name, buffer_capacity, device_context, memory_manager_aid) {}
  ~DeviceQueueDataSourceActor() override = default;

  void OnMemoryAllocFinish(OpContext<DeviceTensor> *context) override;

 protected:
  void FillDataBuffer() override;

 private:
  friend class GraphScheduler;

  // Input data kernel(for example GetNext) fetches data from device queue.
  CNodePtr data_kernel_;
};

// The class represents that the data source is host queue.
class HostQueueDataSourceActor : public DataSourceActor {
 public:
  HostQueueDataSourceActor(std::string name, size_t buffer_capacity, const DeviceContext *device_context,
                           const AID memory_manager_aid, HostTensorQueuePtr host_queue)
      : DataSourceActor(name, buffer_capacity, device_context, memory_manager_aid), host_queue_(host_queue) {}
  ~HostQueueDataSourceActor() override = default;

  void OnMemoryAllocFinish(OpContext<DeviceTensor> *context) override;

 protected:
  void FillDataBuffer() override;

 private:
  friend class GraphScheduler;

  HostTensorQueuePtr host_queue_;
  // Input data nodes fetch data from host queue.
  std::vector<AnfNodePtr> data_nodes_;
};

using DataSourceActorPtr = std::shared_ptr<DataSourceActor>;
using DeviceQueueDSActorPtr = std::shared_ptr<DeviceQueueDataSourceActor>;
using HostQueueDSActorPtr = std::shared_ptr<HostQueueDataSourceActor>;

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DATA_SOURCE_ACTOR_H_
