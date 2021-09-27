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
#include "runtime/framework/actor/debug_aware_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/device_tensor_store.h"
#include "runtime/framework/host_tensor_queue.h"
#include "base/base.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::device::KernelInfo;
using mindspore::kernel::KernelLaunchInfo;

// The data source actor is used to fetch data from data source and process them into device tensors,
// and then send them to kernel actor. The processing flow is FetchData -> FillDataBuffer -> SendMemoryAllocReq
// -> OnMemoryAllocFinish -> SendMemoryFreeReq -> SendOutput.
class DataSourceActor : public DebugAwareActor {
 public:
  DataSourceActor(const std::string &name, KernelTransformType type, size_t buffer_capacity,
                  const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid)
      : DebugAwareActor(name, type, recorder_aid, memory_manager_aid, debug_aid), buffer_capacity_(buffer_capacity) {}
  virtual ~DataSourceActor() = default;

  void Init() override;

  // The process entry of data processing.
  void FetchData(OpContext<DeviceTensor> *const context);

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override{};
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override{};
  // Copy data from data source to the device tensor buffer of actor after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override{};

 protected:
  friend class GraphScheduler;

  // Construct the device tensors and fill to device tensor buffer from the member nodes during the data fetching.
  virtual void FillDataBuffer() = 0;

  // Send output result of graph output to output actor.
  virtual void SendResult(OpContext<DeviceTensor> *const context) = 0;

  // Send recorder info to recorder actor, only the device queue data source actor need.
  virtual void SendRecorderInfo(OpContext<DeviceTensor> *const context) {}

  // Send output to downstream actors to trigger computing after fetching data finished.
  void SendOutput(OpContext<DeviceTensor> *const context);

  // The buffers store the device tensors.
  std::queue<std::vector<DeviceTensor *>> buffers_;
  size_t buffer_capacity_;

  //  The output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<OpDataUniquePtr<DeviceTensor>> output_data_;
};

// The class represents that the data source is device queue.
class DeviceQueueDataSourceActor : public DataSourceActor {
 public:
  DeviceQueueDataSourceActor(const std::string &name, size_t buffer_capacity, const DeviceContext *device_context,
                             const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid)
      : DataSourceActor(name, KernelTransformType::kDeviceDataSourceActor, buffer_capacity, memory_manager_aid,
                        debug_aid, recorder_aid) {
    (void)device_contexts_.emplace_back(device_context);
  }
  ~DeviceQueueDataSourceActor() override = default;

  void Init() override;

  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  void SendDebugReq(OpContext<DeviceTensor> *const context) override;
  void OnDebugFinish(OpContext<DeviceTensor> *const context) override;

 protected:
  void FillDataBuffer() override;
  void SendResult(OpContext<DeviceTensor> *const context) override;
  void SendRecorderInfo(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  // Input data kernel(for example GetNext) fetches data from device queue.
  CNodePtr data_kernel_{nullptr};
  KernelInfo *kernel_info_{nullptr};

  // The kernel launch info is fetched by the device tensors.
  KernelLaunchInfo launch_info_;
};

// The class represents that the data source is host queue.
class HostQueueDataSourceActor : public DataSourceActor {
 public:
  HostQueueDataSourceActor(std::string name, size_t buffer_capacity, const AID memory_manager_aid, const AID *debug_aid,
                           const AID *recorder_aid, HostTensorQueuePtr host_queue)
      : DataSourceActor(name, KernelTransformType::kHostDataSourceActor, buffer_capacity, memory_manager_aid, debug_aid,
                        recorder_aid),
        host_queue_(host_queue) {}
  ~HostQueueDataSourceActor() override = default;

  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  size_t FetchNodePosition(const AnfNodePtr &node) const override;
  AnfNodePtr FetchNode(size_t node_position) const;
  const std::vector<AnfNodePtr> &data_nodes() const { return data_nodes_; }

 protected:
  void FillDataBuffer() override;
  void SendResult(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  // Judge all the data_nodes_ is from the same device.
  bool IsSameDeviceType() const;

  HostTensorQueuePtr host_queue_;
  // Input data nodes fetch data from host queue.
  std::vector<AnfNodePtr> data_nodes_;

  // The location of the data node in the data source actor.
  std::unordered_map<AnfNodePtr, size_t> data_node_position_map_;
};

using DataSourceActorPtr = std::shared_ptr<DataSourceActor>;
using DeviceQueueDSActorPtr = std::shared_ptr<DeviceQueueDataSourceActor>;
using HostQueueDSActorPtr = std::shared_ptr<HostQueueDataSourceActor>;

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DATA_SOURCE_ACTOR_H_
