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
#include "mindrt/include/actor/op_actor.h"
#include "mindrt/include/async/future.h"
#include "runtime/framework/device_tensor_store.h"
#include "runtime/framework/host_tensor_queue.h"
#include "base/base.h"

namespace mindspore {
namespace runtime {
// The data source actor is used to fetch data and process them into device tensors,
// and then send them to kernel actor.
class DataSourceActor : public ActorBase {
 public:
  DataSourceActor(std::string name, size_t buffer_capacity) : ActorBase(name), buffer_capacity_(buffer_capacity) {}
  virtual ~DataSourceActor() = default;

  // The process entry of data processing.
  virtual void FetchData(OpContext<DeviceTensor> *context) = 0;

 protected:
  // To trigger kernel actors running by op arrows.
  std::vector<OpArrowPtr> output_op_arrows_;

  // The buffers store the data.
  std::queue<std::vector<DeviceTensorPtr>> buffers_;
  size_t buffer_capacity_;

  // The sequential number of corresponding batch data.
  std::queue<uuids::uuid *> sequential_nums_;
};

// The class represents that the data source is device queue.
class DeviceQueueDataSourceActor : public DataSourceActor {
 public:
  DeviceQueueDataSourceActor(std::string name, size_t buffer_capacity) : DataSourceActor(name, buffer_capacity) {}
  virtual ~DeviceQueueDataSourceActor() = default;

  void FetchData(OpContext<DeviceTensor> *context) override;

 private:
  friend class GraphScheduler;

  // Input data kernel(for example GetNext) fetches data from device queue.
  CNodePtr data_kernel_;
};

// The class represents that the data source is host queue.
class HostQueueDataSourceActor : public DataSourceActor {
 public:
  HostQueueDataSourceActor(std::string name, size_t buffer_capacity, HostTensorQueuePtr host_queue)
      : DataSourceActor(name, buffer_capacity), host_queue_(host_queue) {}
  virtual ~HostQueueDataSourceActor() = default;

  void FetchData(OpContext<DeviceTensor> *context) override;

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
