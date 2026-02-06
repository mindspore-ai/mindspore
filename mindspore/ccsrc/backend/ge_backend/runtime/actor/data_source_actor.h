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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DATA_SOURCE_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DATA_SOURCE_ACTOR_H_

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <queue>
#include <utility>

#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/runtime/actor/debug_aware_actor.h"
#include "backend/ge_backend/runtime/device_tensor_store.h"
#include "backend/ge_backend/runtime/host_tensor_queue.h"
#include "base/base.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using mindspore::device::KernelInfo;

// The data source actor is used to fetch data from data source and process them into device tensors,
// and then send them to kernel actor. The processing flow is FetchData -> FillDataBuffer -> SendMemoryAllocReq
// -> OnMemoryAllocFinish -> SendMemoryFreeReq -> SendOutput.
class DataSourceActor : public DebugAwareActor {
 public:
  DataSourceActor(const std::string &name, KernelTransformType type, size_t buffer_capacity,
                  const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid)
      : DebugAwareActor(name, type, recorder_aid, memory_manager_aid, debug_aid, nullptr),
        buffer_capacity_(buffer_capacity) {}
  ~DataSourceActor() override = default;

  virtual void ReleaseData() {}

 protected:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  friend class AnyTypeGraphScheduler;

  void Init() override;

  void Run(OpContext<KernelTensor> *const context) override { FetchData(context); }

  // The process entry of data processing.
  void FetchData(OpContext<KernelTensor> *const context);

  // Construct the device tensors and fill to device tensor buffer from the member nodes during the data fetching.
  virtual void FillDataBuffer() = 0;

  void UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &data_arrow,
                        const AnfNodePtr &output_node, OpContext<KernelTensor> *const context) override;

  // The buffers store the device tensors.
  std::queue<std::vector<KernelTensorPtr>> buffers_;
  size_t buffer_capacity_;
};

// The class represents that the data source is host queue.
class HostQueueDataSourceActor : public DataSourceActor {
 public:
  HostQueueDataSourceActor(const std::string &name, size_t buffer_capacity, const AID &memory_manager_aid,
                           const AID *debug_aid, const AID *recorder_aid, const HostTensorQueuePtr &host_queue,
                           const std::string &graph_phase)
      : DataSourceActor(name, KernelTransformType::kHostDataSourceActor, buffer_capacity, memory_manager_aid, debug_aid,
                        recorder_aid),
        host_queue_(host_queue),
        is_infer_phase_(IsInferPhase(graph_phase)) {}
  ~HostQueueDataSourceActor() override = default;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<KernelTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<KernelTensor> *const context) override;
  // Copy data from data source to the device tensor buffer of actor after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<KernelTensor> *const context) override;

  size_t FetchNodePosition(const KernelWithIndex &node) const override;
  KernelWithIndex FetchNode(size_t node_position) const;
  const std::vector<KernelWithIndex> &data_nodes() const { return data_node_with_indexs_; }

  void ReleaseData() override;

 protected:
  void FillDataBuffer() override;

  void AddCopyDataCallBack(bool enable_async_copy, const mindspore::tensor::TensorPtrList &host_tensors,
                           const std::vector<KernelTensorPtr> &kernel_tensors);

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;

  // Judge all the data_nodes_ is from the same device.
  bool IsSameDeviceType() const;

  HostTensorQueuePtr host_queue_;
  // Input data nodes fetch data from host queue.
  std::vector<KernelWithIndex> data_node_with_indexs_;
  // The location of the data node in the data source actor.
  std::map<KernelWithIndex, size_t> data_node_position_map_;
  // The ref relationship of device address.
  std::map<KernelWithIndex, std::vector<KernelTensorPtr>> ref_kernel_tensors_;

  // Whether the super kernel actor is a infer 'prefill' or 'increment' graph or not.
  bool is_infer_phase_;
};

using DataSourceActorPtr = std::shared_ptr<DataSourceActor>;
using HostQueueDSActorPtr = std::shared_ptr<HostQueueDataSourceActor>;
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DATA_SOURCE_ACTOR_H_
