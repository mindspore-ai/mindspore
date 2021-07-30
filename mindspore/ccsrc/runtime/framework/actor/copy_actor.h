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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_COPY_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_COPY_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include "runtime/framework/actor/actor_common.h"
#include "runtime/framework/actor/memory_aware_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/framework/device_tensor_store.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;

// The copy actor is used to receive the device tensors and control info to copy data between input device tensor and
// output device tensor. The processing flow is RunOpData/RunOpControl -> CheckCopyCondition -> SendMemoryAllocReq
// -> OnMemoryAllocFinish -> Copy -> SendMemoryFreeReq -> SendOutput.
class CopyActor : public MemoryAwareActor {
 public:
  CopyActor(const std::string &name, const AID &memory_manager_aid)
      : MemoryAwareActor(name),
        memory_manager_aid_(memory_manager_aid),
        input_datas_num_(0),
        input_controls_num_(0),
        input_device_context_(nullptr),
        output_device_context_(nullptr),
        output_(nullptr) {}
  ~CopyActor() override = default;

  void Init() override;

  // The copy actor run when receive the input data.
  void RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) override;
  // The copy actor run when receive the input control.
  void RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) override;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;
  // The copy processing after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

 private:
  friend class GraphScheduler;

  // Check whether satisfy the condition for copy.
  bool CheckCopyCondition(OpContext<DeviceTensor> *const context) const;
  // Fetch the device tensor for copy.
  void FetchDeviceTensor(OpContext<DeviceTensor> *const context);

  // Send output data and output controls when finish copy.
  void SendOutput(OpContext<DeviceTensor> *const context) const;
  // Erase input data and input controls when finish copy.
  void EraseInput(OpContext<DeviceTensor> *const context);

  // The id of memory manager actor. Send message to it for alloc and free memory during the copy.
  const AID memory_manager_aid_;

  // The dependent input data number.
  size_t input_datas_num_;
  // The dependent input controls number.
  size_t input_controls_num_;

  // Pair<index, anfNode> points to the dependent device tensor store, anfNode is the key of the device tensor store.
  std::pair<size_t, AnfNode *> device_tensor_store_key_;

  // The device interface for copy.
  const DeviceContext *input_device_context_;
  const DeviceContext *output_device_context_;

  // The input device tensor is saved from the input data or fetched by device_tensor_store_key_.
  std::vector<DeviceTensor *> input_device_tensor_;
  // The output device tensor is saved from the output or fetched by device_tensor_store_key_.
  std::vector<DeviceTensor *> output_device_tensor_;

  //  The output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<OpDataUniquePtr<DeviceTensor>> output_data_;

  // The output is created in the copy actor build, so can't be the raw pointer.
  DeviceTensorPtr output_;
};

using CopyActorPtr = std::shared_ptr<CopyActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_COPY_ACTOR_H_
