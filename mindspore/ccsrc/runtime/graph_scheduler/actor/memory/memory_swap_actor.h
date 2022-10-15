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

#ifndef MINDSPORE_SWAP_OUT_ACTOR_H
#define MINDSPORE_SWAP_OUT_ACTOR_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "runtime/graph_scheduler/actor/abstract_actor.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/device/auto_mem_offload.h"

namespace mindspore {
namespace runtime {
class MemorySwapActor : public AbstractActor {
 public:
  MemorySwapActor(const std::string &name, const AID *recorder_aid, size_t stream_id,
                  std::vector<DeviceTensor *> device_tensors_to_swap)
      : AbstractActor(name, KernelTransformType::kMemorySwapActor, recorder_aid),
        stream_id_(stream_id),
        device_tensors_to_swap_(std::move(device_tensors_to_swap)) {}

 protected:
  void FetchRealParameters(OpContext<DeviceTensor> *context);

 protected:
  size_t stream_id_;
  std::vector<DeviceTensor *> device_tensors_to_swap_;
  std::vector<DeviceTensor *> real_parameters_;
};

class MemorySwapInActor : public MemorySwapActor {
 public:
  MemorySwapInActor(const std::string &name, const AID *recorder_aid, size_t stream_id,
                    const DeviceContext *device_context, const std::vector<DeviceTensor *> &device_tensors_to_swap,
                    std::vector<std::vector<DeviceTensor *>> continuous_device_tensors,
                    std::vector<std::vector<size_t>> continuous_device_tensor_sizes, size_t real_parameter_size)
      : MemorySwapActor(name, recorder_aid, stream_id, device_tensors_to_swap),
        continuous_device_tensors_(std::move(continuous_device_tensors)),
        continuous_device_tensor_sizes_(std::move(continuous_device_tensor_sizes)) {
    device_contexts_.emplace_back(device_context);
    real_parameters_.resize(real_parameter_size);
  }

 protected:
  void Run(OpContext<DeviceTensor> *context) override;

 private:
  std::vector<std::vector<DeviceTensor *>> continuous_device_tensors_;
  std::vector<std::vector<size_t>> continuous_device_tensor_sizes_;
};

class MemorySwapOutActor : public MemorySwapActor {
 public:
  MemorySwapOutActor(const std::string &name, const AID *recorder_aid, size_t stream_id,
                     const std::vector<DeviceTensor *> &device_tensors_to_swap,
                     const std::vector<DeviceTensor *> &device_tensors_to_free,
                     const std::vector<bool> &swap_out_real_parameter)
      : MemorySwapActor(name, recorder_aid, stream_id, device_tensors_to_swap),
        device_tensors_to_free_(device_tensors_to_free),
        swap_out_real_parameter_(swap_out_real_parameter) {
    real_parameters_.resize(swap_out_real_parameter.size());
  }

 protected:
  void Run(OpContext<DeviceTensor> *context) override;

 private:
  std::vector<DeviceTensor *> device_tensors_to_free_;
  // Whether offload real parameter when it does not have max original_ref_count_.
  std::vector<bool> swap_out_real_parameter_;
};

using MemSwapActorPtr = std::shared_ptr<MemorySwapActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_SWAP_OUT_ACTOR_H
