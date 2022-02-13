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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_HOST_QUEUE_STORE_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_HOST_QUEUE_STORE_H_

#include <memory>
#include <vector>
#include <queue>
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using mindspore::tensor::TensorPtr;

// Host tensor queue is used to store host tensors(such as non weighted parameters of graph), and its data will be
// fetched by the host queue data source actor.
class HostTensorQueue {
 public:
  HostTensorQueue() = default;
  virtual ~HostTensorQueue() = default;

  void Push(const std::vector<TensorPtr> &tensors) { buffers_.push(tensors); }

  const std::vector<TensorPtr> &Pull() { return buffers_.front(); }

  bool IsEmpty() const { return buffers_.empty(); }

  void Pop() { buffers_.pop(); }

 private:
  std::queue<std::vector<TensorPtr>> buffers_;
};

using HostTensorQueuePtr = std::shared_ptr<HostTensorQueue>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_HOST_QUEUE_STORE_H_
