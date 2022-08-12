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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DATA_QUEUE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DATA_QUEUE_H_

#include <unistd.h>
#include <memory>
#include <vector>
#include <functional>
#include "runtime/hardware/device_context_manager.h"
#include "runtime/data_queue/data_queue.h"
#include "runtime/rt.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
class BACKEND_EXPORT AscendDataQueueDynamic : public DataQueue {
 public:
  explicit AscendDataQueueDynamic(const size_t capacity);
  virtual ~AscendDataQueueDynamic() = default;

  BlockQueueStatus_T Push(std::vector<DataQueueItem> data);
  BlockQueueStatus_T Front(std::vector<DataQueueItem> *data) const;
  BlockQueueStatus_T Pop();
  bool Destroy();

 private:
  struct NodeInfo {
    std::vector<DataQueueItem> data_;
  };
  rtStream_t stream_;
  std::unique_ptr<NodeInfo[]> node_info_;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_BLOCKING_QUEUE_H_
