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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_DATA_QUEUE_HANDLER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_DATA_QUEUE_HANDLER_H_

#include <string>
#include <utility>
#include <functional>
#include <vector>
#include "utils/macros.h"
#include "utils/callback_handler.h"

namespace mindspore {
namespace device {
enum BlockQueueStatus_T : int { SUCCESS = 0, QUEUE_EXIST, QUEUE_NOT_EXIST, ERROR_INPUT, INTERNAL_ERROR, TIMEOUT };
struct DataQueueItem {
  int32_t worker_id_{0};
  std::string data_type_;
  size_t data_len_{0};
  void *data_ptr_{nullptr};
  std::vector<int64_t> shapes_;
  void *device_addr_{nullptr};
};
}  // namespace device
class MS_EXPORT DataQueueHandler {
  HANDLER_DEFINE(device::BlockQueueStatus_T, OpenDynamicBufQueue, const std::string &,
                 const std::function<void(void *, int32_t)>);
  HANDLER_DEFINE(device::BlockQueueStatus_T, Open, const std::string &, const std::function<void(void *, int32_t)>);
  HANDLER_DEFINE(bool, IsClosed);
  HANDLER_DEFINE(void, CloseConfirm);
  HANDLER_DEFINE(void, Close, const std::string &);
  HANDLER_DEFINE(device::BlockQueueStatus_T, Clear, const std::string &);
  HANDLER_DEFINE(device::BlockQueueStatus_T, Push, const std::string &, const std::vector<device::DataQueueItem> &,
                 unsigned int);
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_DATA_QUEUE_HANDLER_H_
