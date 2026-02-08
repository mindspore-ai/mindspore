/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_COMM_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_COMM_UTILS_H_

#include <tuple>
#include <unordered_map>
#include <utility>
#include "runtime/hardware_abstract/event/device_event.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
struct PairHash {
  size_t operator()(const std::pair<size_t, size_t> &p) const {
    auto h1 = std::hash<size_t>{}(p.first);
    auto h2 = std::hash<size_t>{}(p.second);
    return h1 ^ h2;
  }
};

class PYBOOST_API CommUtils {
 public:
  static CommUtils &GetInstance() {
    static CommUtils instance;
    return instance;
  }

  void SyncOpStream(const device::DeviceContext *device_ctx, size_t op_stream_id, size_t comm_stream_id);

 private:
  CommUtils() = default;
  ~CommUtils() = default;

  DeviceEventPtr CreateOrGetCommBeginEvent(const device::DeviceContext *device_ctx, size_t op_stream_id,
                                           size_t comm_stream_id);
  std::unordered_map<std::pair<size_t, size_t>, DeviceEventPtr, PairHash> comm_begin_events_;
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_COMM_UTILS_H_
