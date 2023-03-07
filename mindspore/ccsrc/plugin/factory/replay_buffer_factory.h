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

#ifndef MINDSPORE_CCSRC_PLUGIN_FACTORY_REPLAY_BUFFER_FACTORY_H_
#define MINDSPORE_CCSRC_PLUGIN_FACTORY_REPLAY_BUFFER_FACTORY_H_

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <utility>
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kInvalidHandle = -1;
// Class for replay buffer managerment
// It holds instance which could be accessed across different kernels with
// factory pattern and singleton design pattern.
// The class is templated with specific replay buffer for example FIFO, Priority, EposidicFIFO, ...
// The Create() method support variable parameters.
// Here is an example:
// ```C++
// Define an custom replay buffer
// class CustomReplayBuffer {
//  public:
//   CustomReplayBuffer(params);
// };
// auto factory = ReplayBufferFactory<CustomReplayBuffer>::GetInstance();
// auto handle_instance_pair = factor.Create(params);
// ```
template <typename T>
class ReplayBufferFactory {
 public:
  // Create a factory instance with lazy mode.
  static ReplayBufferFactory &GetInstance() {
    static ReplayBufferFactory factory;
    return factory;
  }

  // Create an replay buffer instance with unique handle and instance returned.
  template <typename... _Args>
  std::tuple<int, std::shared_ptr<T>> Create(_Args... args) {
    auto instance = std::make_shared<T>(args...);
    (void)map_handle_to_instances_.insert(std::make_pair(++handle_, instance));
    return std::make_tuple(handle_, instance);
  }

  // Delete the replay buffer instance.
  void Delete(int64_t handle) { (void)map_handle_to_instances_.erase(handle); }

  // Get replay buffer instance by handle.
  std::shared_ptr<T> GetByHandle(int64_t handle) {
    auto iter = map_handle_to_instances_.find(handle);
    if (iter == map_handle_to_instances_.end()) {
      MS_LOG(EXCEPTION) << "Replay buffer with handle " << handle << " not exist.";
    }

    return iter->second;
  }

 private:
  ReplayBufferFactory() = default;
  ~ReplayBufferFactory() = default;
  DISABLE_COPY_AND_ASSIGN(ReplayBufferFactory)

  int64_t handle_ = kInvalidHandle;
  std::map<int64_t, std::shared_ptr<T>> map_handle_to_instances_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_FACTORY_REPLAY_BUFFER_FACTORY_H_
