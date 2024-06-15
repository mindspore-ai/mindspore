/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_UTILS_H_

#include <pthread.h>

#include <atomic>
#include <memory>
#include <string>
#include <set>
#include <vector>

#include "include/backend/kernel_graph.h"
#include "mindspore/core/utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
template <typename Map, typename K = typename Map::key_type, typename V = typename Map::mapped_type>
std::string MapToString(const Map &value) {
  std::stringstream buffer;
  buffer << "{";
  for (auto it = value.begin(); it != value.end(); it++) {
    if (it != value.begin()) {
      buffer << ", ";
    }
    buffer << it->first << ": " << it->second;
  }
  buffer << "}";
  return buffer.str();
}

class ErrorManagerAdapter {
 public:
  ErrorManagerAdapter() = default;
  ~ErrorManagerAdapter() = default;
  static bool Init();
  static std::string GetErrorMessage(bool add_title = false);

 private:
  static void MessageHandler(std::ostringstream *oss);

 private:
  static std::mutex initialized_mutex_;
  static bool initialized_;
};

std::string GetErrorMsg(uint32_t rt_error_code);

void *callback_thread_func(void *data);

// Callback thread for ascend streams.
struct CallbackThread {
  ~CallbackThread() { cancel(); }

  // pthread_cancel may cause bug now, so just set flag to false.
  void cancel() {
    if (flag_.load()) {
      flag_.store(false);
    }
  }

  int create() {
    flag_.store(true);
    return pthread_create(&thread_, nullptr, &callback_thread_func, this);
  }

  pthread_t thread_;
  std::atomic_bool flag_{true};
  int32_t default_timeout_{500};
};
using CallbackThreadPtr = std::shared_ptr<CallbackThread>;

std::string EnableLcclEnv();

void InitializeAcl();

std::string GetFormatMode();
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_UTILS_H_
