/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_MANAGER_H_
#include <vector>
#include <string>
#include <atomic>
#include <utility>
#include <map>
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/tensor_array.h"
#include "runtime/device/tensors_queue.h"

namespace mindspore {
namespace device {
class TensorArrayMgr {
 public:
  // TensorArrayMgr is used to manage the TensorArrays.
  TensorArrayMgr() {}
  ~TensorArrayMgr() = default;

  static TensorArrayMgr &GetInstance() noexcept {
    static TensorArrayMgr instance;
    return instance;
  }

  TensorArrayMgr(const TensorArrayMgr &) = delete;
  TensorArrayMgr(const TensorArrayMgr &&) = delete;
  TensorArrayMgr &operator=(const TensorArrayMgr &&) = delete;
  TensorArrayMgr &operator=(const TensorArrayMgr &) = delete;

  void AddTensorArray(const int64_t handle, const TensorArrayPtr &ta) {
    MS_LOG(DEBUG) << "Add a TensorArray to map, handle is " << handle;
    (void)tensors_map_.emplace(std::make_pair(handle, ta));
    // Increase handle count when added a TensorArray.
    tensor_array_handle_count += 1;
  }

  TensorArrayPtr GetTensorArray(const int64_t handle) {
    if (tensors_map_.count(handle) == 0) {
      MS_LOG(EXCEPTION) << "Error handle [" << handle << "] to get tensorarray";
    } else {
      MS_LOG(DEBUG) << "Get TensorArray succeed, handle is " << handle;
      return tensors_map_[handle];
    }
  }

  bool EraseTensorArray(const int64_t handle) {
    if (tensors_map_.count(handle) == 1) {
      MS_LOG(DEBUG) << "Erase tensorarray from map, handle number is " << handle;
      (void)tensors_map_.erase(handle);
      return true;
    } else {
      MS_LOG(ERROR) << "Erase failed, no such handle " << handle;
      return false;
    }
  }

  int64_t GetHandleCount() const { return tensor_array_handle_count; }

 private:
  // Store the TensorArrays in a map, as pair(handle, TensorArrayPtr).
  std::map<const int64_t, TensorArrayPtr> tensors_map_;
  // Used as an unique handle number for each TensorArray.
  std::atomic<int64_t> tensor_array_handle_count{0};
};

class TensorsQueueMgr {
 public:
  // TensorsQueueMgr is used to manage the TensorsQueues.
  TensorsQueueMgr() {}
  ~TensorsQueueMgr() = default;

  static TensorsQueueMgr &GetInstance() noexcept {
    static TensorsQueueMgr instance;
    return instance;
  }

  TensorsQueueMgr(const TensorsQueueMgr &) = delete;
  TensorsQueueMgr(const TensorsQueueMgr &&) = delete;
  TensorsQueueMgr &operator=(const TensorsQueueMgr &&) = delete;
  TensorsQueueMgr &operator=(const TensorsQueueMgr &) = delete;

  void AddTensorsQueue(const int64_t handle, const TensorsQueuePtr &tq) {
    MS_LOG(DEBUG) << "Add a TensorsQueue to map, handle is " << handle;
    (void)tensorsqueue_map_.emplace(std::make_pair(handle, tq));
    // Increase handle count when added a TensorsQueue.
    tensors_queue_handle_count += 1;
  }

  TensorsQueuePtr GetTensorsQueue(const int64_t handle) {
    if (tensorsqueue_map_.count(handle) == 0) {
      MS_LOG(EXCEPTION) << "Error handle [" << handle << "] to get TensorsQueue";
    } else {
      MS_LOG(DEBUG) << "Get TensorsQueue succeed, handle is " << handle;
      return tensorsqueue_map_[handle];
    }
  }

  bool EraseTensorsQueue(const int64_t handle) {
    if (tensorsqueue_map_.count(handle) == 1) {
      MS_LOG(DEBUG) << "Erase TensorsQueue from map, handle number is " << handle;
      (void)tensorsqueue_map_.erase(handle);
      return true;
    } else {
      MS_LOG(ERROR) << "Erase TensorsQueue failed, no such handle " << handle;
      return false;
    }
  }

  int64_t GetHandleCount() const { return tensors_queue_handle_count; }

 private:
  // Store the TensorsQueues in a map, as pair(handle, TensorsQueuePtr).
  std::map<const int64_t, TensorsQueuePtr> tensorsqueue_map_;
  // Used as an unique handle number for each TensorsQueue.
  std::atomic<int64_t> tensors_queue_handle_count{0};
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_MANAGER_H_
