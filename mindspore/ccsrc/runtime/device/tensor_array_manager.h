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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_MANAGER_H_

#include <vector>
#include <string>
#include <atomic>
#include <utility>
#include <map>
#include "backend/session/kernel_graph.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/tensor_array.h"

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

  void AddTensorArray(const int64_t handle, const TensorArrayPtr &ta) {
    MS_LOG(DEBUG) << "Add a TensorArray to map, handle is " << handle;
    tensors_map_.emplace(std::make_pair(handle, ta));
    // Increase handle count when added a TensorArray.
    tensor_array_handle_count += 1;
  }

  TensorArrayPtr GetTensorArray(const int64_t handle) {
    if (!tensors_map_.count(handle)) {
      MS_LOG(EXCEPTION) << "Error handle [" << handle << "] to get tensorarray";
    } else {
      MS_LOG(DEBUG) << "Get TensorArray succeed, handle is " << handle;
      return tensors_map_[handle];
    }
  }

  bool EraseTensorArray(const int64_t handle) {
    if (tensors_map_.count(handle)) {
      MS_LOG(DEBUG) << "Erase tensorarray from map, handle number is " << handle;
      tensors_map_.erase(handle);
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
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_TENSOR_ARRAY_MANAGER_H_
