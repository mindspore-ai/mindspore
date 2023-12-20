/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORDUMP_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORDUMP_UTILS_H_

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "acl/acl_tdt.h"
#include "ir/tensor.h"
#include "plugin/device/ascend/hal/device/tensorprint_utils.h"
#include "plugin/device/ascend/hal/device/mbuf_receive_manager.h"

namespace mindspore::device::ascend {
class AsyncFileWriter {
 public:
  explicit AsyncFileWriter(size_t thread_nums);
  ~AsyncFileWriter();
  void Submit(std::function<void()> func);

 private:
  void WorkerThread();

  std::vector<std::thread> threads;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable cv;
  std::atomic_bool stop = false;
  std::atomic_bool threads_started = false;
};

class TensorDumpUtils {
 public:
  static TensorDumpUtils &GetInstance();

  TensorDumpUtils() = default;
  TensorDumpUtils(const TensorDumpUtils &) = delete;
  TensorDumpUtils &operator=(const TensorDumpUtils &) = delete;
  void AsyncSaveDatasetToNpyFile(acltdtDataset *acl_dataset);

 private:
  std::string TensorNameToArrayName(const std::string &tensor_name);
  AsyncFileWriter file_writer{4};
};

}  // namespace mindspore::device::ascend
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORDUMP_UTILS_H_
