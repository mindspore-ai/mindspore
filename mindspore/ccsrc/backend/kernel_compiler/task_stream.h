/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TASK_STREAM_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TASK_STREAM_H_

#include <new>
#include <unordered_map>
#include <vector>
#include <memory>
#include "runtime/base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
class TaskStream {
 public:
  TaskStream() = default;
  ~TaskStream() = default;
  static std::shared_ptr<TaskStream> GetInstance() {
    static const std::shared_ptr<TaskStream> instance = std::make_shared<TaskStream>();
    return instance;
  }

  void set_gen_stream_list(const std::vector<rtStream_t> &stream_list) { gen_stream_list_ = stream_list; }
  void set_run_stream_list(const std::vector<rtStream_t> &stream_list) { run_stream_list_ = stream_list; }
  void SetGenStreamIndex(uint32_t stream_id, uint32_t index) { gen_stream_index_map_[stream_id] = index; }
  std::unordered_map<uint32_t, uint32_t> GetGenStreamIndexMap() { return gen_stream_index_map_; }
  uint32_t GetGenStreamIndex(uint32_t stream_id) {
    auto iter = gen_stream_index_map_.find(stream_id);
    if (iter == gen_stream_index_map_.end()) {
      MS_LOG(EXCEPTION) << "Parameter stream_id not in gen_stream_index_map_, id: " << stream_id;
    }
    return iter->second;
  }
  const std::vector<rtStream_t> &gen_stream_list() const { return gen_stream_list_; }
  const std::vector<rtStream_t> &run_stream_list() const { return run_stream_list_; }

 private:
  std::vector<rtStream_t> gen_stream_list_;
  std::vector<rtStream_t> run_stream_list_;
  std::unordered_map<uint32_t, uint32_t> gen_stream_index_map_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_TASK_STREAM_H_
