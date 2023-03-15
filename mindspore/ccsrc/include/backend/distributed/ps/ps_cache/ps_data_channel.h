/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PS_PS_CACHE_PS_DATA_PS_DATA_CHANNEL_H_
#define MINDSPORE_CCSRC_PS_PS_CACHE_PS_DATA_PS_DATA_CHANNEL_H_

#include <memory>
#include <string>
#include <condition_variable>

namespace mindspore {
namespace ps {
class PsDataChannel {
 public:
  PsDataChannel(const std::string &channel_name, size_t step_num)
      : channel_name_(channel_name),
        step_num_(step_num),
        current_data_step_(0),
        current_graph_step_(0),
        channel_open_(false),
        data_(nullptr),
        data_size_(0) {}
  virtual ~PsDataChannel() = default;
  void set_data(const void *data, const size_t data_size);
  const void *data() const { return data_; }
  size_t data_size() const { return data_size_; }
  void ResetData() { data_ = nullptr; }
  void set_step_num(size_t step_num) { step_num_ = step_num; }
  void TryWakeChannel(bool force_wake = false);

 private:
  void TryLockChannel();
  std::string channel_name_;
  // The step num of each epoch.
  size_t step_num_;
  size_t current_data_step_;
  size_t current_graph_step_;
  bool channel_open_;
  std::mutex channel_mutex_;
  std::condition_variable channel_;
  void *data_;
  size_t data_size_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PS_CACHE_PS_DATA_PS_DATA_CHANNEL_H_
