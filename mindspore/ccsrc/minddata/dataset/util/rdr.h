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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_RDR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_RDR_H_

#include <deque>
#include <mutex>
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class MDChannelInfo {
 public:
  explicit MDChannelInfo(std::string channel_name) : channel_name_(channel_name), preprocess_batch_(0) {}

  ~MDChannelInfo() = default;

  std::string ToString();

  Status RecordBatchQueue(int64_t batch_queue_size);

  Status RecordPreprocessBatch(int64_t preprocess_batch);

  Status RecordPushStartTime();

  Status RecordPushEndTime();

 private:
  std::string channel_name_;
  std::deque<int64_t> batch_queue_;
  int64_t preprocess_batch_;
  std::deque<std::string> push_start_time_;
  std::deque<std::string> push_end_time_;
  std::mutex mutex_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_RDR_H_
