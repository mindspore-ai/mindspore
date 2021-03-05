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
#include "minddata/dataset/util/rdr.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
const int32_t kMdRdrRecordLimit = 10;

std::string MDChannelInfo::ToString() {
  std::ostringstream ss;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ss << "preprocess_batch: " << preprocess_batch_ << "; ";
    ss << "batch_queue: ";
    for (uint32_t i = 0; i < batch_queue_.size(); i++) {
      ss << batch_queue_.at(i);
      if (i < batch_queue_.size() - 1) {
        ss << ", ";
      }
    }

    ss << "; push_start_time: ";
    for (uint32_t i = 0; i < push_start_time_.size(); i++) {
      ss << push_start_time_.at(i);
      if (i < push_start_time_.size() - 1) {
        ss << ", ";
      }
    }

    ss << "; push_end_time: ";
    for (uint32_t i = 0; i < push_end_time_.size(); i++) {
      ss << push_end_time_.at(i);
      if (i < push_end_time_.size() - 1) {
        ss << ", ";
      }
    }
    ss << ".";
  }
  return ss.str();
}

Status MDChannelInfo::RecordBatchQueue(int64_t batch_queue_size) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (batch_queue_.size() == kMdRdrRecordLimit) {
      batch_queue_.pop_front();
    }
    batch_queue_.push_back(batch_queue_size);
  }
  return Status::OK();
}

Status MDChannelInfo::RecordPreprocessBatch(int64_t preprocess_batch) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    preprocess_batch_ = preprocess_batch;
  }
  return Status::OK();
}

Status MDChannelInfo::RecordPushStartTime() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (push_start_time_.size() == kMdRdrRecordLimit) {
      push_start_time_.pop_front();
    }
    push_start_time_.push_back(GetTimeString());
  }
  return Status::OK();
}

Status MDChannelInfo::RecordPushEndTime() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (push_end_time_.size() == kMdRdrRecordLimit) {
      push_end_time_.pop_front();
    }
    push_end_time_.push_back(GetTimeString());
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
