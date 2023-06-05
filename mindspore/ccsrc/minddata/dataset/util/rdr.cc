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

std::string MDChannelInfo::ToFormatString() {
  std::ostringstream ss;
  {
    // the output like below:
    // preprocess_batch: 282;
    // batch_queue: 64, 64, 64, 63, 64, 64, 64, 63, 64, 64;
    //             push_start_time -> push_end_time
    //                             -> 2022-05-09-14:31:00.603.866
    // 2022-05-09-14:31:00.621.146 -> 2022-05-09-14:31:01.018.964
    // 2022-05-09-14:31:01.043.705 -> 2022-05-09-14:31:01.396.650
    // 2022-05-09-14:31:01.421.501 -> 2022-05-09-14:31:01.807.671
    // 2022-05-09-14:31:01.828.931 -> 2022-05-09-14:31:02.179.945
    // 2022-05-09-14:31:02.201.960 -> 2022-05-09-14:31:02.555.941
    // 2022-05-09-14:31:02.584.413 -> 2022-05-09-14:31:02.943.839
    // 2022-05-09-14:31:02.969.583 -> 2022-05-09-14:31:03.309.299
    // 2022-05-09-14:31:03.337.607 -> 2022-05-09-14:31:03.684.034
    // 2022-05-09-14:31:03.717.230 -> 2022-05-09-14:31:04.038.521
    // 2022-05-09-14:31:04.064.571 ->
    std::unique_lock<std::mutex> lock(mutex_);
    ss << "\n";
    ss << "preprocess_batch: " << preprocess_batch_ << ";\n";
    ss << "batch_queue: ";
    for (uint32_t i = 0; i < batch_queue_.size(); i++) {
      ss << batch_queue_.at(i);
      if (i < batch_queue_.size() - 1) {
        ss << ", ";
      }
    }
    ss << ";\n";
    ss << "            push_start_time -> push_end_time\n";
    if (!push_start_time_.empty()) {
      if (!push_end_time_.empty()) {
        // start_time[0] bigger than end_time[0]
        uint32_t end_time_index = 0;
        if (push_start_time_.at(0) > push_end_time_.at(0)) {
          ss << "                            -> " << push_end_time_.at(end_time_index) << "\n";
          end_time_index = 1;
        }
        for (uint32_t i = 0; i < push_start_time_.size(); i++, end_time_index++) {
          ss << push_start_time_.at(i) << " -> ";
          if (end_time_index < push_end_time_.size()) {
            ss << push_end_time_.at(end_time_index);
          }
          ss << "\n";
        }
      } else {
        ss << push_start_time_.at(0) << " -> \n";  // only one start time without end time
      }
    }
    ss << "For more details, please refer to the FAQ at "
       << "https://www.mindspore.cn/docs/en/master/faq/data_processing.html.";
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
