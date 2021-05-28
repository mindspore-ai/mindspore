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

#include "utils/summary/event_writer.h"
#include <string>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace summary {

// implement the EventWriter
EventWriter::EventWriter(const std::string &file_full_name) : filename_(file_full_name), events_write_count_(0) {
  fs_ = system::Env::GetFileSystem();
  if (fs_ == nullptr) {
    MS_LOG(EXCEPTION) << "Get the file system failed.";
  }
  event_file_ = fs_->CreateWriteFile(filename_);
  if (event_file_ == nullptr) {
    MS_LOG(EXCEPTION) << "Create the event file(" << file_full_name << ") failed.";
  }
  // set the event writer status
  status_ = true;
}

EventWriter::~EventWriter() {
  if (event_file_ != nullptr) {
    bool result = Close();
    if (!result) {
      MS_LOG(ERROR) << "Close file(" << filename_ << ") failed.";
    }
  }
}

// get the write event count
int32_t EventWriter::GetWriteEventCount() const { return events_write_count_; }

// Open the file
bool EventWriter::Open() {
  if (event_file_ == nullptr) {
    MS_LOG(ERROR) << "Open the file(" << filename_ << ") failed.";
    return false;
  }
  bool result = event_file_->Open();
  if (!result) {
    MS_LOG(ERROR) << "Open the file(" << filename_ << ") failed.";
  }
  return result;
}

// write the event serialization string to file
void EventWriter::Write(const std::string &event_str) {
  if (event_file_ == nullptr) {
    MS_LOG(ERROR) << "Write failed because file could not be opened.";
    return;
  }
  events_write_count_++;
  bool result = WriteRecord(event_str);
  if (!result) {
    MS_LOG(ERROR) << "Event write failed.";
  }
}

bool EventWriter::Flush() {
  // Confirm the event file is exist?
  if (!fs_->FileExist(filename_)) {
    MS_LOG(ERROR) << "Failed to flush to file(" << filename_ << ") because the file not exist.";
    return false;
  }
  if (event_file_ == nullptr) {
    MS_LOG(ERROR) << "Can't flush because the event file is null.";
    return false;
  }
  // Sync the file
  if (!event_file_->Flush()) {
    MS_LOG(ERROR) << "Failed to sync to file(" << filename_ << "), the event count(" << events_write_count_ << ").";
    return false;
  }
  MS_LOG(DEBUG) << "Flush " << events_write_count_ << " events to disk file(" << filename_ << ").";
  return true;
}

bool EventWriter::Close() noexcept {
  MS_LOG(DEBUG) << "Close the event writer.";
  bool result = true;
  if (!status_) {
    MS_LOG(INFO) << "The event writer is closed.";
    return result;
  }
  if (event_file_ != nullptr) {
    result = event_file_->Close();
    if (!result) {
      MS_LOG(ERROR) << "Close the file(" << filename_ << ") failed.";
    }
  }
  return result;
}

bool EventWriter::Shut() noexcept {
  MS_LOG(DEBUG) << "ShutDown the event writer.";
  if (!status_) {
    MS_LOG(INFO) << "The event writer is closed.";
    return true;
  }
  bool result = Flush();
  if (!result) {
    MS_LOG(ERROR) << "Flush failed when close the file.";
  }
  if (event_file_ != nullptr) {
    bool _close = event_file_->Close();
    if (!_close) {
      MS_LOG(ERROR) << "Close the file(" << filename_ << ") failed.";
      result = _close;
    }
  }
  events_write_count_ = 0;
  status_ = false;
  return result;
}

bool EventWriter::WriteRecord(const std::string &data) {
  if (event_file_ == nullptr) {
    MS_LOG(ERROR) << "Writer not initialized or previously closed.";
    return false;
  }
  // Write the data to event file
  const unsigned int kArrayLen = sizeof(uint64_t);
  char data_len_array[kArrayLen];
  char crc_array[sizeof(uint32_t)];

  // step 1: write the data length
  system::EncodeFixed64(data_len_array, kArrayLen, static_cast<int64_t>(data.size()));
  bool result = event_file_->Write(string(data_len_array, sizeof(data_len_array)));
  if (!result) {
    MS_LOG(ERROR) << "Write the Summary data length failed.";
    return false;
  }

  // step 2: write the crc of data length
  system::EncodeFixed64(data_len_array, kArrayLen, SizeToInt(data.size()));
  uint32_t crc = system::Crc32c::GetMaskCrc32cValue(data_len_array, sizeof(data_len_array));
  system::EncodeFixed32(crc_array, crc);
  result = event_file_->Write(string(crc_array, sizeof(crc_array)));
  if (!result) {
    MS_LOG(ERROR) << "Write the Summary data length crc failed.";
    return false;
  }

  // step 3: write the data
  result = event_file_->Write(data);
  if (!result) {
    MS_LOG(ERROR) << "Write the Summary data failed.";
    return false;
  }

  // step 4: write data crc
  crc = system::Crc32c::GetMaskCrc32cValue(data.data(), data.size());
  system::EncodeFixed32(crc_array, crc);
  result = event_file_->Write(string(crc_array, sizeof(crc_array)));
  if (!result) {
    MS_LOG(ERROR) << "Write the Summary footer failed.";
    return false;
  }

  return true;
}

}  // namespace summary
}  // namespace mindspore
