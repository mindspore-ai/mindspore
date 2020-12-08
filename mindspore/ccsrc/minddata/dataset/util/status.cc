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
#include "minddata/dataset/util/status.h"
#include <sstream>
#include <string>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils/ms_utils.h"
#include "./securec.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/util/task_manager.h"
#else
#include "minddata/dataset/util/log_adapter.h"
#endif

namespace mindspore {
namespace dataset {
std::string CodeAsString(const StatusCode c) {
  const char *s = nullptr;
  if (c == StatusCode::kOK) {
    // Optimize the most frequent case
    return std::string("OK");
  } else {
    switch (c) {
      case StatusCode::kOutOfMemory:
        s = "Out of memory";
        break;
      case StatusCode::kInterrupted:
        s = "Interrupted system call";
        break;
      case StatusCode::kShapeMisMatch:
        s = "Shape is incorrect.";
        break;
      case StatusCode::kNoSpace:
        s = "No space left on device";
        break;
      case StatusCode::kPyFuncException:
        s = "Exception thrown from PyFunc";
        break;
      case StatusCode::kDuplicateKey:
        s = "Duplicate key";
        break;
      case StatusCode::kProfilingError:
        s = "Error encountered while profiling";
        break;
      case StatusCode::kSyntaxError:
        s = "Syntax error";
        break;
      case StatusCode::kBuddySpaceFull:
        s = "BuddySpace full";
        break;
      case StatusCode::kNetWorkError:
        s = "Network error";
        break;
      case StatusCode::kUnexpectedError:
      default:
        s = "Unexpected error";
        break;
    }
  }
  return std::string(s);
}

Status::Status(StatusCode c) noexcept : code_(c), err_msg_(CodeAsString(c)) {}

Status::Status() noexcept : code_(StatusCode::kOK), err_msg_("") {}

Status::~Status() noexcept {}

Status::Status(const Status &s) : code_(s.code_), err_msg_(s.err_msg_) {}

Status &Status::operator=(const Status &s) {
  if (this == &s) {
    return *this;
  }
  code_ = s.code_;
  err_msg_ = s.err_msg_;
  return *this;
}

Status::Status(Status &&s) noexcept {
  code_ = s.code_;
  s.code_ = StatusCode::kOK;
  err_msg_ = std::move(s.err_msg_);
}

Status &Status::operator=(Status &&s) noexcept {
  if (this == &s) {
    return *this;
  }
  code_ = s.code_;
  s.code_ = StatusCode::kOK;
  err_msg_ = std::move(s.err_msg_);
  return *this;
}

Status::Status(const StatusCode code, const std::string &msg) : code_(code), err_msg_(msg) {}

Status::Status(const StatusCode code, int line_of_code, const char *file_name, const std::string &extra) {
  code_ = code;
  std::ostringstream ss;
#ifndef ENABLE_ANDROID
  ss << "Thread ID " << this_thread::get_id() << " " << CodeAsString(code) << ". ";
  if (!extra.empty()) {
    ss << extra;
  }
  ss << "\n";
#endif

  ss << "Line of code : " << line_of_code << "\n";
  if (file_name != nullptr) {
    ss << "File         : " << file_name << "\n";
  }
  err_msg_ = ss.str();
  if (code == StatusCode::kUnexpectedError) {
    MS_LOG(ERROR) << err_msg_;
  } else if (code == StatusCode::kNetWorkError) {
    MS_LOG(WARNING) << err_msg_;
  } else {
    MS_LOG(INFO) << err_msg_;
  }
}

std::ostream &operator<<(std::ostream &os, const Status &s) {
  os << s.ToString();
  return os;
}

std::string Status::ToString() const { return err_msg_; }

StatusCode Status::get_code() const { return code_; }

#if !defined(_WIN32) && !defined(_WIN64)
float GetMemoryUsage() {
  char buf[128] = {0};

  FILE *fd;
  fd = fopen("/proc/meminfo", "r");
  if (fd == nullptr) {
    MS_LOG(WARNING) << "The meminfo file: /proc/meminfo is opened failed.";
    return 0.0;
  }

  uint32_t status_count = 0;
  uint64_t mem_total = 0L;
  uint64_t mem_available = 0L;
  while (fgets(buf, sizeof(buf), fd)) {
    if (status_count == 2) {  // get MemTotal and MemAvailable yet
      break;
    }

    // get title
    std::string line(buf);
    std::string::size_type position = line.find(":");
    std::string title = line.substr(0, position);

    // get the value when MemTotal or MemAvailable
    if (title == "MemTotal") {
      std::string::size_type pos1 = line.find_last_of(" ");
      std::string::size_type pos2 = line.find_last_of(" ", pos1 - 1);
      mem_total = atol(line.substr(pos2, pos1 - pos2).c_str());
      status_count++;
    } else if (title == "MemAvailable") {
      std::string::size_type pos1 = line.find_last_of(" ");
      std::string::size_type pos2 = line.find_last_of(" ", pos1 - 1);
      mem_available = atol(line.substr(pos2, pos1 - pos2).c_str());
      status_count++;
    }

    (void)memset_s(buf, sizeof(buf), 0, sizeof(buf));
  }
  fclose(fd);

  if (status_count != 2 || mem_total == 0 || mem_available > mem_total) {
    MS_LOG(WARNING) << "Get memory usage failed.";
    return 0.0;
  }

  return (1.0 - static_cast<float>(static_cast<double>(mem_available) / static_cast<double>(mem_total)));
}
#endif
}  // namespace dataset
}  // namespace mindspore
