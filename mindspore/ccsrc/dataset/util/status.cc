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
#include "dataset/util/status.h"
#include <sstream>
#include "common/utils.h"
#include "dataset/util/task_manager.h"

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
      case StatusCode::kUnexpectedError:
      default:
        s = "Unexpected error";
        break;
    }
  }
  return std::string(s);
}

Status::Status(StatusCode c) noexcept : code_(c), err_msg_(std::move(CodeAsString(c))) {}

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
  ss << "Thread ID " << this_thread::get_id() << " " << CodeAsString(code) << ". ";
  if (!extra.empty()) {
    ss << extra;
  }
  ss << "\n";
  ss << "Line of code : " << line_of_code << "\n";
  if (file_name != nullptr) {
    ss << "File         : " << file_name << "\n";
  }
  err_msg_ = ss.str();
  MS_LOG(INFO) << err_msg_;
}

std::ostream &operator<<(std::ostream &os, const Status &s) {
  os << s.ToString();
  return os;
}

std::string Status::ToString() const { return err_msg_; }

StatusCode Status::get_code() const { return code_; }
}  // namespace dataset
}  // namespace mindspore
