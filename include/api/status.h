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
#ifndef MINDSPORE_INCLUDE_API_STATUS_H
#define MINDSPORE_INCLUDE_API_STATUS_H

#include <string>

namespace mindspore {
namespace api {
enum StatusCode {
  SUCCESS = 0,
  FAILED,
  INVALID_INPUTS,
  // insert new status code here
  UNKNOWN = 0xFFFFFFFF
};

class Status {
 public:
  Status() : status_code_(FAILED) {}
  Status(enum StatusCode status_code, const std::string &status_msg = "")    // NOLINT(runtime/explicit)
    : status_code_(status_code), status_msg_(status_msg) {}
  ~Status() = default;

  bool IsSuccess() const { return status_code_ == SUCCESS; }
  enum StatusCode StatusCode() const { return status_code_; }
  std::string StatusMessage() const { return status_msg_; }
  bool operator==(const Status &other) const { return status_code_ == other.status_code_; }
  bool operator==(enum StatusCode other_code) const { return status_code_ == other_code; }
  bool operator!=(const Status &other) const { return status_code_ != other.status_code_; }
  bool operator!=(enum StatusCode other_code) const { return status_code_ != other_code; }
  operator bool() const = delete;

 private:
  enum StatusCode status_code_;
  std::string status_msg_;
};
}  // namespace api
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_STATUS_H
