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

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_REQUEST_PROCESS_RESULT_CODE_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_REQUEST_PROCESS_RESULT_CODE_H_

#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <iostream>

namespace mindspore {
namespace ps {
namespace core {
enum class RequestProcessResultCode { kSuccess = 0, kSystemError = 1, kInvalidInputs = 2 };
class LogStream {
 public:
  LogStream() { sstream_ = std::make_shared<std::stringstream>(); }
  ~LogStream() = default;

  template <typename T>
  LogStream &operator<<(const T &val) noexcept {
    (*sstream_) << val;
    return *this;
  }

  template <typename T>
  LogStream &operator<<(const std::vector<T> &val) noexcept {
    (*sstream_) << "[";
    for (size_t i = 0; i < val.size(); i++) {
      (*this) << val[i];
      if (i + 1 < val.size()) {
        (*sstream_) << ", ";
      }
    }
    (*sstream_) << "]";
    return *this;
  }

  LogStream &operator<<(std::ostream &func(std::ostream &os)) noexcept {
    (*sstream_) << func;
    return *this;
  }

  const std::shared_ptr<std::stringstream> &stream() const { return sstream_; }

 private:
  std::shared_ptr<std::stringstream> sstream_;
};

/* This class encapsulates user defined messages and user defined result codes, used to return http response message.
 *
 */
class RequestProcessResult {
 public:
  RequestProcessResult() : code_(RequestProcessResultCode::kSystemError) {}
  explicit RequestProcessResult(enum RequestProcessResultCode code, const std::string &msg = "")
      : code_(code), msg_(msg) {}
  ~RequestProcessResult() = default;

  bool IsSuccess() const { return code_ == RequestProcessResultCode::kSuccess; }
  enum RequestProcessResultCode ResultCode() const { return code_; }
  std::string StatusMessage() const { return msg_; }

  bool operator==(const RequestProcessResult &other) const { return code_ == other.code_; }
  bool operator==(enum RequestProcessResultCode other_code) const { return code_ == other_code; }
  bool operator!=(const RequestProcessResult &other) const { return code_ != other.code_; }
  bool operator!=(enum RequestProcessResultCode other_code) const { return code_ != other_code; }

  operator bool() const = delete;

  RequestProcessResult &operator<(const LogStream &stream) noexcept {
    msg_ = stream.stream()->str();
    return *this;
  }
  RequestProcessResult &operator=(const std::string &message) noexcept {
    msg_ = message;
    return *this;
  }

 private:
  enum RequestProcessResultCode code_;
  std::string msg_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_REQUEST_PROCESS_RESULT_CODE_H_
