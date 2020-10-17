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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_STATUS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_STATUS_H_

#if defined(__GNUC__) || defined(__clang__)
#define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif

#include <iostream>
#include <string>
#include <utility>

namespace mindspore {
namespace dataset {
#define RETURN_IF_NOT_OK(_s) \
  do {                       \
    Status __rc = (_s);      \
    if (__rc.IsError()) {    \
      return __rc;           \
    }                        \
  } while (false)

#define RETURN_STATUS_UNEXPECTED(_e)                                     \
  do {                                                                   \
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, _e); \
  } while (false)

#define CHECK_FAIL_RETURN_UNEXPECTED(_condition, _e)                       \
  do {                                                                     \
    if (!(_condition)) {                                                   \
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, _e); \
    }                                                                      \
  } while (false)

#define RETURN_UNEXPECTED_IF_NULL(_ptr)                                         \
  do {                                                                          \
    if ((_ptr) == nullptr) {                                                    \
      std::string err_msg = "The pointer[" + std::string(#_ptr) + "] is null."; \
      RETURN_STATUS_UNEXPECTED(err_msg);                                        \
    }                                                                           \
  } while (false)

#define RETURN_OK_IF_TRUE(_condition) \
  do {                                \
    if (_condition) {                 \
      return Status::OK();            \
    }                                 \
  } while (false)

#define RETURN_STATUS_SYNTAX_ERROR(_e)                               \
  do {                                                               \
    return Status(StatusCode::kSyntaxError, __LINE__, __FILE__, _e); \
  } while (false)

enum class StatusCode : char {
  kOK = 0,
  kOutOfMemory = 1,
  kShapeMisMatch = 2,
  kInterrupted = 3,
  kNoSpace = 4,
  kPyFuncException = 5,
  kDuplicateKey = 6,
  kPythonInterpreterFailure = 7,
  kTDTPushFailure = 8,
  kFileNotExist = 9,
  kProfilingError = 10,
  kBoundingBoxOutOfBounds = 11,
  kBoundingBoxInvalidShape = 12,
  kSyntaxError = 13,
  kTimeOut = 14,
  kBuddySpaceFull = 15,
  kNetWorkError = 16,
  kNotImplementedYet = 17,
  // Make this error code the last one. Add new error code above it.
  kUnexpectedError = 127
};

std::string CodeAsString(const StatusCode c);

class Status {
 public:
  Status() noexcept;

  explicit Status(StatusCode c) noexcept;

  ~Status() noexcept;

  // Copy constructor
  Status(const Status &s);

  Status &operator=(const Status &s);

  // Move constructor
  Status(Status &&) noexcept;

  Status &operator=(Status &&) noexcept;

  Status(const StatusCode code, const std::string &msg);

  Status(const StatusCode code, int line_of_code, const char *file_name, const std::string &extra = "");

  // Return a success status
  static Status OK() { return Status(StatusCode::kOK); }

  std::string ToString() const;

  StatusCode get_code() const;

  friend std::ostream &operator<<(std::ostream &os, const Status &s);

  explicit operator bool() const { return (get_code() == StatusCode::kOK); }

  bool operator==(const Status &other) const { return (this->get_code() == other.get_code()); }

  bool operator!=(const Status &other) const { return !(*this == other); }

  bool IsOk() const { return (get_code() == StatusCode::kOK); }

  bool IsError() const { return !IsOk(); }

  bool IsOutofMemory() const { return (get_code() == StatusCode::kOutOfMemory); }

  bool IsInterrupted() const { return (get_code() == StatusCode::kInterrupted); }

  bool IsShapeIncorrect() const { return (get_code() == StatusCode::kShapeMisMatch); }

  bool IsNoSpace() const { return (get_code() == StatusCode::kNoSpace); }

  bool IsNetWorkError() const { return (get_code() == StatusCode::kNetWorkError); }

 private:
  StatusCode code_;
  std::string err_msg_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_STATUS_H_
