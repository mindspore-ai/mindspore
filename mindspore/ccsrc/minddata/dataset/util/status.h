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

#include "include/api/status.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
#define RETURN_IF_NOT_OK(_s)       \
  do {                             \
    mindspore::Status __rc = (_s); \
    if (__rc.IsError()) {          \
      return __rc;                 \
    }                              \
  } while (false)

#define STATUS_ERROR(_error_code, _e) mindspore::Status(_error_code, __LINE__, DATASET_SRC_FILE_NAME, _e)

#define RETURN_STATUS_ERROR(_error_code, _e) \
  do {                                       \
    return STATUS_ERROR(_error_code, _e);    \
  } while (false)

#define RETURN_STATUS_UNEXPECTED(_e)                                    \
  do {                                                                  \
    RETURN_STATUS_ERROR(mindspore::StatusCode::kMDUnexpectedError, _e); \
  } while (false)

#define CHECK_FAIL_RETURN_UNEXPECTED(_condition, _e) \
  do {                                               \
    if (!(_condition)) {                             \
      RETURN_STATUS_UNEXPECTED(_e);                  \
    }                                                \
  } while (false)

#define RETURN_SYNTAX_ERROR(_e)                                     \
  do {                                                              \
    RETURN_STATUS_ERROR(mindspore::StatusCode::kMDSyntaxError, _e); \
  } while (false)

#define CHECK_FAIL_RETURN_SYNTAX_ERROR(_condition, _e) \
  do {                                                 \
    if (!(_condition)) {                               \
      RETURN_SYNTAX_ERROR(_e);                         \
    }                                                  \
  } while (false)

#define LOG_AND_RETURN_STATUS_SYNTAX_ERROR(_e) \
  do {                                         \
    MS_LOG(ERROR) << _e;                       \
    RETURN_SYNTAX_ERROR(_e);                   \
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
      return mindspore::Status::OK(); \
    }                                 \
  } while (false)

#define RETURN_SECOND_IF_ERROR(_s, _r) \
  do {                                 \
    mindspore::Status __rc = (_s);     \
    if (__rc.IsError()) {              \
      MS_LOG(ERROR) << __rc;           \
      return _r;                       \
    }                                  \
  } while (false)

#define RETURN_STATUS_OOM(_e)                                       \
  do {                                                              \
    RETURN_STATUS_ERROR(mindspore::StatusCode::kMDOutOfMemory, _e); \
  } while (false)

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
const float MAX_MEMORY_USAGE_THRESHOLD = 0.8;
float GetMemoryUsage();
#endif
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_STATUS_H_
