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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_ERROR_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_ERROR_H_

#include <map>
#include <string>
#include "include/api/status.h"

namespace mindspore {
namespace mindrecord {
#define RETURN_IF_NOT_OK(_s) \
  do {                       \
    Status __rc = (_s);      \
    if (__rc.IsError()) {    \
      return __rc;           \
    }                        \
  } while (false)

#define RELEASE_AND_RETURN_IF_NOT_OK(_s, _db, _in) \
  do {                                             \
    Status __rc = (_s);                            \
    if (__rc.IsError()) {                          \
      if ((_db) != nullptr) {                      \
        sqlite3_close(_db);                        \
      }                                            \
      (_in).close();                               \
      return __rc;                                 \
    }                                              \
  } while (false)

#define CHECK_FAIL_RETURN_UNEXPECTED(_condition, _e)                         \
  do {                                                                       \
    if (!(_condition)) {                                                     \
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, _e); \
    }                                                                        \
  } while (false)

#define RETURN_UNEXPECTED_IF_NULL(_ptr)                                         \
  do {                                                                          \
    if ((_ptr) == nullptr) {                                                    \
      std::string err_msg = "The pointer[" + std::string(#_ptr) + "] is null."; \
      RETURN_STATUS_UNEXPECTED(err_msg);                                        \
    }                                                                           \
  } while (false)

#define RETURN_STATUS_UNEXPECTED(_e)                                       \
  do {                                                                     \
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, _e); \
  } while (false)

enum MSRStatus {
  SUCCESS = 0,
  FAILED = 1,
};

}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_ERROR_H_
