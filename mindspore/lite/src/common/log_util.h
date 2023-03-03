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

#ifndef MINDSPORE_LITE_SRC_COMMON_LOG_UTIL_H_
#define MINDSPORE_LITE_SRC_COMMON_LOG_UTIL_H_

#include "src/common/log_adapter.h"
#include "include/errorcode.h"

#define MSLITE_CHECK_PTR(ptr)                                    \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return mindspore::lite::RET_ERROR;                         \
    }                                                            \
  } while (0)

#define CHECK_MALLOC_RES(ptr, errcode)        \
  do {                                        \
    if ((ptr) == nullptr) {                   \
      MS_LOG(ERROR) << "malloc data failed."; \
      return errcode;                         \
    }                                         \
  } while (0)

#define MSLITE_CHECK_PTR_RETURN(ptr, errcode)                     \
  do {                                                            \
    if ((ptr) == nullptr) {                                       \
      MS_LOG(ERROR) << ": The pointer [" << #ptr << "] is null."; \
      return errcode;                                             \
    }                                                             \
  } while (0)

#ifndef ENABLE_HIGH_PERFORMANCE
#define CHECK_NULL_RETURN(ptr)                       \
  do {                                               \
    if ((ptr) == nullptr) {                          \
      MS_LOG(ERROR) << #ptr << " must not be null!"; \
      return mindspore::lite::RET_NULL_PTR;          \
    }                                                \
  } while (0)

#define CHECK_NULL_RETURN_VOID(ptr)                  \
  do {                                               \
    if ((ptr) == nullptr) {                          \
      MS_LOG(ERROR) << #ptr << " must not be null!"; \
      return;                                        \
    }                                                \
  } while (0)

#define CHECK_LESS_RETURN(size1, size2)                               \
  do {                                                                \
    if ((size1) < (size2)) {                                          \
      MS_LOG(ERROR) << #size1 << " must not be less than " << #size2; \
      return mindspore::lite::RET_ERROR;                              \
    }                                                                 \
  } while (0)

#define CHECK_LARGE_RETURN(size1, size2)                               \
  do {                                                                 \
    if ((size1) > (size2)) {                                           \
      MS_LOG(ERROR) << #size1 << " must not be large than " << #size2; \
      return mindspore::lite::RET_ERROR;                               \
    }                                                                  \
  } while (0)

#define CHECK_NOT_EQUAL_RETURN(size1, size2)                     \
  do {                                                           \
    if ((size1) != (size2)) {                                    \
      MS_LOG(ERROR) << #size1 << " must be equal to " << #size2; \
      return mindspore::lite::RET_ERROR;                         \
    }                                                            \
  } while (0)

#define CHECK_EQUAL_RETURN(size1, size2)                             \
  do {                                                               \
    if ((size1) == (size2)) {                                        \
      MS_LOG(ERROR) << #size1 << " must be not equal to " << #size2; \
      return mindspore::lite::RET_ERROR;                             \
    }                                                                \
  } while (0)

#define CHECK_LESS_RETURN_RET(size1, size2, ret, free_parm)           \
  do {                                                                \
    if ((size1) < (size2)) {                                          \
      MS_LOG(ERROR) << #size1 << " must not be less than " << #size2; \
      free(free_parm);                                                \
      return ret;                                                     \
    }                                                                 \
  } while (0)

#else
#define CHECK_NULL_RETURN(ptr)
#define CHECK_NULL_RETURN_VOID(ptr)
#define CHECK_LESS_RETURN(size1, size2)
#define CHECK_NOT_EQUAL_RETURN(size1, size2)
#define CHECK_EQUAL_RETURN(size1, size2)
#define CHECK_LESS_RETURN_RET(size1, size2, ret, do_exec)
#endif
#endif  // MINDSPORE_LITE_SRC_COMMON_LOG_UTIL_H_
