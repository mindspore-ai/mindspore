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
  } while (0);

#endif  // MINDSPORE_LITE_SRC_COMMON_LOG_UTIL_H_
