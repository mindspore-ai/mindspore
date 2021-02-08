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

#ifndef MINDSPORE_LITE_MICRO_CODER_LOG_H_
#define MINDSPORE_LITE_MICRO_CODER_LOG_H_

#include "src/common/log_adapter.h"
#include "include/errorcode.h"

#define MS_CHECK_PTR(ptr)                                        \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return mindspore::lite::RET_ERROR;                         \
    }                                                            \
  } while (0)

#define MS_CHECK_PTR_WITH_EXE(ptr, FUNC)                         \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      FUNC;                                                      \
      return mindspore::lite::RET_ERROR;                         \
    }                                                            \
  } while (0)

#define MS_CHECK_PTR_RET_NULL(ptr)                               \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return nullptr;                                            \
    }                                                            \
  } while (0)

#define MS_CHECK_PTR_IF_NULL(ptr)                                \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return;                                                    \
    }                                                            \
  } while (0)

#define MS_CHECK_RET_CODE(code, msg)     \
  do {                                   \
    if ((code) != RET_OK) {              \
      MS_LOG(ERROR) << msg;              \
      return mindspore::lite::RET_ERROR; \
    }                                    \
  } while (0)

#define MS_CHECK_RET_CODE_WITH_EXE(code, msg, FUNC) \
  do {                                              \
    if ((code) != RET_OK) {                         \
      MS_LOG(ERROR) << msg;                         \
      FUNC;                                         \
      return mindspore::lite::RET_ERROR;            \
    }                                               \
  } while (0)

#define MS_CHECK_RET_CODE_RET_NULL(code, msg) \
  do {                                        \
    if ((code) != RET_OK) {                   \
      MS_LOG(ERROR) << msg;                   \
      return nullptr;                         \
    }                                         \
  } while (0)

#define MS_CHECK_TRUE(code, msg)         \
  do {                                   \
    if (!(code)) {                       \
      MS_LOG(ERROR) << msg;              \
      return mindspore::lite::RET_ERROR; \
    }                                    \
  } while (0)

#define MS_CHECK_TRUE_WITH_EXE(code, msg, FUNC) \
  do {                                          \
    if (!(code)) {                              \
      MS_LOG(ERROR) << msg;                     \
      FUNC;                                     \
      return mindspore::lite::RET_ERROR;        \
    }                                           \
  } while (0)

#define MS_CHECK_TRUE_WITHOUT_RET(code, msg) \
  do {                                       \
    if (!(code)) {                           \
      MS_LOG(ERROR) << msg;                  \
      return;                                \
    }                                        \
  } while (0)

#define MS_CHECK_TRUE_RET_NULL(code, msg) \
  do {                                    \
    if (!(code)) {                        \
      MS_LOG(ERROR) << msg;               \
      return nullptr;                     \
    }                                     \
  } while (0)

#define MS_CHECK_TRUE_RET_BOOL(code, msg) \
  do {                                    \
    if (!(code)) {                        \
      MS_LOG(ERROR) << msg;               \
      return false;                       \
    }                                     \
  } while (0)

#endif  // MINDSPORE_LITE_MICRO_CODER_LOG_H_
