/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_CHECK_BASE_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_CHECK_BASE_H_

#include <limits.h>
#include "common/log_util.h"

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))
#define DOWN_DIV(x, y) ((x) / (y))
#define DOWN_ROUND(x, y) ((x) / (y) * (y))

#define kNHWC_N 0
#define kNHWC_H 1
#define kNHWC_W 2
#define kNHWC_C 3
#define kNCHW_N 0
#define kNCHW_C 1
#define kNCHW_H 2
#define kNCHW_W 3
#define kInputSize1 1
#define kInputSize2 2
#define kInputSize3 3
#ifdef Debug
#include <cassert>
#define MS_ASSERT(f) assert(f)
#else
#define MS_ASSERT(f) ((void)0)
#endif

#define SIZE_MUL_OVERFLOW(x, y) (((x) == 0) ? false : (SIZE_MAX / (x)) < (y))
#define INT_MUL_OVERFLOW(x, y)                                                                 \
  (((x) == 0) ? false                                                                          \
              : ((x) > 0 ? (((y) >= 0) ? (INT_MAX / (x)) < (y) : (INT_MAX / (x)) < (-1 * (y))) \
                         : (((y) >= 0) ? (INT_MAX / (x)) > (-1 * (y)) : (INT_MAX / (x)) > (y))))

#define INT_MUL_OVERFLOW_THRESHOLD(x, y, threshold)                                                    \
  (((x) == 0) ? false                                                                                  \
              : ((x) > 0 ? (((y) >= 0) ? ((threshold) / (x)) < (y) : ((threshold) / (x)) < (-1 * (y))) \
                         : (((y) >= 0) ? ((threshold) / (x)) > (-1 * (y)) : ((threshold) / (x)) > (y))))

#define INT_ADD_OVERFLOW(x, y) (INT_MAX - (x)) < (y)

#define CHECK_LESS_RETURN(size1, size2)                               \
  do {                                                                \
    if ((size1) < (size2)) {                                          \
      MS_LOG(ERROR) << #size1 << " must not be less than " << #size2; \
      return mindspore::lite::RET_ERROR;                              \
    }                                                                 \
  } while (0)

// Check whether value is true, if not return 'errcode'
// and print error string msg
#define MS_CHECK_TRUE_MSG(value, errcode, msg) \
  do {                                         \
    if (!(value)) {                            \
      MS_LOG(ERROR) << msg;                    \
      return errcode;                          \
    }                                          \
  } while (0)

// Check whether value is true, if not return void
// and print error string msg
#define MS_CHECK_TRUE_MSG_VOID(value, msg) \
  do {                                     \
    if (!(value)) {                        \
      MS_LOG(ERROR) << msg;                \
      return;                              \
    }                                      \
  } while (0)

#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_COMMON_CHECK_BASE_H_
