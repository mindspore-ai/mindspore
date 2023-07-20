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

#ifndef MINDSPORE_CCSRC_C_API_BASE_STATUS_H_
#define MINDSPORE_CCSRC_C_API_BASE_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum STATUS {
  RET_OK = 0,        /* No error */
  RET_ERROR = -1,    /* Normal error */
  RET_NULL_PTR = -2, /* Nullptr error */
} STATUS;

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_BASE_STATUS_H_
