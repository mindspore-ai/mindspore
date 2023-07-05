/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_FP32_FILL_FP32_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_FP32_FILL_FP32_WRAPPER_H_

#include <stdbool.h>

typedef struct {
  float *output_;
  int size_;
  float data_;
} FillFp32Args;

typedef struct {
  int *output_;
  int size_;
  int data_;
} FillInt32Args;

typedef struct {
  bool *output_;
  int size_;
  bool data_;
} FillBoolArgs;

#ifdef __cplusplus
extern "C" {
#endif

int DoFillFp32(void *cdata, int task_id, float lhs_scale, float rhs_scale);
int DoFillInt32(void *cdata, int task_id, float lhs_scale, float rhs_scale);
int DoFillBool(void *cdata, int task_id, float lhs_scale, float rhs_scale);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_FP32_FILL_FP32_WRAPPER_H_
