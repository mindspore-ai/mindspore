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
#include <float.h>
#include "nnacl/fp32/dropout_fp32.h"
#include "nnacl/dropout_fp32_simd.h"

void DropoutFp32(const float *input, float scale, int length, float *output) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(DropoutFp32, i, input, scale, length, output);

  for (; i < length; ++i) {
    output[i] = scale * input[i];
  }
}
