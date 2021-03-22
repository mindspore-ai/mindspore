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

#include "nnacl/fp16/exp_fp16.h"
#include <math.h>
#include <string.h>
#include "nnacl/errorcode.h"

void ExpFp16(const float16_t *src, float16_t *dst, int num) {
  int i = 0;
#ifdef ENABLE_ARM64
  int count = (num / C8NUM) * C8NUM;
  for (; i < count; i += C8NUM) {
    simd_exp_fp16(vld1q_f16(src + i), dst + i);
  }
#endif
  for (; i < num; ++i) {
    single_exp_fp16(src[i], dst + i);
  }
}
