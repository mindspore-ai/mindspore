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

#include "nnacl/fp16_grad/dropout_grad.h"

void DropoutFp16Grad(const float16_t *yt_ptr, const float16_t *mask, float16_t *output_ptr, int length,
                     float16_t scale) {
  for (int i = 0; i < length; i++) {
    output_ptr[i] = yt_ptr[i] * mask[i] * scale;
  }
}
