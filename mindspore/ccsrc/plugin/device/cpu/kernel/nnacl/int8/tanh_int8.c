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

#include "nnacl/int8/tanh_int8.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

void TanhInt8(const int8_t *input_ptr, int8_t *output_ptr, int size, const TanhQuantParameter *quant) {
  for (int i = 0; i < size; ++i) {
    float fp32_src = (input_ptr[i] - quant->in_zp_) * quant->in_scale_;
    float fp32_dst = TanhOpt(fp32_src);
    int32_t int32_dst = (int32_t)round(fp32_dst * 1.0 / quant->out_scale_ + quant->out_zp_);
    output_ptr[i] = (int8_t)MSMAX(MSMIN(int32_dst, 127), -128);
  }
  return;
}
