/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/gelu_fp32.h"
#include "nnacl/gelu_parameter.h"
#include <string.h>
#include <math.h>
#include "nnacl/errorcode.h"

int DoGeLU(const float *src, float *out, int64_t real_dst_count, const GeLUParameter *param) {
  if (src == NULL || out == NULL) {
    return NNACL_ERR;
  }

  if (param->approximate_) {
    for (int i = 0; i < real_dst_count; i++) {
      out[i] = 0.5 * src[i] * (1.0 + tanh(0.7978845608028654 * (src[i] + 0.044715 * pow(src[i], 3))));
    }
  } else {
    for (int i = 0; i < real_dst_count; i++) {
      out[i] = 0.5 * src[i] * (1.0 + erf(src[i] / 1.4142135623730951));
    }
  }

  return NNACL_OK;
}
