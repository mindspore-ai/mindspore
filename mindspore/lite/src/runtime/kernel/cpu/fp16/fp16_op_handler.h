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
#ifdef ENABLE_ARM
#include <arm_neon.h>
#ifdef ENABLE_FP16
#include "nnacl/fp16/cast_fp16.h"
#endif
#endif
#include "nnacl/nnacl_common.h"

#ifdef __cplusplus
extern "C" {
#endif
static inline void Float32ToFloat16_fp16_handler(const void *input, void *output, int number, bool support_fp16) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  if (support_fp16) {
    Float32ToFloat16(reinterpret_cast<const float *>(input), reinterpret_cast<float16_t *>(output), number);
  } else {
#endif
    auto src_data = reinterpret_cast<const float *>(input);
    auto dst_data = reinterpret_cast<uint16_t *>(output);
    for (int i = 0; i < number; i++) {
      dst_data[i] = Float32ToShort(src_data[i]);
    }
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  }
#endif
}

static inline void Float16ToFloat32_fp16_handler(const void *input, void *output, int number, bool support_fp16) {
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  if (support_fp16) {
    Float16ToFloat32(reinterpret_cast<const float16_t *>(input), reinterpret_cast<float *>(output), number);
  } else {
#endif
    auto src_data = reinterpret_cast<const uint16_t *>(input);
    auto dst_data = reinterpret_cast<float *>(output);
    for (int i = 0; i < number; i++) {
      dst_data[i] = ShortToFloat32(src_data[i]);
    }
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
  }
#endif
}

#ifdef __cplusplus
}
#endif
