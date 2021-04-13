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
#endif
#include "nnacl/fp16/cast_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
static inline void Float32ToFloat16_fp16_handler(const void *input, void *output, int number) {
  Float32ToFloat16(reinterpret_cast<const float *>(input), reinterpret_cast<float16_t *>(output), number);
}
static inline void Float16ToFloat32_fp16_handler(const void *input, void *output, int number) {
  Float16ToFloat32(reinterpret_cast<const float16_t *>(input), reinterpret_cast<float *>(output), number);
}

#ifdef __cplusplus
}
#endif
