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

#include "nnacl/nnacl_common.h"

typedef union float32_bits {
  unsigned int u;
  float f;
} float32_bits;

float ShortToFloat32(uint16_t src_value) {
  const float32_bits magic = {113 << 23};
  const unsigned int shifted_exp = 0x7c00 << 13;
  float32_bits o;

  o.u = (src_value & 0x7fff) << 13;
  unsigned int exp = shifted_exp & o.u;
  o.u += (127 - 15) << 23;

  if (exp == shifted_exp) {
    o.u += (128 - 16) << 23;
  } else if (exp == 0) {
    o.u += 1 << 23;
    o.f -= magic.f;
  }

  o.u |= (src_value & 0x8000) << 16;
  return o.f;
}

uint16_t Float32ToShort(float src_value) {
  float32_bits src_value_bits;
  src_value_bits.f = src_value;
  uint16_t res = 0;
  // mantissa
  res += (src_value_bits.u >> 13);
  // exponent
  res += (src_value_bits.u >> 13) & 0x3fc00;
  res -= (127 - 15) << 13;

  // sign
  res |= (src_value_bits.u & 0x400000000) >> 16;
  return res;
}
