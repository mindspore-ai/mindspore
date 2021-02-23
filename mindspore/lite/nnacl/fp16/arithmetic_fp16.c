/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp16/arithmetic_fp16.h"
#include <math.h>
#include "nnacl/common_func.h"
#include "nnacl/nnacl_utils.h"

int BroadcastAddFp16(const float16_t *in0, const float16_t *in1, float16_t *tile_in0, float16_t *tile_in1,
                     float16_t *out, int size, ArithmeticParameter *param) {
  TileDimensionsFp16(in0, in1, tile_in0, tile_in1, param);
  return ElementAddFp16(tile_in0, tile_in1, out, size);
}

void TileOneDimensionFp16(const float16_t *inData, float16_t *outData, int dim, size_t ndim, const int *inShape,
                          const int *inStrides, const int *outStrides, const int *multiple) {
  int srcDimSize = inShape[dim];
  if (dim == ndim - 1) {
    for (int i = 0; i < multiple[dim]; i++) {
      memcpy(outData, inData, srcDimSize * sizeof(float16_t));
      outData += srcDimSize;
    }
    return;
  }
  for (size_t i = 0; i < srcDimSize; i++) {
    for (size_t j = 0; j < multiple[dim]; j++) {
      TileOneDimensionFp16(inData + inStrides[dim] * i, outData + outStrides[dim] * (i + j * srcDimSize), dim + 1, ndim,
                           inShape, inStrides, outStrides, multiple);
    }
  }
}

void TileDimensionsFp16(const float16_t *data0, const float16_t *data1, float16_t *tile_data0, float16_t *tile_data1,
                        ArithmeticParameter *param) {
  CalcMultiplesAndStrides(param);
  TileOneDimensionFp16(data0, tile_data0, 0, param->ndim_, param->in_shape0_, param->in_strides0_, param->out_strides_,
                       param->multiples0_);
  TileOneDimensionFp16(data1, tile_data1, 0, param->ndim_, param->in_shape1_, param->in_strides1_, param->out_strides_,
                       param->multiples1_);
}

int ElementMulFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] * input1[index];
  }
  return NNACL_OK;
}

int ElementOptMulFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vmulq_f16(vin0_opt, vin1);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] * input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vmulq_f16(vin0, vin1_opt);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] * input1[0];
    }
  }
  return NNACL_OK;
}

int ElementMulReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
#ifdef ENABLE_NEON
  float16x8_t zeros = vdupq_n_f16(0.0);
#endif
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    float16_t res = input0[index] * input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementOptMulReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t zeros = vdupq_n_f16(0.0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vmulq_f16(vin0_opt, vin1);
      vout = vmaxq_f16(vout, zeros);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      float16_t res = input0[0] * input1[index];
      output[index] = res > 0 ? res : 0;
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vmulq_f16(vin0, vin1_opt);
      vout = vmaxq_f16(vout, zeros);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      float16_t res = input0[index] * input1[0];
      output[index] = res > 0 ? res : 0;
    }
  }
  return NNACL_OK;
}

int ElementMulRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t zeros = vdupq_n_f16(0.0);
  float16x8_t bounds = vdupq_n_f16(6.0);
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vmulq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMIN(MSMAX(input0[index] * input1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementOptMulRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t zeros = vdupq_n_f16(0.0);
  float16x8_t bounds = vdupq_n_f16(6.0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vmulq_f16(vin0_opt, vin1);
      vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[0] * input1[index], 0), 6);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vmulq_f16(vin0, vin1_opt);
      vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] * input1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementAddFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vst1q_f16(output + index, vout);
  }
  for (; index <= element_size - 4; index += C4NUM) {
    float16x4_t vin0 = vld1_f16(input0 + index);
    float16x4_t vin1 = vld1_f16(input1 + index);
    float16x4_t vout = vadd_f16(vin0, vin1);
    vst1_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] + input1[index];
  }
  return NNACL_OK;
}

int ElementOptAddFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vaddq_f16(vin0_opt, vin1);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] + input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vaddq_f16(vin0, vin1_opt);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] + input1[0];
    }
  }
  return NNACL_OK;
}

int ElementAddReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t zeros = vdupq_n_f16(0.0);
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output + index, vout);
  }
  float16x4_t zeros1 = vdup_n_f16(0.0f);
  for (; index <= element_size - 4; index += C4NUM) {
    float16x4_t vin0 = vld1_f16(input0 + index);
    float16x4_t vin1 = vld1_f16(input1 + index);
    float16x4_t vout = vadd_f16(vin0, vin1);
    vout = vmax_f16(vout, zeros1);
    vst1_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    float16_t res = input0[index] + input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementOptAddReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t zeros = vdupq_n_f16(0.0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vaddq_f16(vin0_opt, vin1);
      vout = vmaxq_f16(vout, zeros);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      float16_t res = input0[0] + input1[index];
      output[index] = res > 0 ? res : 0;
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vaddq_f16(vin0, vin1_opt);
      vout = vmaxq_f16(vout, zeros);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      float16_t res = input0[index] + input1[0];
      output[index] = res > 0 ? res : 0;
    }
  }
  return NNACL_OK;
}

int ElementAddRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t zeros = vdupq_n_f16(0.0);
  float16x8_t bounds = vdupq_n_f16(6.0);
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vaddq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output + index, vout);
  }
  float16x4_t zeros1 = vdup_n_f16(0.0);
  float16x4_t bounds1 = vdup_n_f16(6.0);
  for (; index <= element_size - 4; index += C4NUM) {
    float16x4_t vin0 = vld1_f16(input0 + index);
    float16x4_t vin1 = vld1_f16(input1 + index);
    float16x4_t vout = vadd_f16(vin0, vin1);
    vout = vmin_f16(vmax_f16(vout, zeros1), bounds1);
    vst1_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMIN(MSMAX(input0[index] + input1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementOptAddRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t zeros = vdupq_n_f16(0.0);
  float16x8_t bounds = vdupq_n_f16(6.0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vaddq_f16(vin0_opt, vin1);
      vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[0] + input1[index], 0), 6);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vaddq_f16(vin0, vin1_opt);
      vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] + input1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementSubFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] - input1[index];
  }
  return NNACL_OK;
}

int ElementOptSubFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vsubq_f16(vin0_opt, vin1);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] - input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vsubq_f16(vin0, vin1_opt);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] - input1[0];
    }
  }
  return NNACL_OK;
}

int ElementSubReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t zeros = vdupq_n_f16(0.0);
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    float16_t res = input0[index] - input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementOptSubReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t zeros = vdupq_n_f16(0.0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vsubq_f16(vin0_opt, vin1);
      vout = vmaxq_f16(vout, zeros);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      float16_t res = input0[0] - input1[index];
      output[index] = res > 0 ? res : 0;
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vsubq_f16(vin0, vin1_opt);
      vout = vmaxq_f16(vout, zeros);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      float16_t res = input0[index] - input1[0];
      output[index] = res > 0 ? res : 0;
    }
  }
  return NNACL_OK;
}

int ElementSubRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t zeros = vdupq_n_f16(0.0);
  float16x8_t bounds = vdupq_n_f16(6.0);
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vsubq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMIN(MSMAX(input0[index] - input1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementOptSubRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t zeros = vdupq_n_f16(0.0);
  float16x8_t bounds = vdupq_n_f16(6.0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vsubq_f16(vin0_opt, vin1);
      vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[0] - input1[index], 0), 6);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vsubq_f16(vin0, vin1_opt);
      vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] - input1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    NNACL_ASSERT(input1[index] != 0);
    output[index] = input0[index] / input1[index];
  }
  return NNACL_OK;
}

int ElementOptDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vdivq_f16(vin0_opt, vin1);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      NNACL_ASSERT(input1[index] != 0);
      output[index] = input0[0] / input1[index];
    }
  } else {
    if (input1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vdivq_f16(vin0, vin1_opt);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] / input1[0];
    }
  }
  return NNACL_OK;
}

int ElementDivReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t zeros = vdupq_n_f16(0.0);
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vout = vmaxq_f16(vout, zeros);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    if (input1[index] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    NNACL_ASSERT(input1[index] != 0);
    float16_t res = input0[index] / input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementOptDivReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t zeros = vdupq_n_f16(0.0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vmaxq_f16(vdivq_f16(vin0_opt, vin1), zeros);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      if (input1[index] == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
      NNACL_ASSERT(input1[index] != 0);
      output[index] = MSMAX(input0[0] / input1[index], 0);
    }
  } else {
    if (input1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vmaxq_f16(vdivq_f16(vin0, vin1_opt), zeros);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[index] / input1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementDivRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t zeros = vdupq_n_f16(0.0);
  float16x8_t bounds = vdupq_n_f16(6.0);
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vdivq_f16(vin0, vin1);
    vout = vminq_f16(vmaxq_f16(vout, zeros), bounds);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    if (input1[index] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[index] = MSMIN(MSMAX(input0[index] / input1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementOptDivRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t zeros = vdupq_n_f16(0.0);
  float16x8_t bounds = vdupq_n_f16(6.0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vminq_f16(vmaxq_f16(vdivq_f16(vin0_opt, vin1), zeros), bounds);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      if (input1[index] == 0) {
        return NNACL_ERRCODE_DIVISOR_ZERO;
      }
      output[index] = MSMIN(MSMAX(input0[0] / input1[index], 0), 6);
    }
  } else {
    if (input1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vminq_f16(vmaxq_f16(vdivq_f16(vin0, vin1_opt), zeros), bounds);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] / input1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementFloorModFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; ++i) {
    if (input1[i] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = input0[i] - floorf(input0[i] / input1[i]) * input1[i];
  }
  return NNACL_OK;
}

int ElementOptFloorModFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  if (param->in_elements_num1_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      NNACL_ASSERT(input1[0] != 0);
      output[i] = input0[i] - floorf(input0[i] / input1[0]) * input1[0];
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      NNACL_ASSERT(input1[i] != 0);
      output[i] = input0[i] - floorf(input0[i] / input1[i]) * input1[i];
    }
  }
  return NNACL_OK;
}

int ElementFloorDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; ++i) {
    NNACL_ASSERT(input1[i] != 0);
    output[i] = floorf(input0[i] / input1[i]);
  }
  return NNACL_OK;
}
int ElementOptFloorDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param) {
  if (param->in_elements_num1_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      NNACL_ASSERT(input1[0] != 0);
      output[i] = floorf(input0[i] / input1[0]);
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      NNACL_ASSERT(input1[i] != 0);
      output[i] = floorf(input0[i] / input1[i]);
    }
  }
  return NNACL_OK;
}

int ElementLogicalAndFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t vtrue = vdupq_n_f16(1);
  float16x8_t vfalse = vdupq_n_f16(0);
  uint16x8_t mask = vmovq_n_u16(((uint16_t)(1u << 15) - 1));
  uint16x8_t zeros = vdupq_n_u16(0);
  for (; index <= element_size - 8; index += C8NUM) {
    uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vld1q_f16(input0 + index)), mask);
    uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vld1q_f16(input1 + index)), mask);
    float16x8_t vout = vbslq_f16(vceqq_u16(vandq_u16(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float16_t)((bool)(input0[index]) & (bool)(input1[index]));
  }
  return NNACL_OK;
}

int ElementOptLogicalAndFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                             ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t vtrue = vdupq_n_f16(1);
  float16x8_t vfalse = vdupq_n_f16(0);
  uint16x8_t mask = vmovq_n_u16(((uint16_t)(1u << 15) - 1));
  uint16x8_t zeros = vdupq_n_u16(0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1_ = vld1q_f16(input1 + index);
      uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vin0_opt), mask);
      uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vin1_), mask);
      float16x8_t vout = vbslq_f16(vceqq_u16(vandq_u16(vin0, vin1), zeros), vfalse, vtrue);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = (float16_t)((bool)(input0[0]) & (bool)(input1[index]));
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0_ = vld1q_f16(input0 + index);
      uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vin0_), mask);
      uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vin1_opt), mask);
      float16x8_t vout = vbslq_f16(vceqq_u16(vandq_u16(vin0, vin1), zeros), vfalse, vtrue);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = (float16_t)((bool)(input0[index]) & (bool)(input1[0]));
    }
  }
  return NNACL_OK;
}

int ElementLogicalOrFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float16x8_t vtrue = vdupq_n_f16(1);
  float16x8_t vfalse = vdupq_n_f16(0);
  uint16x8_t mask = vmovq_n_u16(((uint16_t)(1u << 15) - 1));
  uint16x8_t zeros = vdupq_n_u16(0);
  for (; index <= element_size - 8; index += C8NUM) {
    uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vld1q_f16(input0 + index)), mask);
    uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vld1q_f16(input1 + index)), mask);
    float16x8_t vout = vbslq_f16(vceqq_u16(vorrq_u16(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float16_t)((bool)(input0[index]) | (bool)(input1[index]));
  }
  return NNACL_OK;
}

int ElementOptLogicalOrFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                            ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
  float16x8_t vtrue = vdupq_n_f16(1);
  float16x8_t vfalse = vdupq_n_f16(0);
  uint16x8_t mask = vmovq_n_u16(((uint16_t)(1u << 15) - 1));
  uint16x8_t zeros = vdupq_n_u16(0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1_ = vld1q_f16(input1 + index);
      uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vin0_opt), mask);
      uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vin1_), mask);
      float16x8_t vout = vbslq_f16(vceqq_u16(vorrq_u16(vin0, vin1), zeros), vfalse, vtrue);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = (float16_t)((bool)(input0[0]) | (bool)(input1[index]));
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0_ = vld1q_f16(input0 + index);
      uint16x8_t vin0 = vandq_u16(vreinterpretq_s16_f16(vin0_), mask);
      uint16x8_t vin1 = vandq_u16(vreinterpretq_s16_f16(vin1_opt), mask);
      float16x8_t vout = vbslq_f16(vceqq_u16(vorrq_u16(vin0, vin1), zeros), vfalse, vtrue);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = (float16_t)((bool)(input0[index]) | (bool)(input1[0]));
    }
  }
  return NNACL_OK;
}

int ElementSquaredDifferenceFp16(const float16_t *input0, const float16_t *input1, float16_t *output,
                                 int element_size) {
  ElementSubFp16(input0, input1, output, element_size);
  return ElementMulFp16(output, output, output, element_size);
}

int ElementOptSquaredDifferenceFp16(const float16_t *input0, const float16_t *input1, float16_t *output,
                                    int element_size, ArithmeticParameter *param) {
  ElementOptSubFp16(input0, input1, output, element_size, param);
  return ElementMulFp16(output, output, output, element_size);
}

int ElementMaximumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vmaxq_f16(vin0, vin1);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMAX(input0[index], input1[index]);
  }
  return NNACL_OK;
}

int ElementOptMaximumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vmaxq_f16(vin0_opt, vin1);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[0], input1[index]);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vmaxq_f16(vin0, vin1_opt);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[index], input1[0]);
    }
  }
  return NNACL_OK;
}

int ElementMinimumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    float16x8_t vout = vminq_f16(vin0, vin1);
    vst1q_f16(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMIN(input0[index], input1[index]);
  }
  return NNACL_OK;
}

int ElementOptMinimumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      float16x8_t vout = vminq_f16(vin0_opt, vin1);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(input0[0], input1[index]);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      float16x8_t vout = vminq_f16(vin0, vin1_opt);
      vst1q_f16(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(input0[index], input1[0]);
    }
  }
  return NNACL_OK;
}

int ElementNotEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    uint8x8_t vout = vmovn_u16(vceqq_f16(vin0, vin1));
    vst1_u8(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] != input1[index];
  }
  return NNACL_OK;
}

int ElementOptNotEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                           ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      uint8x8_t vout = vmovn_u16(vceqq_f16(vin0_opt, vin1));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] != input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      uint8x8_t vout = vmovn_u16(vceqq_f16(vin0, vin1_opt));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] != input1[0];
    }
  }
  return NNACL_OK;
}

int ElementEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    uint8x8_t vout = vmovn_u16(vceqq_f16(vin0, vin1));
    vst1_u8(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] == input1[index];
  }
  return NNACL_OK;
}

int ElementOptEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                        ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      uint8x8_t vout = vmovn_u16(vceqq_f16(vin0_opt, vin1));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] == input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      uint8x8_t vout = vmovn_u16(vceqq_f16(vin0, vin1_opt));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] == input1[0];
    }
  }
  return NNACL_OK;
}

int ElementLessFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    uint8x8_t vout = vmovn_u16(vcltq_f16(vin0, vin1));
    vst1_u8(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] < input1[index];
  }
  return NNACL_OK;
}

int ElementOptLessFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                       ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      uint8x8_t vout = vmovn_u16(vcltq_f16(vin0_opt, vin1));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] < input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      uint8x8_t vout = vmovn_u16(vcltq_f16(vin0, vin1_opt));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] < input1[0];
    }
  }
  return NNACL_OK;
}

int ElementLessEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    uint8x8_t vout = vmovn_u16(vcleq_f16(vin0, vin1));
    vst1_u8(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] <= input1[index];
  }
  return NNACL_OK;
}

int ElementOptLessEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                            ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      uint8x8_t vout = vmovn_u16(vcleq_f16(vin0_opt, vin1));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] <= input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      uint8x8_t vout = vmovn_u16(vcleq_f16(vin0, vin1_opt));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] <= input1[0];
    }
  }
  return NNACL_OK;
}

int ElementGreaterFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    uint8x8_t vout = vmovn_u16(vcgtq_f16(vin0, vin1));
    vst1_u8(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] > input1[index];
  }
  return NNACL_OK;
}

int ElementOptGreaterFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                          ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      uint8x8_t vout = vmovn_u16(vcgtq_f16(vin0_opt, vin1));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] > input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      uint8x8_t vout = vmovn_u16(vcgtq_f16(vin0, vin1_opt));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] > input1[0];
    }
  }
  return NNACL_OK;
}

int ElementGreaterEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 8; index += C8NUM) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vin1 = vld1q_f16(input1 + index);
    uint8x8_t vout = vmovn_u16(vcgeq_f16(vin0, vin1));
    vst1_u8(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] >= input1[index];
  }
  return NNACL_OK;
}

int ElementOptGreaterEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                               ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float16x8_t vin0_opt = vdupq_n_f16(input0[0]);
  float16x8_t vin1_opt = vdupq_n_f16(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin1 = vld1q_f16(input1 + index);
      uint8x8_t vout = vmovn_u16(vcgeq_f16(vin0_opt, vin1));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] >= input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 8; index += C8NUM) {
      float16x8_t vin0 = vld1q_f16(input0 + index);
      uint8x8_t vout = vmovn_u16(vcgeq_f16(vin0, vin1_opt));
      vst1_u8(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] >= input1[0];
    }
  }
  return NNACL_OK;
}
