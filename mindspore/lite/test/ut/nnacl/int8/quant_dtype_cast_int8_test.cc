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
#include <iostream>
#include <cmath>
#include "common/common_test.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"
#ifdef ENABLE_ARM64
#include "nnacl/fp16/quant_dtype_cast_fp16.h"
#endif

namespace mindspore {

class QuantCastInt8Test : public mindspore::CommonTest {
 public:
  QuantCastInt8Test() {}
};

void Int8ToFp32Util(const int8_t *quant_values, float *real_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return;
  }

  for (int i = 0; i < size; i++) {
    real_values[i] = (quant_values[i] - zp) * scale;
  }
  return;
}

void Fp32ToInt8Util(const float *real_values, int8_t *quant_values, float scale, int32_t zp, int size,
                    int32_t min_value, int32_t max_value) {
  if (quant_values == NULL || real_values == NULL) {
    return;
  }
  const float inverse_scale = 1.0f / scale;
  for (int i = 0; i < size; ++i) {
    if (real_values[i] == INFINITY) {
      quant_values[i] = max_value;
    } else if (real_values[i] == -INFINITY) {
      quant_values[i] = min_value;
    } else {
      int temp = round(real_values[i] * inverse_scale + zp);
      temp = temp < max_value ? temp : max_value;
      temp = temp > min_value ? temp : min_value;
      quant_values[i] = (int8_t)temp;
    }
  }
  return;
}

#ifdef ENABLE_ARM64
void Fp16ToInt8Util(const float16_t *real_values, int8_t *quant_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return;
  }
  for (int i = 0; i < size; ++i) {
    if (real_values[i] == INFINITY) {
      quant_values[i] = INT8_MAX;
      continue;
    }
    if (real_values[i] == -INFINITY) {
      quant_values[i] = INT8_MIN;
      continue;
    }
    float temp = round(static_cast<float>(real_values[i]) / scale + zp);
    if (temp > INT8_MAX) {
      quant_values[i] = INT8_MAX;
    } else if (temp < INT8_MIN) {
      quant_values[i] = INT8_MIN;
    } else {
      quant_values[i] = (int8_t)temp;
    }
  }
  return;
}

void ConstructFp16Int8Data(float16_t *real_values, int8_t *benchmark_data, int kSize, int32_t zp, float scale) {
  constexpr int kDiv = 2;
  for (int i = 0; i < kSize; ++i) {
    real_values[i] = static_cast<float16_t>(i - kSize / kDiv);
  }
  Fp16ToInt8Util(real_values, benchmark_data, scale, zp, kSize);
}

TEST_F(QuantCastInt8Test, Fp16Int8Size3) {
  constexpr int kSize = 8;
  float16_t real_values[kSize] = {-INFINITY, INFINITY, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5};
  int32_t zp = -1;
  float scale = 0.3f;

  int8_t benchmark_data[kSize];
  Fp16ToInt8Util(real_values, benchmark_data, scale, zp, kSize);
  int8_t quant_values[kSize];
  (void)DoQuantizeFp16ToInt8(real_values, quant_values, scale, zp, kSize);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(quant_values[i], benchmark_data[i]);
  }
}

TEST_F(QuantCastInt8Test, Fp16Int8Size32) {
  constexpr int kSize = 32;
  int32_t zp = 0;
  float scale = 0.3f;
  float16_t real_values[kSize];
  int8_t benchmark_data[kSize];
  ConstructFp16Int8Data(real_values, benchmark_data, kSize, zp, scale);

  int8_t quant_values[kSize];
  (void)DoQuantizeFp16ToInt8(real_values, quant_values, scale, zp, kSize);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(quant_values[i], benchmark_data[i]);
  }
}

TEST_F(QuantCastInt8Test, Fp16Int8Size33) {
  constexpr int kSize = 33;
  float16_t real_values[kSize];
  int32_t zp = 2;
  float scale = 0.1f;
  int8_t benchmark_data[kSize];
  ConstructFp16Int8Data(real_values, benchmark_data, kSize, zp, scale);

  int8_t quant_values[kSize];
  (void)DoQuantizeFp16ToInt8(real_values, quant_values, scale, zp, kSize);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(quant_values[i], benchmark_data[i]);
  }
}

void Int8ToFp16Util(const int8_t *quant_values, float16_t *real_values, float scale, int32_t zp, int size) {
  if (quant_values == NULL || real_values == NULL) {
    return;
  }
  for (int i = 0; i < size; ++i) {
    real_values[i] = (quant_values[i] - zp) * scale;
  }
  return;
}

void ConstructInt8ToFp16Data(int8_t *quant_values, float16_t *benchmark_data, int kSize, int32_t zp, float scale) {
  constexpr int kDiv = 2;
  for (int i = 0; i < kSize; ++i) {
    quant_values[i] = (i - kSize / kDiv);
  }
  Int8ToFp16Util(quant_values, benchmark_data, scale, zp, kSize);
}

TEST_F(QuantCastInt8Test, Int8Fp16Size6) {
  constexpr int kSize = 6;
  int8_t quant_values[kSize];
  int32_t zp = 2;
  float scale = 0.3f;
  float16_t benchmark_data[kSize];
  ConstructInt8ToFp16Data(quant_values, benchmark_data, kSize, zp, scale);

  float16_t real_values[kSize];
  DoDequantizeInt8ToFp16(quant_values, real_values, scale, zp, kSize);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_LE(real_values[i] - benchmark_data[i], float16_t(1e-5));
  }
}

TEST_F(QuantCastInt8Test, Int8Fp16Size16) {
  constexpr int kSize = 16;
  int8_t quant_values[kSize];
  int32_t zp = 2;
  float scale = 0.3f;
  float16_t benchmark_data[kSize];
  ConstructInt8ToFp16Data(quant_values, benchmark_data, kSize, zp, scale);

  float16_t real_values[kSize];
  DoDequantizeInt8ToFp16(quant_values, real_values, scale, zp, kSize);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_LE(real_values[i] - benchmark_data[i], float16_t(1e-5));
  }
}

TEST_F(QuantCastInt8Test, Int8Fp16Size18) {
  constexpr int kSize = 18;
  int8_t quant_values[kSize];
  int32_t zp = 2;
  float scale = 0.3f;
  float16_t benchmark_data[kSize];
  ConstructInt8ToFp16Data(quant_values, benchmark_data, kSize, zp, scale);

  float16_t real_values[kSize];
  DoDequantizeInt8ToFp16(quant_values, real_values, scale, zp, kSize);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_LE(real_values[i] - benchmark_data[i], float16_t(1e-5));
  }
}
#endif

TEST_F(QuantCastInt8Test, Int8Fp32Size8) {
  constexpr int kSize = 8;
  int8_t quant_values[kSize] = {-128, 1, 1, 2, 3, 5, 8, 13};
  int32_t zp = 1;
  float scale = 0.5f;
  float benchmark_data[kSize];
  Int8ToFp32Util(quant_values, benchmark_data, scale, zp, kSize);
  float real_value[kSize];
  DoDequantizeInt8ToFp32(quant_values, real_value, scale, zp, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(real_value[i] - benchmark_data[i]), 1e-6);
  }
}
TEST_F(QuantCastInt8Test, Int8Fp32Size16) {
  constexpr int kSize = 16;
  int8_t quant_values[kSize] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
  int32_t zp = 1;
  float scale = 0.5f;
  float benchmark_data[kSize];
  Int8ToFp32Util(quant_values, benchmark_data, scale, zp, kSize);
  float real_value[kSize];
  DoDequantizeInt8ToFp32(quant_values, real_value, scale, zp, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(real_value[i] - benchmark_data[i]), 1e-6);
  }
}

TEST_F(QuantCastInt8Test, Int8Fp32Size17) {
  constexpr int kSize = 17;
  int8_t quant_values[kSize] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33};
  int32_t zp = 1;
  float scale = 0.5f;
  float benchmark_data[kSize];
  Int8ToFp32Util(quant_values, benchmark_data, scale, zp, kSize);
  float real_value[kSize];
  DoDequantizeInt8ToFp32(quant_values, real_value, scale, zp, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(real_value[i] - benchmark_data[i]), 1e-6);
  }
}

TEST_F(QuantCastInt8Test, Int8Fp32NegZp) {
  constexpr int kSize = 17;
  int8_t quant_values[kSize] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 127};
  int32_t zp = -1;
  float scale = 0.5f;
  float benchmark_data[kSize];
  Int8ToFp32Util(quant_values, benchmark_data, scale, zp, kSize);
  float real_value[kSize];
  DoDequantizeInt8ToFp32(quant_values, real_value, scale, zp, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(real_value[i] - benchmark_data[i]), 1e-6);
  }
}

TEST_F(QuantCastInt8Test, Fp32Int8Size8) {
  constexpr int kSize = 8;
  float real_values[kSize] = {-0.5f, 0.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.5f, 6.0f};
  int32_t zp = 5;
  float scale = 0.3f;
  int32_t min_value = -128;
  int32_t max_value = 127;
  int8_t benchmark_data[kSize];
  Fp32ToInt8Util(real_values, benchmark_data, scale, zp, kSize, min_value, max_value);
  int8_t quant_values[kSize];
  DoQuantizeFp32ToInt8(real_values, quant_values, scale, zp, kSize, min_value, max_value);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(quant_values[i], benchmark_data[i]);
  }
}
// size 8 neg zp
TEST_F(QuantCastInt8Test, Fp32Int8NegZp) {
  constexpr int kSize = 8;
  float real_values[kSize] = {-0.5f, 0.0f, 0.0f, 0.5f, 1.0f, 2.0f, 3.5f, 6.0f};
  int32_t zp = -1;
  float scale = 0.3f;
  int32_t min_value = -128;
  int32_t max_value = 127;
  int8_t benchmark_data[kSize];
  Fp32ToInt8Util(real_values, benchmark_data, scale, zp, kSize, min_value, max_value);
  int8_t quant_values[kSize];
  DoQuantizeFp32ToInt8(real_values, quant_values, scale, zp, kSize, min_value, max_value);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(quant_values[i], benchmark_data[i]);
  }
}

// size 16
TEST_F(QuantCastInt8Test, Fp32Int8Size16) {
  constexpr int kSize = 16;
  float real_values[kSize] = {-0.25f, -0.5f, 0.0f, 0.0f, 0.25f, 0.5f,  1.0f,  2.0f,
                              3.5f,   6.0f,  7.0f, 8.0f, 9.0f,  10.0f, 11.0f, 12.0f};
  int32_t zp = -1;
  float scale = 0.5f;
  int32_t min_value = -128;
  int32_t max_value = 127;
  int8_t benchmark_data[kSize];
  Fp32ToInt8Util(real_values, benchmark_data, scale, zp, kSize, min_value, max_value);

  int8_t quant_values[kSize];
  DoQuantizeFp32ToInt8(real_values, quant_values, scale, zp, kSize, min_value, max_value);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(quant_values[i], benchmark_data[i]);
  }
}
// size 17
TEST_F(QuantCastInt8Test, Fp32Int8Size17) {
  constexpr int kSize = 17;
  float real_values[kSize] = {-0.25f, -0.5f, 0.0f, 0.0f, 0.25f, 0.5f,  1.0f,  2.0f, 3.5f,
                              6.0f,   7.0f,  8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f};
  int32_t zp = -1;
  float scale = 0.5f;
  int32_t min_value = -128;
  int32_t max_value = 127;
  int8_t benchmark_data[kSize];
  Fp32ToInt8Util(real_values, benchmark_data, scale, zp, kSize, min_value, max_value);

  int8_t quant_values[kSize];
  DoQuantizeFp32ToInt8(real_values, quant_values, scale, zp, kSize, min_value, max_value);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(quant_values[i], benchmark_data[i]);
  }
}
// size 8 inf
TEST_F(QuantCastInt8Test, Fp32Int8Inf) {
  constexpr int kSize = 8;
  float real_values[kSize] = {-0.5f, 0.0f, 0.0f, 0.5f, 1.0f, 2.0f, 1.0f / 0.0f, -1.0f / 0.0f};
  int32_t zp = -1;
  float scale = 0.3f;
  int32_t min_value = -128;
  int32_t max_value = 127;
  int8_t benchmark_data[kSize];
  Fp32ToInt8Util(real_values, benchmark_data, scale, zp, kSize, min_value, max_value);

  int8_t quant_values[kSize];
  DoQuantizeFp32ToInt8(real_values, quant_values, scale, zp, kSize, min_value, max_value);

  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(quant_values[i], benchmark_data[i]);
  }
}

}  // namespace mindspore
