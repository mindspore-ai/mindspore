/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "nnacl/op_base.h"
#include "nnacl/base/cast_base.h"
#include "nnacl/kernel/cast.h"

namespace mindspore {
class CastTest : public mindspore::CommonTest {
 public:
  CastTest() {}
};

TEST_F(CastTest, CreateCastSucc) {
  OpParameter cast_op_parameter;
  cast_op_parameter.type_ = PrimType_Cast;
  cast_op_parameter.thread_num_ = 4;
  KernelBase *cast_kernel = CreateCast(&cast_op_parameter, kNumberTypeFloat32);
  ASSERT_NE(cast_kernel, nullptr);
}

TEST_F(CastTest, CreateCastSucc2) {
  OpParameter cast_op_parameter;
  cast_op_parameter.type_ = PrimType_Cast;
  cast_op_parameter.thread_num_ = 1000;
  constexpr int invalid_data_type = 1000;
  KernelBase *cast_kernel = CreateCast(&cast_op_parameter, invalid_data_type);
  ASSERT_NE(cast_kernel, nullptr);
}

TEST_F(CastTest, BoolToFloat32Test) {
  constexpr int kSize = 8;
  bool src_values[kSize] = {false, false, true, true, false, true, false, true};
  float dst_values[kSize] = {0f, 0f, 1f, 1f, 0f, 1f, 0f, 1f};
  float dst[kSize];
  memset(dst, 0, kSize * sizeof(float));
  BoolToFloat32(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Uint8ToFloat32Test) {
  constexpr int kSize = 8;
  uint8_t src_values[kSize] = {1, 2, 10, 20, 30, 19, 223, 111};
  float dst_values[kSize] = {1f, 2f, 10f, 20f, 30f, 19f, 223f, 111f};
  float dst[kSize];
  memset(dst, 0, kSize * sizeof(float));
  Uint8ToFloat32(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Int32ToFloat32Test) {
  constexpr int kSize = 8;
  int32_t src_values[kSize] = {1, 2, 10, 20, 30, 65535, 223, 111};
  float dst_values[kSize] = {1f, 2f, 10f, 20f, 30f, 65535f, 223f, 111f};
  float dst[kSize];
  memset(dst, 0, kSize * sizeof(float));
  Int32ToFloat32(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Int64ToFloat32Test) {
  constexpr int kSize = 8;
  int64_t src_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  float dst_values[kSize] = {1f, -20f, 10f, 20f, -3000f, 65535f, 223f, 111f};
  float dst[kSize];
  memset(dst, 0, kSize * sizeof(float));
  Int64ToFloat32(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Float32ToInt64Test) {
  constexpr int kSize = 8;
  float src_values[kSize] = {1f, -20f, 10f, 20f, -3000f, 65535f, 223f, 111f};
  int64_t dst_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  int64_t dst[kSize];
  memset(dst, 0, kSize * sizeof(int64_t));
  Float32ToInt64(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Float32ToInt64Test) {
  constexpr int kSize = 8;
  float src_values[kSize] = {1f, -20f, 10f, 20f, -3000f, 65535f, 223f, 111f};
  int64_t dst_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  int64_t dst[kSize];
  memset(dst, 0, kSize * sizeof(int64_t));
  Float32ToInt64(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Float32ToInt32Test) {
  constexpr int kSize = 8;
  float src_values[kSize] = {1f, -20f, 10f, 20.5f, -3000f, 65535f, 223.9f, 111f};
  int32_t dst_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  int32_t dst[kSize];
  memset(dst, 0, kSize * sizeof(int32_t));
  Float32ToInt32(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Int32ToInt64Test) {
  constexpr int kSize = 8;
  int32_t src_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  int64_t dst_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  int64_t dst[kSize];
  memset(dst, 0, kSize * sizeof(int64_t));
  Int32ToInt64(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Int64ToInt32Test) {
  constexpr int kSize = 8;
  int64_t src_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  int32_t dst_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  int32_t dst[kSize];
  memset(dst, 0, kSize * sizeof(int32_t));
  Int64ToInt32(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Float32ToInt16Test) {
  constexpr int kSize = 8;
  float src_values[kSize] = {1.9f, -20f, 10.3f, 20f, -3000.1f, 65535f, 223f, 111f};
  int16_t dst_values[kSize] = {1, -20, 10, 20, -3000, 65535, 223, 111};
  int16_t dst[kSize];
  memset(dst, 0, kSize * sizeof(int16_t));
  Float32ToInt16(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, BoolToInt32Test) {
  constexpr int kSize = 8;
  bool src_values[kSize] = {false, false, true, true, false, true, false, true};
  int32_t dst_values[kSize] = {0, 0, 1, 1, 0, 1, 0, 1};
  int32_t dst[kSize];
  memset(dst, 0, kSize * sizeof(int32_t));
  BoolToInt32(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Float32ToBoolTest) {
  constexpr int kSize = 8;
  float src_values[kSize] = {0f, -20f, 0f, 20f, 0f, 65535f, 223f, 111f};
  bool dst_values[kSize] = {false, true, false, true, false, true, true, true};
  bool dst[kSize];
  memset(dst, 0, kSize * sizeof(bool));
  Float32ToBool(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Float32ToUint8Test) {
  constexpr int kSize = 8;
  float src_values[kSize] = {0f, 1f, 10f, 100f, 127f, 11f, 22f, 33f};
  uint8_t dst_values[kSize] = {0, 1, 10, 100, 127, 11, 22, 33};
  uint8_t dst[kSize];
  memset(dst, 0, kSize * sizeof(uint8_t));
  Float32ToUint8(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

#ifdef ENABLE_FP16
TEST_F(CastTest, Uint16ToFloat32Test) {
  constexpr int kSize = 8;
  uint16_t src_values[kSize] = {0, 1, 10, 100, 127, 65535, 1000, 2000};
  float dst_values[kSize] = {0f, 1f, 10f, 100f, 127f, 65535f, 1000f, 2000f};
  float dst[kSize];
  memset(dst, 0, kSize * sizeof(float));
  Fp16ToFloat32(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Float32ToUint16Test) {
  constexpr int kSize = 8;
  float src_values[kSize] = {0f, 1f, 10f, 100f, 127f, 65535f, 1000f, 2000f};
  uint16_t dst_values[kSize] = {0, 1, 10, 100, 127, 65535, 1000, 2000};
  uint16_t dst[kSize];
  memset(dst, 0, kSize * sizeof(uint16_t));
  Float32ToFp16(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Int64ToFp16Test) {
  constexpr int kSize = 8;
  int64_t src_values[kSize] = {0, 1, 10, 100, 127, 65535, 1000, 2000};
  uint16_t dst_values[kSize] = {0, 1, 10, 100, 127, 65535, 1000, 2000};
  uint16_t dst[kSize];
  memset(dst, 0, kSize * sizeof(uint16_t));
  Float32ToFp16(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}

TEST_F(CastTest, Int32ToFp16Test) {
  constexpr int kSize = 8;
  int32_t src_values[kSize] = {0, 1, 10, 100, 127, 65535, 1000, 2000};
  uint16_t dst_values[kSize] = {0, 1, 10, 100, 127, 65535, 1000, 2000};
  uint16_t dst[kSize];
  memset(dst, 0, kSize * sizeof(uint16_t));
  Int32ToFp16(src_values, dst, kSize);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_LT(std::abs(dst[i] - dst_values[i]), 1e-6);
  }
}
#endif
}  // namespace mindspore
