/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/arithmetic_compare_fp32.h"

inline bool EqualFp32(float x, float y);
inline bool EqualBool(bool x, bool y);
inline bool NotEqualFp32(float x, float y);
inline bool LessFp32(float x, float y);
inline bool LessEqualFp32(float x, float y);
inline bool GreaterFp32(float x, float y);
inline bool GreaterEqualFp32(float x, float y);

inline bool EqualInt32(int x, int y);
inline bool NotEqualInt32(int x, int y);
inline bool NotEqualInt64(int64_t x, int64_t y);
inline bool LessInt32(int x, int y);
inline bool LessEqualInt32(int x, int y);
inline bool GreaterInt32(int x, int y);
inline bool GreaterEqualInt32(int x, int y);

bool EqualFp32(float x, float y) { return x == y; }
bool EqualBool(bool x, bool y) { return x == y; }
bool NotEqualFp32(float x, float y) { return x != y; }
bool LessFp32(float x, float y) { return x < y; }
bool LessEqualFp32(float x, float y) { return x <= y; }
bool GreaterFp32(float x, float y) { return x > y; }
bool GreaterEqualFp32(float x, float y) { return x >= y; }

bool EqualInt32(int x, int y) { return x == y; }
bool NotEqualInt32(int x, int y) { return x != y; }
bool NotEqualInt64(int64_t x, int64_t y) { return x != y; }
bool LessInt32(int x, int y) { return x < y; }
bool LessEqualInt32(int x, int y) { return x <= y; }
bool GreaterInt32(int x, int y) { return x > y; }
bool GreaterEqualInt32(int x, int y) { return x >= y; }

#define ELEMENT_COMPARE(input0, input1, output, element_size, compare_func) \
  do {                                                                      \
    for (int i = 0; i < element_size; i++) {                                \
      output[i] = compare_func(input0[i], input1[i]);                       \
    }                                                                       \
    return NNACL_OK;                                                        \
  } while (0)

#define ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, compare_func) \
  do {                                                                                        \
    int i = 0;                                                                                \
    if (first_scalar) {                                                                       \
      for (; i < element_size; i++) {                                                         \
        output[i] = compare_func(input0[0], input1[i]);                                       \
      }                                                                                       \
    } else {                                                                                  \
      for (; i < element_size; i++) {                                                         \
        output[i] = compare_func(input0[i], input1[0]);                                       \
      }                                                                                       \
    }                                                                                         \
    return NNACL_OK;                                                                          \
  } while (0)

// equal:
int ElementEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, EqualFp32);
}

int ElementEqualBool(const bool *input0, const bool *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, EqualBool);
}

int ElementOptEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                        bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, EqualFp32);
}

int ElementEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, EqualInt32);
}

int ElementOptEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                         bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, EqualInt32);
}

// not equal
int ElementNotEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, NotEqualFp32);
}

int ElementOptNotEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                           bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, NotEqualFp32);
}

int ElementNotEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, NotEqualInt32);
}

int ElementOptNotEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                            bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, NotEqualInt32);
}

int ElementNotEqualInt64(const int64_t *input0, const int64_t *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, NotEqualInt64);
}

int ElementOptNotEqualInt64(const int64_t *input0, const int64_t *input1, uint8_t *output, int element_size,
                            bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, NotEqualInt64);
}

// less
int ElementLessFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, LessFp32);
}

int ElementOptLessFp32(const float *input0, const float *input1, uint8_t *output, int element_size, bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, LessFp32);
}

int ElementLessInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, LessInt32);
}

int ElementOptLessInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                        bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, LessInt32);
}

// less equal
int ElementLessEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, LessEqualFp32);
}

int ElementOptLessEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                            bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, LessEqualFp32);
}

int ElementLessEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, LessEqualInt32);
}

int ElementOptLessEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                             bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, LessEqualInt32);
}

// greater
int ElementGreaterFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, GreaterFp32);
}

int ElementOptGreaterFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                          bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, GreaterFp32);
}

int ElementGreaterInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, GreaterInt32);
}

int ElementOptGreaterInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                           bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, GreaterInt32);
}

// greater equal
int ElementGreaterEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, GreaterEqualFp32);
}

int ElementOptGreaterEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size,
                               bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, GreaterEqualFp32);
}

int ElementGreaterEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  ELEMENT_COMPARE(input0, input1, output, element_size, GreaterEqualInt32);
}

int ElementOptGreaterEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size,
                                bool first_scalar) {
  ELEMENT_COMPARE_OPT(input0, input1, output, element_size, first_scalar, GreaterEqualInt32);
}
