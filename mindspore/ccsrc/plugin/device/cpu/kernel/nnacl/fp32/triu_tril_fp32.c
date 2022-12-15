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
#include "nnacl/fp32/triu_tril_fp32.h"

void TriuByte8(const void *src, void *dst, int64_t k, int64_t height, int64_t width, int64_t out_elems) {
  const int64_t *src_data = (const int64_t *)src;
  int64_t *dst_data = (int64_t *)dst;
  for (int64_t m = 0; m < out_elems; m++) {
    int64_t m_factor = m * height * width;
    for (int64_t h = 0; h < height; h++) {
      int64_t h_factor = m_factor + h * width;
      for (int64_t w = 0; w < width; w++) {
        int64_t index = h_factor + w;
        dst_data[index] = h + k <= w ? src_data[index] : 0;
      }
    }
  }
}

void TriuByte4(const void *src, void *dst, int64_t k, int64_t height, int64_t width, int64_t out_elems) {
  const int32_t *src_data = (const int32_t *)src;
  int32_t *dst_data = (int32_t *)dst;
  for (int64_t m = 0; m < out_elems; m++) {
    int64_t m_factor = m * height * width;
    for (int64_t h = 0; h < height; h++) {
      int64_t h_factor = m_factor + h * width;
      for (int64_t w = 0; w < width; w++) {
        int64_t index = h_factor + w;
        dst_data[index] = h + k <= w ? src_data[index] : 0;
      }
    }
  }
}

void TriuByte2(const void *src, void *dst, int64_t k, int64_t height, int64_t width, int64_t out_elems) {
  const int16_t *src_data = (const int16_t *)src;
  int16_t *dst_data = (int16_t *)dst;
  for (int64_t m = 0; m < out_elems; m++) {
    int64_t m_factor = m * height * width;
    for (int64_t h = 0; h < height; h++) {
      int64_t h_factor = m_factor + h * width;
      for (int64_t w = 0; w < width; w++) {
        int64_t index = h_factor + w;
        dst_data[index] = h + k <= w ? src_data[index] : 0;
      }
    }
  }
}
void TriuByte1(const void *src, void *dst, int64_t k, int64_t height, int64_t width, int64_t out_elems) {
  const int8_t *src_data = (const int8_t *)src;
  int8_t *dst_data = (int8_t *)dst;
  for (int64_t m = 0; m < out_elems; m++) {
    int64_t m_factor = m * height * width;
    for (int64_t h = 0; h < height; h++) {
      int64_t h_factor = m_factor + h * width;
      for (int64_t w = 0; w < width; w++) {
        int64_t index = h_factor + w;
        dst_data[index] = h + k <= w ? src_data[index] : 0;
      }
    }
  }
}

void TrilByte8(const void *src, void *dst, int64_t k, int64_t height, int64_t width, int64_t out_elems) {
  const int64_t *src_data = (const int64_t *)src;
  int64_t *dst_data = (int64_t *)dst;
  for (int64_t m = 0; m < out_elems; m++) {
    int64_t m_factor = m * height * width;
    for (int64_t h = 0; h < height; h++) {
      int64_t h_factor = m_factor + h * width;
      for (int64_t w = 0; w < width; w++) {
        int64_t index = h_factor + w;
        dst_data[index] = h + k >= w ? src_data[index] : 0;
      }
    }
  }
}

void TrilByte4(const void *src, void *dst, int64_t k, int64_t height, int64_t width, int64_t out_elems) {
  const int32_t *src_data = (const int32_t *)src;
  int32_t *dst_data = (int32_t *)dst;
  for (int64_t m = 0; m < out_elems; m++) {
    int64_t m_factor = m * height * width;
    for (int64_t h = 0; h < height; h++) {
      int64_t h_factor = m_factor + h * width;
      for (int64_t w = 0; w < width; w++) {
        int64_t index = h_factor + w;
        dst_data[index] = h + k >= w ? src_data[index] : 0;
      }
    }
  }
}
void TrilByte2(const void *src, void *dst, int64_t k, int64_t height, int64_t width, int64_t out_elems) {
  const int16_t *src_data = (const int16_t *)src;
  int16_t *dst_data = (int16_t *)dst;
  for (int64_t m = 0; m < out_elems; m++) {
    int64_t m_factor = m * height * width;
    for (int64_t h = 0; h < height; h++) {
      int64_t h_factor = m_factor + h * width;
      for (int64_t w = 0; w < width; w++) {
        int64_t index = h_factor + w;
        dst_data[index] = h + k >= w ? src_data[index] : 0;
      }
    }
  }
}
void TrilByte1(const void *src, void *dst, int64_t k, int64_t height, int64_t width, int64_t out_elems) {
  const int8_t *src_data = (const int8_t *)src;
  int8_t *dst_data = (int8_t *)dst;
  for (int64_t m = 0; m < out_elems; m++) {
    int64_t m_factor = m * height * width;
    for (int64_t h = 0; h < height; h++) {
      int64_t h_factor = m_factor + h * width;
      for (int64_t w = 0; w < width; w++) {
        int64_t index = h_factor + w;
        dst_data[index] = h + k >= w ? src_data[index] : 0;
      }
    }
  }
}
