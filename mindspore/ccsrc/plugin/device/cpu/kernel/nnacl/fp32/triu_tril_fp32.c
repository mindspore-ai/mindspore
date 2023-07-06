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

int TriuTrilGetCalculateNum(KernelBase *self, int64_t *mul, int64_t *height, int64_t *width) {
  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  for (size_t i = 0; i < input_tensor->shape_size_; i++) {
    if (input_tensor->shape_[i] <= 0) {
      return NNACL_TRIU_TRIL_INPUT_SHAPE_ERROR;
    }
  }

  size_t input_hw_dims = Num2;
  NNACL_CHECK_FALSE(input_tensor->shape_size_ < DIMENSION_2D, NNACL_TRIU_INPUT_DIMS_INVALID);

  *mul = 1;
  for (size_t i = 0; i < input_tensor->shape_size_ - input_hw_dims; i++) {
    *mul *= input_tensor->shape_[i];
  }
  *height = input_tensor->shape_[input_tensor->shape_size_ - Num2];
  *width = input_tensor->shape_[input_tensor->shape_size_ - Num1];

  return NNACL_OK;
}

int TriuTrilGetKValue(KernelBase *self, int64_t *k) {
  if (self->in_size_ <= 1) {
    *k = 0;
    return NNACL_OK;
  }

  TensorC *k_tensor = self->in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(k_tensor);
  NNACL_CHECK_NULL_RETURN_ERR(k_tensor->data_);

  switch (k_tensor->data_type_) {
    case kNumberTypeInt:
    case kNumberTypeInt32:
      *k = *((int32_t *)k_tensor->data_);
      break;
    case kNumberTypeInt64:
      *k = *((int64_t *)k_tensor->data_);
      break;
    default:
      return NNACL_TRIU_K_TENSOR_DATA_TYPE_INVALID;
  }
  return NNACL_OK;
}

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
