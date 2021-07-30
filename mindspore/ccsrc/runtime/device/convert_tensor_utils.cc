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
#include "runtime/device/convert_tensor_utils.h"
#include <vector>
namespace mindspore {
namespace device {
void HalfToFloat(void *dst, const void *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  auto half_data = static_cast<const float16 *>(src);
  auto float_data = static_cast<float *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    float tmp = half_to_float(half_data[i]);
    float_data[i] = tmp;
  }
}

void FloatToHalf(void *dst, const void *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  auto float_data = static_cast<const float *>(src);
  auto half_data = static_cast<float16 *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    half_data[i] = float16(float_data[i]);
  }
}

void DoubleToFloat(void *dst, const void *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  auto double_data = static_cast<const double *>(src);
  auto float_data = static_cast<float *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    float_data[i] = static_cast<float>(double_data[i]);
  }
}

void FloatToDouble(void *dst, const void *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  auto float_data = static_cast<const float *>(src);
  auto double_data = static_cast<double *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    double_data[i] = static_cast<double>(float_data[i]);
  }
}

void ShortToInt(void *dst, const void *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  auto half_data = static_cast<const int16_t *>(src);
  auto int_data = static_cast<int *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    int_data[i] = static_cast<int>(half_data[i]);
  }
}

void IntToShort(void *dst, const void *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  auto int_data = static_cast<const int *>(src);
  auto half_data = static_cast<int16_t *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    half_data[i] = static_cast<int16_t>(int_data[i]);
  }
}

void LongToInt(void *dst, const void *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  auto long_data = static_cast<const int64_t *>(src);
  auto int_data = static_cast<int *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    int_data[i] = static_cast<int>(long_data[i]);
  }
}

void IntToLong(void *dst, const void *src, size_t elem_num) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  auto int_data = static_cast<const int *>(src);
  auto long_data = static_cast<int64_t *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    long_data[i] = static_cast<int64_t>(int_data[i]);
  }
}

void ConvertSameType(void *const dst, const void *src, size_t size, TypeId type) {
  if (dst == nullptr || src == nullptr) {
    return;
  }
  if (type == kNumberTypeFloat16) {
    auto dst_data = static_cast<float16 *>(dst);
    auto src_data = static_cast<const float16 *>(src);
    ConvertSameType(dst_data, src_data, size >> 1);
  } else if (type == kNumberTypeFloat32) {
    auto dst_data = static_cast<float *>(dst);
    auto src_data = static_cast<const float *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(float));
  } else if (type == kNumberTypeFloat64) {
    auto dst_data = static_cast<double *>(dst);
    auto src_data = static_cast<const double *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(double));
  } else if (type == kNumberTypeInt16) {
    auto dst_data = static_cast<int16_t *>(dst);
    auto src_data = static_cast<const int16_t *>(src);
    ConvertSameType(dst_data, src_data, size >> 1);
  } else if (type == kNumberTypeInt32) {
    auto dst_data = static_cast<int *>(dst);
    auto src_data = static_cast<const int *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(int));
  } else if (type == kNumberTypeInt64) {
    auto dst_data = static_cast<int64_t *>(dst);
    auto src_data = static_cast<const int64_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(int64_t));
  } else {
    MS_LOG(EXCEPTION) << "Invalid Type: " << TypeIdLabel(type);
  }
}
}  // namespace device
}  // namespace mindspore
