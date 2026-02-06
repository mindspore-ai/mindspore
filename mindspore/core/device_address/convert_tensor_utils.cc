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
#include "mindspore/core/include/device_address/convert_tensor_utils.h"
#include <complex>
#include <vector>
#include "base/float8_e4m3fn.h"
#include "base/float8_e5m2.h"
#include "base/hifloat8.h"
#include "utils/shape_utils.h"
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
  } else if (type == kNumberTypeBFloat16) {
    auto dst_data = static_cast<bfloat16 *>(dst);
    auto src_data = static_cast<const bfloat16 *>(src);
    ConvertSameType(dst_data, src_data, size >> 1);
  } else if (type == kNumberTypeHiFloat8) {
    auto dst_data = static_cast<hifloat8 *>(dst);
    auto src_data = static_cast<const hifloat8 *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(hifloat8));
  } else if (type == kNumberTypeFloat8E5M2) {
    auto dst_data = static_cast<float8_e5m2 *>(dst);
    auto src_data = static_cast<const float8_e5m2 *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(float8_e5m2));
  } else if (type == kNumberTypeFloat8E4M3FN) {
    auto dst_data = static_cast<float8_e4m3fn *>(dst);
    auto src_data = static_cast<const float8_e4m3fn *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(float8_e4m3fn));
  } else if (type == kNumberTypeInt8) {
    auto dst_data = static_cast<int8_t *>(dst);
    auto src_data = static_cast<const int8_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(int8_t));
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
  } else if (type == kNumberTypeBool) {
    auto dst_data = static_cast<bool *>(dst);
    auto src_data = static_cast<const bool *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(bool));
  } else if (type == kNumberTypeUInt8) {
    auto dst_data = static_cast<uint8_t *>(dst);
    auto src_data = static_cast<const uint8_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(uint8_t));
  } else if (type == kNumberTypeUInt16) {
    auto dst_data = static_cast<uint16_t *>(dst);
    auto src_data = static_cast<const uint16_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(uint16_t));
  } else if (type == kNumberTypeUInt32) {
    auto dst_data = static_cast<uint32_t *>(dst);
    auto src_data = static_cast<const uint32_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(uint32_t));
  } else if (type == kNumberTypeUInt64) {
    auto dst_data = static_cast<uint64_t *>(dst);
    auto src_data = static_cast<const uint64_t *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(uint64_t));
  } else if (type == kNumberTypeComplex64) {
    auto dst_data = static_cast<std::complex<float> *>(dst);
    auto src_data = static_cast<const std::complex<float> *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(std::complex<float>));
  } else if (type == kNumberTypeComplex128) {
    auto dst_data = static_cast<std::complex<double> *>(dst);
    auto src_data = static_cast<const std::complex<double> *>(src);
    ConvertSameType(dst_data, src_data, size / sizeof(std::complex<double>));
  } else {
    MS_LOG(EXCEPTION) << "Invalid Type: " << TypeIdLabel(type);
  }
}
}  // namespace device
}  // namespace mindspore
