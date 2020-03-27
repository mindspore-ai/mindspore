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
#include "device/convert_tensor_utils.h"
#include <vector>
namespace mindspore {
namespace device {
void HalfToFloat(void *dst, const void *src, size_t elem_num) {
  auto half_data = static_cast<const Eigen::half *>(src);
  auto float_data = static_cast<float *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    float tmp = Eigen::half_impl::half_to_float(half_data[i]);
    float_data[i] = tmp;
  }
}

void FloatToHalf(void *dst, const void *src, size_t elem_num) {
  auto float_data = static_cast<const float *>(src);
  auto half_data = static_cast<Eigen::half *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    half_data[i] = Eigen::half(float_data[i]);
  }
}

void DoubleToFloat(void *dst, const void *src, size_t elem_num) {
  auto double_data = static_cast<const double *>(src);
  auto float_data = static_cast<float *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    float_data[i] = static_cast<float>(double_data[i]);
  }
}

void FloatToDouble(void *dst, const void *src, size_t elem_num) {
  auto float_data = static_cast<const float *>(src);
  auto double_data = static_cast<double *>(dst);
  for (size_t i = 0; i < elem_num; ++i) {
    double_data[i] = static_cast<double>(float_data[i]);
  }
}
}  // namespace device
}  // namespace mindspore
