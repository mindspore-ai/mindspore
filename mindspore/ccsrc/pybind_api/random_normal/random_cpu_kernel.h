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
#ifndef PYBIND_API_API_IR_RANDOM_NORMAL_RANDOM_CPU_KERNEL_H_
#define PYBIND_API_API_IR_RANDOM_NORMAL_RANDOM_CPU_KERNEL_H_
#include <vector>
#include "include/common/utils/philox_generator.h"
#include "pybind11/pybind11.h"
#include "include/common/pybind_api/api_register.h"
#include "utils/log_adapter.h"

namespace py = pybind11;

namespace mindspore {
template <class T, typename vartype>
class NormalDistribution;
template <class T>
class NormalDistribution<T, float> {
 public:
  bool UInt32ToFloat32(uint32_t input, float *output) {
    const uint32_t temp_value = input & 0x7fffffu;
    const uint32_t exp = static_cast<uint32_t>(127);
    const uint32_t val = (exp << 23) | temp_value;
    errno_t mem_ret = memcpy_s(output, sizeof(float), &val, sizeof(uint32_t));
    if (mem_ret != EOK) {
      MS_LOG(ERROR) << "UInt32ToFloat32 memcpy is failed";
      return false;
    }
    *output = *output - 1.0f;
    return true;
  }

  std::array<float, kResultNum> operator()(T *generator) {
    std::array<uint32_t, 4> generate_value = (*generator)();
    const float PI = 3.14;
    for (uint32_t i = 0; i < kResultNum; i += 2) {
      float temp[2];
      UInt32ToFloat32(generate_value[i], &temp[0]);
      UInt32ToFloat32(generate_value[i + 1], &temp[1]);
      const float threshold = 1.0e-7f;
      temp[0] = temp[0] < threshold ? threshold : temp[0];
      temp[1] = temp[1] < threshold ? threshold : temp[1];
      result_[i] = sqrt(-2.0 * log(temp[0])) * sin(2 * PI * temp[1]);
      result_[i + 1] = sqrt(-2.0 * log(temp[0])) * cos(2 * PI * temp[1]);
    }
    return result_;
  }

 private:
  std::array<float, kResultNum> result_;
};

bool InitRandomNormal(std::vector<int64_t> out_shape, int64_t seed, int64_t seed2, const py::object &output_tensor);
}  // namespace mindspore

#endif  // PYBIND_API_API_IR_RANDOM_NORMAL_RANDOM_CPU_KERNEL_H_
