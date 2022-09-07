/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <cstdint>
#include <random>
#include "include/common/random.h"
#include "include/common/pybind_api/api_register.h"
#include "utils/log_adapter.h"

namespace mindspore::initializer {
//
// Generate float random numbers into a python buffer.
//
template <typename Generator, typename Distribution, typename... Args>
void GenerateFloatRandoms(std::uint64_t seed, const py::buffer &py_buf, Args... args) {
  // Check buffer info.
  py::buffer_info info = py_buf.request();
  if (info.format != py::format_descriptor<float>::format()) {
    MS_LOG(EXCEPTION) << "Unsupported data type '" << info.format << "'.";
  }
  // Get buffer pointer and size.
  if (info.size < 0) {
    MS_LOG(EXCEPTION) << "Negative buffer size: " << info.size << ".";
  }
  const size_t buf_size = static_cast<size_t>(info.size);
  float *buf = reinterpret_cast<float *>(info.ptr);
  MS_EXCEPTION_IF_NULL(buf);

  // Parallel generate randoms into buffer.
  random::GenerateRandomsParallel<float, Generator, Distribution>(seed, buf, buf_size, args...);
}

void RandomUniform(std::uint64_t seed, const py::buffer &py_buf, float a, float b) {
  using Generator = random::Philox;
  using Distribution = random::UniformDistribution<double>;
  GenerateFloatRandoms<Generator, Distribution>(seed, py_buf, a, b);
}

void RandomNormal(std::uint64_t seed, const py::buffer &py_buf, float mean, float sigma) {
  using Generator = random::Philox;
  using Distribution = random::NormalDistribution<double>;
  GenerateFloatRandoms<Generator, Distribution>(seed, py_buf, mean, sigma);
}

void TruncatedNormal(std::uint64_t seed, const py::buffer &py_buf, float a, float b, float mean, float sigma) {
  using Generator = random::Philox;
  using Distribution = random::TruncatedNormal<double>;
  GenerateFloatRandoms<Generator, Distribution>(seed, py_buf, a, b, mean, sigma);
}

void RegRandomNormal(py::module *m) {
  (void)m->def("_random_uniform", RandomUniform);
  (void)m->def("_random_normal", RandomNormal);
  (void)m->def("_truncated_normal", TruncatedNormal);
}
}  // namespace mindspore::initializer
