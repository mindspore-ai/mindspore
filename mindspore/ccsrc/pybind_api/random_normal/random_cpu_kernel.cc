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
#include "pybind_api/random_normal/random_cpu_kernel.h"

#include <thread>
#include <memory>
#include "runtime/device/cpu/cpu_device_address.h"
#include "ir/tensor.h"

namespace mindspore {
bool InitRandomNormal(std::vector<int64_t> out_shape, int64_t seed, int64_t seed2, const py::object &output_tensor) {
  if (out_shape.size() == 0) {
    std::cout << "output data shape is error" << std::endl;
  }
  int64_t total_count = 1;
  for (uint32_t i = 0; i < out_shape.size(); i++) {
    total_count *= out_shape[i];
  }
  uint32_t thread_num = 16;
  if (total_count <= thread_num) {
    thread_num = 1;
  }
  auto temp = py::cast<std::shared_ptr<mindspore::tensor::Tensor>>(output_tensor);
  float *start_ptr = reinterpret_cast<float *>(temp->data_c());
  if (start_ptr == nullptr) {
    std::cout << "start_ptr is nullptr" << std::endl;
    return false;
  }
  int64_t batchSize = total_count / thread_num;
  std::vector<std::thread> threads(thread_num);
  seed = (seed == 0 && seed2 == 0) ? clock() : seed;
  mindspore::PhiloxGenerator generator = mindspore::PhiloxGenerator(seed, seed2);
  float *offset_ptr = nullptr;
  if (thread_num != 1) {
    for (uint32_t i = 0; i < thread_num - 1; i++) {
      offset_ptr = start_ptr + batchSize * i;
      threads[i] = std::thread(mindspore::FillRandoms<mindspore::NormalDistribution<mindspore::PhiloxGenerator, float>>,
                               generator, offset_ptr, batchSize, i);
    }
    offset_ptr = start_ptr + batchSize * (thread_num - 1);
    threads[thread_num - 1] =
      std::thread(mindspore::FillRandoms<mindspore::NormalDistribution<mindspore::PhiloxGenerator, float>>, generator,
                  offset_ptr, total_count - (thread_num - 1) * batchSize, thread_num - 1);
  } else {
    threads[0] = std::thread(mindspore::FillRandoms<mindspore::NormalDistribution<mindspore::PhiloxGenerator, float>>,
                             generator, start_ptr, total_count, 0);
  }
  for (uint32_t i = 0; i < thread_num; i++) {
    threads[i].join();
  }
  return true;
}

REGISTER_PYBIND_DEFINE(random_normal,
                       ([](py::module *const m) { (void)m->def("random_normal", &InitRandomNormal, "testnormal"); }));
}  // namespace mindspore
