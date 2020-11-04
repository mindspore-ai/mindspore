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
#include "ps/random_normal/random_normal.h"
#include <iostream>
#include <thread>
#include <memory>
#include "utils/convert_utils_base.h"
#include "pybind_api/random_normal/random_cpu_kernel.h"

namespace mindspore {
namespace ps {
bool InitRandomNormal(float mean, float stddev, std::vector<size_t> out_shape, size_t global_seed, size_t op_seed,
                      float *output_data) {
  if (out_shape.size() == 0) {
    std::cout << "output data shape is error" << std::endl;
  }
  int64_t total_count = 1;
  for (uint32_t i = 0; i < out_shape.size(); i++) {
    total_count *= SizeToLong(out_shape[i]);
  }
  uint32_t thread_num = 16;
  if (total_count <= thread_num) {
    thread_num = 1;
  }
  float *start_ptr = output_data;
  if (start_ptr == nullptr) {
    std::cout << "start_ptr is nullptr" << std::endl;
    return false;
  }
  int64_t batchSize = total_count / thread_num;
  std::vector<std::thread> threads(thread_num);
  int64_t seed = SizeToLong(global_seed);
  int64_t seed2 = SizeToLong(op_seed);
  seed = (seed == 0 && seed2 == 0) ? clock() : seed;
  PhiloxGenerator generator = PhiloxGenerator(seed, seed2);
  if (thread_num != 1) {
    for (uint32_t i = 0; i < thread_num - 1; i++) {
      float *offset_ptr = start_ptr + batchSize * i;
      threads[i] =
        std::thread(FillRandoms<NormalDistribution<PhiloxGenerator, float>>, generator, offset_ptr, batchSize, i);
    }
    float *offset_ptr = start_ptr + batchSize * (thread_num - 1);
    threads[thread_num - 1] = std::thread(FillRandoms<NormalDistribution<PhiloxGenerator, float>>, generator,
                                          offset_ptr, total_count - (thread_num - 1) * batchSize, thread_num - 1);
  } else {
    threads[0] =
      std::thread(FillRandoms<NormalDistribution<PhiloxGenerator, float>>, generator, start_ptr, total_count, 0);
  }
  for (uint32_t i = 0; i < thread_num; i++) {
    threads[i].join();
  }
  for (int64_t i = 0; i < total_count; i++) {
    output_data[i] *= stddev;
  }
  return true;
}
}  // namespace ps
}  // namespace mindspore
