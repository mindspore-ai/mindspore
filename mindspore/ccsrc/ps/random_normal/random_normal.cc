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
#include <algorithm>
#include "utils/convert_utils_base.h"
#include "pybind_api/random_normal/random_cpu_kernel.h"

namespace mindspore {
namespace ps {
static const uint32_t kMaxThreadNum = 16;
static const uint32_t kCPUCoreNum = std::thread::hardware_concurrency();

namespace {
// Update standard deviation to parameter: stddev
void UpdateStandardDeviation(float stddev, size_t total_count, float *output) {
  MS_EXCEPTION_IF_NULL(output);

  auto update_stddev_task = [](float stddev, size_t task_len, float *data) {
    for (size_t i = 0; i < task_len; i++) {
      data[i] *= stddev;
    }
  };

  uint32_t thread_num = std::max(kMaxThreadNum, kCPUCoreNum);
  if (total_count <= thread_num) {
    thread_num = 1;
  }

  std::vector<std::thread> threads(thread_num);
  size_t task_offset = 0;

  for (size_t i = 0; i < thread_num; ++i) {
    size_t task_len = total_count / thread_num + (i < (total_count % thread_num) ? 1 : 0);
    threads[i] = std::thread(update_stddev_task, stddev, task_len, output + task_offset);
    task_offset += task_len;
  }

  for (size_t i = 0; i < thread_num; i++) {
    threads[i].join();
  }
}
}  // namespace

bool InitRandomNormal(float mean, float stddev, std::vector<size_t> out_shape, size_t global_seed, size_t op_seed,
                      float *output_data) {
  MS_ERROR_IF_NULL_W_RET_VAL(output_data, false);
  if (out_shape.size() == 0) {
    std::cout << "output data shape is error" << std::endl;
  }
  int64_t total_count = 1;
  for (uint32_t i = 0; i < out_shape.size(); i++) {
    total_count *= SizeToLong(out_shape[i]);
  }

  uint32_t thread_num = std::max(kMaxThreadNum, kCPUCoreNum);
  if (total_count <= thread_num) {
    thread_num = 1;
  }
  float *start_ptr = output_data;
  if (start_ptr == nullptr) {
    std::cout << "start_ptr is nullptr" << std::endl;
    return false;
  }
  // The value of thread_num is >= 1.
  int64_t batchSize = total_count / thread_num;
  std::vector<std::thread> threads(thread_num);
  int64_t seed = SizeToLong(global_seed);
  int64_t seed2 = SizeToLong(op_seed);
  seed = (seed == 0 && seed2 == 0) ? clock() : seed;
  PhiloxGenerator generator = PhiloxGenerator(seed, seed2);
  if (thread_num != 1) {
    float *offset_ptr = nullptr;
    for (uint32_t i = 0; i < thread_num - 1; i++) {
      offset_ptr = start_ptr + batchSize * i;
      threads[i] =
        std::thread(FillRandoms<NormalDistribution<PhiloxGenerator, float>>, generator, offset_ptr, batchSize, i);
    }
    offset_ptr = start_ptr + batchSize * (thread_num - 1);
    threads[thread_num - 1] = std::thread(FillRandoms<NormalDistribution<PhiloxGenerator, float>>, generator,
                                          offset_ptr, total_count - (thread_num - 1) * batchSize, thread_num - 1);
  } else {
    threads[0] =
      std::thread(FillRandoms<NormalDistribution<PhiloxGenerator, float>>, generator, start_ptr, total_count, 0);
  }
  for (uint32_t i = 0; i < thread_num; i++) {
    threads[i].join();
  }

  UpdateStandardDeviation(stddev, total_count, output_data);
  return true;
}
}  // namespace ps
}  // namespace mindspore
