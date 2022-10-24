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

#ifndef MINDSPORE_LITE_SRC_COMMON_RANDOM_DATA_GENERATOR_H_
#define MINDSPORE_LITE_SRC_COMMON_RANDOM_DATA_GENERATOR_H_

#include <random>
#include "src/common/log_adapter.h"
#include "include/api/types.h"
namespace mindspore {
namespace lite {
int GenRandomData(mindspore::MSTensor *tensors);

int GenRandomData(size_t size, void *data, int data_type);

template <typename T, typename Distribution>
void FillRandomData(size_t size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  MS_ASSERT(data != nullptr);
  size_t elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&]() { return static_cast<T>(distribution(random_engine)); });
}
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_COMMON_RANDOM_DATA_GENERATOR_H_
