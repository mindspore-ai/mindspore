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
#ifndef MINDSPORE_CCSRC_PS_RANDOM_NORMAL_RANDOM_NORMAL_H_
#define MINDSPORE_CCSRC_PS_RANDOM_NORMAL_RANDOM_NORMAL_H_
#include <vector>
#include <cstddef>

namespace mindspore {
namespace ps {
bool InitRandomNormal(float mean, float stddev, std::vector<size_t> out_shape, size_t global_seed, size_t op_seed,
                      float *output_data);
}  // namespace ps
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PS_RANDOM_NORMAL_RANDOM_NORMAL_H_
