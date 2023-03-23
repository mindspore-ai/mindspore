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

#ifndef PARALLEL_AUTO_PARALLEL_REC_STRATEGY_H_
#define PARALLEL_AUTO_PARALLEL_REC_STRATEGY_H_

#include <cstddef>
#include <cstdint>

namespace mindspore {
namespace parallel {
constexpr size_t MAX_INPUT_NUM = 20;
constexpr size_t STR_DIM_NUM = 4;

struct TensorStr4D {
  float str_n = 1;
  float str_c = 1;
  float str_h = 1;
  float str_w = 1;
};

struct StrategyRec {
  TensorStr4D inputTensor[MAX_INPUT_NUM];
  TensorStr4D outputTensor;
  int64_t cut_counter = 0;
  double cost = 0;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_STRATEGY_H_
