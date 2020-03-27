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

#ifndef PARALLEL_AUTO_PARALLEL_REC_TENSOR_H_
#define PARALLEL_AUTO_PARALLEL_REC_TENSOR_H_

#include "parallel/auto_parallel/rec_core/rec_strategy.h"

namespace mindspore {
namespace parallel {
enum TensorType { kInt8, kFloat16, kFloat32, kDouble64 };

struct Shape4D {
  int32_t shape_n = 1;
  int32_t shape_c = 1;
  int32_t shape_h = 1;
  int32_t shape_w = 1;
};

struct TensorParam {
  TensorType tensor_type = kFloat32;  // default as float.
  Shape4D tensor_shape;
  TensorStr4D tensor_str;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // PARALLEL_AUTO_PARALLEL_REC_TENSOR_H_
