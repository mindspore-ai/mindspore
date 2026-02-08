/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <vector>
#include <stdint.h>
#include "custom_aot_extra.h"

extern "C" std::vector<std::vector<int64_t>> MoeSoftMaxTopkInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
  std::vector<std::vector<int64_t>> res_output_shape;
  auto input0_size = ndims[0];
  std::vector<int64_t> out1_shape;
  out1_shape.emplace_back(shapes[0][0]);
  out1_shape.emplace_back(extra->Attr<int64_t>("attr_k"));
  res_output_shape.emplace_back(out1_shape);

  std::vector<int64_t> out2_shape;
  out2_shape.emplace_back(shapes[0][0]);
  out2_shape.emplace_back(extra->Attr<int64_t>("attr_k"));
  res_output_shape.emplace_back(out2_shape);

  return res_output_shape;
}
