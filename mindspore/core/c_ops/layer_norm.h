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

#ifndef MINDSPORE_CORE_C_OPS_LAYERNORM_H_
#define MINDSPORE_CORE_C_OPS_LAYERNORM_H_
#include "c_ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
constexpr auto kNameLayerNorm = "LayerNorm";
class LayerNorm : public PrimitiveC {
 public:
  LayerNorm() : PrimitiveC(kNameLayerNorm) {}
  ~LayerNorm() = default;
  MS_DECLARE_PARENT(LayerNorm, PrimitiveC);
  void Init(int64_t begin_norm_axis = 1, int64_t begin_params_axis = 1, float epsilon = 1e-7);
  void set_begin_norm_axis(int64_t begin_norm_axis);
  void set_begin_params_axis(int64_t begin_params_axis);
  void set_epsilon(float epsilon);
  int64_t get_begin_norm_axis();
  int64_t get_begin_params_axis();
  float get_epsilon();
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_C_OPS_LAYERNORM_H_
