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

#ifndef MINDSPORE_CORE_OPS_LAYER_NORM_H_
#define MINDSPORE_CORE_OPS_LAYER_NORM_H_
#include <string>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLayerNorm = "LayerNorm";
class LayerNorm : public PrimitiveC {
 public:
  LayerNorm() : PrimitiveC(kNameLayerNorm) {}
  explicit LayerNorm(const std::string k_name) : PrimitiveC(k_name) {}
  ~LayerNorm() = default;
  MS_DECLARE_PARENT(LayerNorm, PrimitiveC);
  void Init(const int64_t begin_norm_axis = 1, const int64_t begin_params_axis = 1, const float epsilon = 1e-7);
  void set_begin_norm_axis(const int64_t begin_norm_axis);
  void set_begin_params_axis(const int64_t begin_params_axis);
  void set_epsilon(const float epsilon);
  int64_t get_begin_norm_axis() const;
  int64_t get_begin_params_axis() const;
  float get_epsilon() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LAYER_NORM_H_
