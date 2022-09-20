/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_GRAD_LAYERNORMGRADGRAD_H_
#define MINDSPORE_CORE_OPS_GRAD_LAYERNORMGRADGRAD_H_

#include <string>
#include <vector>
#include <memory>

#include "abstract/abstract_value.h"
#include "ops/primitive_c.h"
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLayerNormGradGrad = "LayerNormGradGrad";
class MIND_API LayerNormGradGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LayerNormGradGrad);
  LayerNormGradGrad() : BaseOperator(kNameLayerNormGradGrad) {
    InitIOName({"x", "dy", "variance", "mean", "gamma", "d_dx", "d_dg", "d_db"}, {"sopd_x", "sopd_dy", "sopd_gamma"});
  }
  explicit LayerNormGradGrad(const std::string k_name) : BaseOperator(k_name) {
    InitIOName({"x", "dy", "variance", "mean", "gamma", "d_dx", "d_dg", "d_db"}, {"sopd_x", "sopd_dy", "sopd_gamma"});
  }
  void Init(const int64_t begin_norm_axis = 1, const int64_t begin_params_axis = 1);
  void set_begin_norm_axis(const int64_t begin_norm_axis);
  void set_begin_params_axis(const int64_t begin_params_axis);
  int64_t get_begin_norm_axis() const;
  int64_t get_begin_params_axis() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRAD_LAYERNORMGRADGRAD_H_
