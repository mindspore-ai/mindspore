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

#ifndef MINDSPORE_CORE_OPS_SGD_H_
#define MINDSPORE_CORE_OPS_SGD_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSGD = "SGD";
class SGD : public PrimitiveC {
 public:
  SGD() : PrimitiveC(kNameSGD) {}
  ~SGD() = default;
  MS_DECLARE_PARENT(SGD, PrimitiveC);
  void Init(const float dampening = 0.0, const float weight_decay = 0.0, const bool nesterov = false);
  void set_dampening(const float dampening);
  void set_weight_decay(const float weight_decay);
  void set_nesterov(const bool nesterov);
  float get_dampening() const;
  float get_weight_decay() const;
  bool get_nesterov() const;
};
AbstractBasePtr SGDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args);
using PrimSGD = std::shared_ptr<SGD>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SGD_H_
