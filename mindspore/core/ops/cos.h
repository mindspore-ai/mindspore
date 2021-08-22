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

#ifndef MINDSPORE_CORE_OPS_COS_H_
#define MINDSPORE_CORE_OPS_COS_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCos = "Cos";
class Cos : public PrimitiveC {
 public:
  Cos() : PrimitiveC(kNameCos) {}
  ~Cos() = default;
  MS_DECLARE_PARENT(Cos, PrimitiveC);
  void Init(float alpha = 0.0);
};
AbstractBasePtr CosInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args);
using PrimCos = std::shared_ptr<Cos>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_COS_H_
