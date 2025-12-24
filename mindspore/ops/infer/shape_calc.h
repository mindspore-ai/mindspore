/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_SHAPE_CALC_H_
#define MINDSPORE_CORE_OPS_SHAPE_CALC_H_

#include <memory>
#include <vector>
#include "ir/anf.h"
#include "ir/functor.h"
#include "mindapi/base/macros.h"
#include "ops/base_operator.h"

namespace mindspore::ops {
constexpr auto kNameShapeCalc = "ShapeCalc";
constexpr auto kAttrCalcResult = "calc_result";
class OPS_API ShapeCalc : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ShapeCalc);
  ShapeCalc() : BaseOperator(kNameShapeCalc) { InitIOName({"inputs"}, {"outputs"}); }

  ShapeCalcBaseFunctorPtr get_functor() const;
  ShapeArray get_calc_result() const;
};

OPS_API bool TryGetShapeArg(const AbstractBasePtr &abs, ShapeArray *args, std::vector<std::vector<size_t>> *pos_idx);
}  // namespace mindspore::ops

#endif  // MINDSPORE_CORE_OPS_SHAPE_CALC_H_
