/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_COMPARISON_FUNCTION_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_COMPARISON_FUNCTION_INFO_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "ir/value.h"
#include "parallel/ops_info/arithmetic_info.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
class EqualInfo : public ArithmeticBase {
 public:
  EqualInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
            const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~EqualInfo() override = default;
};

class NotEqualInfo : public ArithmeticBase {
 public:
  NotEqualInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
               const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~NotEqualInfo() override = default;
};

class MaximumInfo : public ArithmeticBase {
 public:
  MaximumInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
              const PrimitiveAttrs& attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs) {}
  ~MaximumInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_OPS_INFO_PARALLEL_COMPARISON_FUNCTION_INFO_H_
