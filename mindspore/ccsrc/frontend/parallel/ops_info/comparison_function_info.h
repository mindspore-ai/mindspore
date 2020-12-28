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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_COMPARISON_FUNCTION_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_COMPARISON_FUNCTION_INFO_H_

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/arithmetic_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class EqualInfo : public ArithmeticBase {
 public:
  EqualInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<EqualCost>()) {}
  ~EqualInfo() override = default;
};

class ApproximateEqualInfo : public ArithmeticBase {
 public:
  ApproximateEqualInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                       const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<ApproximateEqualCost>()) {}
  ~ApproximateEqualInfo() override = default;
};

class NotEqualInfo : public ArithmeticBase {
 public:
  NotEqualInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<NotEqualCost>()) {}
  ~NotEqualInfo() override = default;
};

class MaximumInfo : public ArithmeticBase {
 public:
  MaximumInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MaximumCost>()) {}
  ~MaximumInfo() override = default;
};

class MinimumInfo : public ArithmeticBase {
 public:
  MinimumInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<MinimumCost>()) {}
  ~MinimumInfo() override = default;
};

class GreaterInfo : public ArithmeticBase {
 public:
  GreaterInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<GreaterCost>()) {}
  ~GreaterInfo() override = default;
};

class GreaterEqualInfo : public ArithmeticBase {
 public:
  GreaterEqualInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<GreaterEqualCost>()) {}
  ~GreaterEqualInfo() override = default;
};

class LessInfo : public ArithmeticBase {
 public:
  LessInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<LessCost>()) {}
  ~LessInfo() override = default;
};

class LessEqualInfo : public ArithmeticBase {
 public:
  LessEqualInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ArithmeticBase(name, inputs_shape, outputs_shape, attrs, std::make_shared<LessEqualCost>()) {}
  ~LessEqualInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_COMPARISON_FUNCTION_INFO_H_
