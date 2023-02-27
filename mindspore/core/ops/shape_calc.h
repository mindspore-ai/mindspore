/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <unordered_set>
#include <memory>
#include "ir/anf.h"
#include "ops/base_operator.h"
#include "utils/hash_map.h"
#include "mindapi/base/macros.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
namespace ops {
constexpr auto kNameShapeCalc = "ShapeCalc";
constexpr auto kAttrShapeFunc = "shape_func";
constexpr auto kAttrInferFunc = "infer_func";
constexpr auto kAttrValueDependIndices = "value_depend_indices";
using ShapeFunc = std::function<ShapeArray(const ShapeArray &)>;
using InferFunc = std::function<ShapeVector(const ShapeArray &, const std::unordered_set<size_t> &)>;

class MS_CORE_API ShapeFunction : public Value {
 public:
  explicit ShapeFunction(const ShapeFunc &impl) : impl_(impl) {}
  ~ShapeFunction() override = default;
  MS_DECLARE_PARENT(ShapeFunction, Value)

  bool operator==(const Value &other) const override {
    if (other.isa<ShapeFunction>()) {
      return &other == this;
    }
    return false;
  }

  ShapeFunc impl() const { return impl_; }

 private:
  ShapeFunc impl_{nullptr};
};
using ShapeFunctionPtr = std::shared_ptr<ShapeFunction>;

class MS_CORE_API InferFunction : public Value {
 public:
  explicit InferFunction(const InferFunc &impl) : impl_(impl) {}
  ~InferFunction() override = default;
  MS_DECLARE_PARENT(InferFunction, Value)

  bool operator==(const Value &other) const override {
    if (other.isa<InferFunction>()) {
      return &other == this;
    }
    return false;
  }

  InferFunc impl() const { return impl_; }

 private:
  InferFunc impl_{nullptr};
};
using InferFunctionPtr = std::shared_ptr<InferFunction>;

class MIND_API ShapeCalc : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ShapeCalc);
  ShapeCalc() : BaseOperator(kNameShapeCalc) { InitIOName({"inputs"}, {"outputs"}); }

  ShapeFunc get_shape_func() const;
  std::vector<int64_t> get_value_depend_indices() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SHAPE_CALC_H_
