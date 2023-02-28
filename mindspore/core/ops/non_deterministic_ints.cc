/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "ops/non_deterministic_ints.h"

#include <string>
#include <memory>
#include <set>
#include <vector>

#include "ir/dtype/number.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NonDeterministicIntsInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const uint32_t kMinShapeDim = 2;
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto shape_v = GetShapeValue(primitive, input_args[0]);
  if (!IsDynamic(shape_v)) {
    if (shape_v.size() < kMinShapeDim) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'shape' must be at least 2-dimensional.";
    }
    if (std::any_of(shape_v.begin(), shape_v.end(), [](int64_t x) { return x <= 0; })) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', 'shape' can't contain non-positive dim.";
    }
    auto shape_m = static_cast<int64_t>(SizeOf(shape_v));
    if (shape_m > max_length) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the number of elements of output must be less than max length: " << max_length
                               << ", but got " << shape_m
                               << ". The shape of output must be reduced or max_length must be increased";
    }
  }
  return std::make_shared<abstract::Shape>(shape_v);
}

TypePtr NonDeterministicIntsInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  const std::set<TypePtr> valid_input_types = {kInt32, kInt64, kUInt32, kUInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("shape", input_args[0]->BuildType(), valid_input_types, prim_name);
  auto dtype_value = prim->GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype_value);
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "The dtype of NonDeterministicInts is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  const std::set<TypePtr> valid_output_types = {kInt32, kInt64, kUInt32, kUInt64};
  return CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_output_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(NonDeterministicInts, BaseOperator);
AbstractBasePtr NonDeterministicIntsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = NonDeterministicIntsInferType(primitive, input_args);
  auto infer_shape = NonDeterministicIntsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGNonDeterministicIntsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NonDeterministicIntsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NonDeterministicIntsInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NonDeterministicIntsInfer(engine, primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NonDeterministicInts, prim::kPrimNonDeterministicInts, AGNonDeterministicIntsInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
