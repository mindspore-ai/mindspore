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

#include "ops/cross.h"

#include <memory>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void CheckAndUpdateDim(const PrimitivePtr &primitive, const ShapeVector &x1_shape, const int64_t &default_dim,
                       int64_t *dim) {
  if (*dim == default_dim) {
    int64_t dim_size_value = 3;
    for (size_t i = 0; i < x1_shape.size(); i++) {
      if (x1_shape[i] == dim_size_value) {
        *dim = SizeToLong(i);
        break;
      }
      if (i == x1_shape.size() - 1 && x1_shape[i] != dim_size_value) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size of inputs dim must be 3, but got "
                                 << x1_shape[i] << ".";
      }
    }
  }
  if ((*dim < -static_cast<int64_t>(x1_shape.size()) || *dim > static_cast<int64_t>(x1_shape.size()) - 1) &&
      *dim != default_dim) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', dim must be between "
                             << -static_cast<int64_t>(x1_shape.size()) << " and "
                             << static_cast<int64_t>(x1_shape.size()) - 1 << " , but got " << *dim << ".";
  }
  if (*dim < 0 && *dim != default_dim) {
    *dim = static_cast<int64_t>(x1_shape.size()) + *dim;
  }
  return;
}

abstract::ShapePtr CrossInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  // support dynamic rank
  if (IsDynamicRank(x1_shape) || IsDynamicRank(x2_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto dim = GetValue<int64_t>(primitive->GetAttr("dim"));
  if (x1_shape.size() != x2_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the shape of two inputs must have the same size, but got 'x1' shape size: "
                             << x1_shape.size() << ", 'x2' shape size: " << x2_shape.size() << ".";
  }

  // Dynamic Shape
  if (IsDynamic(x1_shape) || IsDynamic(x2_shape)) {
    ShapeVector shape_out;
    for (size_t i = 0; i < x1_shape.size(); ++i) {
      shape_out.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(shape_out);
  }

  for (size_t i = 0; i < x1_shape.size(); ++i) {
    if (x1_shape[i] != x2_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', 'x1' and 'x2' must have the same shape. but got 'x1' shape: " << x1_shape
                               << ", 'x2' shape: " << x2_shape << ".";
    }
  }
  (void)CheckAndConvertUtils::CheckInteger("dim of x1", SizeToLong(x1_shape.size()), kGreaterThan, 0,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("dim of x2", SizeToLong(x2_shape.size()), kGreaterThan, 0,
                                           primitive->name());

  int64_t default_dim = -65530;
  CheckAndUpdateDim(primitive, x1_shape, default_dim, &dim);

  int64_t dim_size = 3;
  if (x1_shape[LongToSize(dim)] != dim_size && x2_shape[LongToSize(dim)] != dim_size && dim != default_dim) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the size of inputs dim must be 3, but got "
                             << x1_shape[LongToSize(dim)] << ".";
  }
  return std::make_shared<abstract::Shape>(x1_shape);
}

TypePtr CrossInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("Cross infer", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kInt8,    kInt16,  kInt32,  kInt64,  kUInt8,     kFloat16,   kFloat32,
                                         kFloat64, kUInt16, kUInt32, kUInt64, kComplex64, kComplex128};
  auto x1_type = input_args[0]->BuildType();
  auto x2_type = input_args[1]->BuildType();
  auto tensor_type = x2_type->cast<TensorTypePtr>();
  auto element = tensor_type->element();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2", x2_type, valid_types, primitive->name());
  return CheckAndConvertUtils::CheckTensorTypeValid("x1", x1_type, {element}, primitive->name());
}
}  // namespace

AbstractBasePtr CrossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = CrossInferType(primitive, input_args);
  auto infer_shape = CrossInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(Cross, BaseOperator);
void Cross::Init(const int64_t dim) { this->set_dim(dim); }

void Cross::set_dim(const int64_t dim) { (void)this->AddAttr("dim", api::MakeValue(dim)); }

int64_t Cross::get_dim() const {
  auto value_ptr = this->GetAttr("dim");
  return GetValue<int64_t>(value_ptr);
}

// AG means auto generated
class MIND_API AGCrossInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CrossInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CrossInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CrossInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Cross, prim::kPrimCross, AGCrossInfer, false);
}  // namespace ops
}  // namespace mindspore
