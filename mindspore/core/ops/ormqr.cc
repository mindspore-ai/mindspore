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

#include <set>
#include <vector>
#include <memory>
#include <map>
#include <string>

#include "ops/ormqr.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr OrmqrInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNoBatch = 2;
  const size_t kRowIndex = 2;
  const size_t kColIndex = 1;
  const size_t kTwo = 2;
  auto left = GetValue<bool>(primitive->GetAttr(kAttrLeft));
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto tau_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto other_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamic(x_shape) || IsDynamicRank(tau_shape) || IsDynamic(tau_shape) ||
      IsDynamicRank(other_shape) || IsDynamic(other_shape)) {
    return std::make_shared<abstract::Shape>(other_shape);
  }
  auto x_rank = x_shape.size();
  auto tau_rank = tau_shape.size();
  auto other_rank = other_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(x_rank), kGreaterEqual, kTwo, primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("other_rank", SizeToLong(other_rank), kGreaterEqual, kTwo,
                                           primitive->name());

  if ((x_rank - kColIndex) != tau_rank) {
    MS_EXCEPTION(ValueError) << "For Ormqr,  tau should have one dimension less than x"
                             << ", while rank of x is " << x_shape.size() << " and "
                             << "rank of tau is " << tau_shape.size() << ".";
  }
  if (x_rank != other_rank) {
    MS_EXCEPTION(ValueError) << "For Ormqr,  other should have same dimension with x"
                             << ", while rank of x is " << x_shape.size() << " and "
                             << "rank of other is " << other_shape.size() << ".";
  }
  if (x_shape.size() > kInputNoBatch) {
    for (size_t i = 0; i < x_rank - kRowIndex; i++) {
      if (x_shape[i] != tau_shape[i]) {
        MS_EXCEPTION(ValueError) << "For Ormqr, tau.shape[:-2] must be equal to x.shape[:-2], but x.shape[" << i
                                 << "] is " << x_shape[i] << ", and tau.shape[" << i << "] is " << tau_shape[i] << ".";
      }
      if (x_shape[i] != other_shape[i]) {
        MS_EXCEPTION(ValueError) << "For Ormqr, other.shape[:-2] must be equal to x.shape[:-2], but x.shape[" << i
                                 << "] is " << x_shape[i] << ", and other.shape[" << i << "] is " << other_shape[i]
                                 << ".";
      }
    }
  }
  if (left) {
    if (*(other_shape.end() - kRowIndex) < *(tau_shape.end() - kColIndex)) {
      MS_EXCEPTION(ValueError) << "For Ormqr, other.shape[-2] must be greater than or equal to tau.shape[-1]"
                               << ", while other.shape[-2] is " << other_shape[other_rank - kRowIndex] << " and "
                               << "tau.shape[-1] is " << tau_shape[tau_rank - kColIndex] << ".";
    }
    if (*(x_shape.end() - kRowIndex) != *(other_shape.end() - kRowIndex)) {
      MS_EXCEPTION(ValueError) << "For Ormqr, other.shape[-2] must be equal to x.shape[-2]"
                               << ", while x.shape[-2] is " << x_shape[x_rank - kRowIndex] << " and "
                               << "other.shape[-2] is " << other_shape[other_rank - kRowIndex] << ".";
    }
  } else {
    if (*(other_shape.end() - kColIndex) < *(tau_shape.end() - kColIndex)) {
      MS_EXCEPTION(ValueError) << "For Ormqr, other.shape[-1] must be greater than or equal to tau.shape[-1]"
                               << ", while other.shape[-1] is " << other_shape[other_rank - kColIndex] << " and "
                               << "tau.shape[-1] is " << tau_shape[tau_rank - kColIndex] << ".";
    }
    if (*(x_shape.end() - kRowIndex) != *(other_shape.end() - kColIndex)) {
      MS_EXCEPTION(ValueError) << "For Ormqr, other.shape[-1] must be equal to x.shape[-2]"
                               << ", while x.shape[-2] is " << x_shape[x_rank - kRowIndex] << " and "
                               << "other.shape[-1] is " << other_shape[other_rank - kColIndex] << ".";
    }
  }

  return std::make_shared<abstract::Shape>(other_shape);
}

TypePtr OrmqrInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  std::map<std::string, TypePtr> types;
  auto x_type = input_args[0]->BuildType();
  auto tau_type = input_args[kInputIndex1]->BuildType();
  auto other_type = input_args[kInputIndex2]->BuildType();
  (void)types.emplace("x", x_type);
  (void)types.emplace("tau", tau_type);
  (void)types.emplace("other", other_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return x_type;
}
}  // namespace

void Ormqr::Init(const bool left, const bool transpose) {
  set_left(left);
  set_transpose(transpose);
}

void Ormqr::set_left(const bool left) { (void)this->AddAttr(kAttrLeft, api::MakeValue(left)); }

void Ormqr::set_transpose(const bool transpose) { (void)this->AddAttr(kAttrTranspose, api::MakeValue(transpose)); }

bool Ormqr::get_left() const { return GetValue<bool>(GetAttr(kAttrLeft)); }
bool Ormqr::get_transpose() const { return GetValue<bool>(GetAttr(kAttrTranspose)); }

MIND_API_OPERATOR_IMPL(Ormqr, BaseOperator);
AbstractBasePtr OrmqrInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = OrmqrInferType(primitive, input_args);
  auto infer_shape = OrmqrInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGOrmqrInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return OrmqrInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return OrmqrInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return OrmqrInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Ormqr, prim::kPrimOrmqr, AGOrmqrInfer, false);
}  // namespace ops
}  // namespace mindspore
