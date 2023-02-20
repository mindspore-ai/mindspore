/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/fill_diagonal.h"

#include <memory>
#include <set>

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
#include "ir/primitive.h"
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
abstract::ShapePtr FillDiagonalInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  int64_t x_size = SizeToLong(x_shape.size());
  const int64_t kDimSize2 = 2;
  if (x_size < kDimSize2) {
    MS_EXCEPTION(ValueError) << "The primitive[FillDiagonal] argument [input_x] must be a Tensor whose dimension is "
                                "greater than or equal to 2, but got its dimension ["
                             << x_size << "].";
  }
  if (x_size > kDimSize2) {
    for (int64_t i = 1; i < x_size; i++) {
      if (x_shape[LongToSize(i)] != x_shape[LongToSize(i - 1)] && x_shape[LongToSize(i)] != -1 &&
          x_shape[LongToSize(i - 1)] != -1) {
        MS_EXCEPTION(ValueError) << "The primitive[FillDiagonal] argument [input_x] must be a Tensor with the same "
                                    "size in all dimensions when its dimension is greater than 2";
      }
    }
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr FillDiagonalInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat32, kInt32, kInt64};
  auto x_dtype =
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, primitive->name());
  return std::make_shared<TensorType>(x_dtype);
}
}  // namespace

void FillDiagonal::Init(const float fill_value, const bool wrap) {
  set_fill_value(fill_value);
  set_wrap(wrap);
}

void FillDiagonal::set_fill_value(const float fill_value) {
  (void)this->AddAttr(kFillValue, api::MakeValue(fill_value));
}

float FillDiagonal::get_fill_value() const { return GetValue<float>(GetAttr(kFillValue)); }

void FillDiagonal::set_wrap(const bool wrap) { (void)this->AddAttr(kWrap, api::MakeValue(wrap)); }

bool FillDiagonal::get_wrap() const { return GetValue<bool>(GetAttr(kWrap)); }

MIND_API_OPERATOR_IMPL(FillDiagonal, BaseOperator);
AbstractBasePtr FillDiagonalInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto infer_type = FillDiagonalInferType(primitive, input_args);
  auto infer_shape = FillDiagonalInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGFillDiagonalInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FillDiagonalInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FillDiagonalInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FillDiagonalInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FillDiagonal, prim::kPrimFillDiagonal, AGFillDiagonalInfer, false);
}  // namespace ops
}  // namespace mindspore
