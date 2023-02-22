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
#include <vector>
#include <memory>
#include <algorithm>

#include "ops/diagonal.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DiagonalInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kDimNum = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  const int64_t dyn_shape = abstract::Shape::kShapeDimAny;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = x_shape.size();
  auto offset = GetValue<int64_t>(primitive->GetAttr("offset"));
  auto dim1 = GetValue<int64_t>(primitive->GetAttr("dim1"));
  auto dim2 = GetValue<int64_t>(primitive->GetAttr("dim2"));

  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  CheckAndConvertUtils::CheckInRange<int64_t>("dim1", dim1, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  CheckAndConvertUtils::CheckInRange<int64_t>("dim2", dim2, kIncludeBoth, {-x_rank, x_rank - 1}, prim_name);
  if (x_rank < kDimNum) {
    MS_EXCEPTION(ValueError) << "For 'Diagonal', input must be at least 2-dimensional, but got : " << x_rank << ".";
  }
  auto tmp_dim1 = (dim1 < 0) ? dim1 + x_rank : dim1;
  auto tmp_dim2 = (dim2 < 0) ? dim2 + x_rank : dim2;
  if (tmp_dim1 == tmp_dim2) {
    MS_EXCEPTION(ValueError) << "For 'Diagonal', dim1 and dim2 cannot be identical, but got : dim1 =" << dim1
                             << " and dim2 = " << dim2 << ".";
  }
  std::vector<int64_t> out_shape;
  for (size_t tmp_dim = 0; tmp_dim < x_rank; tmp_dim++) {
    if (tmp_dim != tmp_dim1 && tmp_dim != tmp_dim2) {
      out_shape.push_back(x_shape[tmp_dim]);
    }
  }
  int64_t dsize = dyn_shape;
  if (x_shape[tmp_dim1] != dyn_shape && x_shape[tmp_dim2] != dyn_shape) {
    if (offset >= 0) {
      dsize = std::max<int64_t>(std::min(x_shape[tmp_dim1], x_shape[tmp_dim2] - offset), 0);
    } else {
      dsize = std::max<int64_t>(std::min(x_shape[tmp_dim1] + offset, x_shape[tmp_dim2]), 0);
    }
  }
  out_shape.push_back(dsize);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr DiagonalInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto x_dtype = input_args[0]->BuildType();
  return CheckAndConvertUtils::CheckTensorTypeValid("input type", x_dtype, common_valid_types, primitive->name());
}
}  // namespace
AbstractBasePtr DiagonalInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = DiagonalInferType(primitive, input_args);
  auto infer_shape = DiagonalInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(Diagonal, BaseOperator);

// AG means auto generated
class MIND_API AGDiagonalInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagonalInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagonalInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DiagonalInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Diagonal, prim::kPrimDiagonal, AGDiagonalInfer, false);
}  // namespace ops
}  // namespace mindspore
