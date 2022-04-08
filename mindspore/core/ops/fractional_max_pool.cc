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

#include "ops/fractional_max_pool.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kPoolingRatioDims = 4;
abstract::TupleShapePtr FractionalMaxPoolInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  const int64_t x_rank = 4;
  (void)CheckAndConvertUtils::CheckInteger("input_rank", SizeToLong(in_shape.size()), kEqual, x_rank, op_name);
  for (int i = 0; i < x_rank; i++) {
    if (in_shape[i] <= 0) {
      MS_EXCEPTION(ValueError) << "Input shape of FractionalMaxPool must be > 0 " << std::to_string(in_shape[i]) << ".";
    }
  }
  auto pooling_ratio = GetValue<std::vector<float>>(primitive->GetAttr(kPoolingRatio));
  if (pooling_ratio.size() != kPoolingRatioDims) {
    MS_EXCEPTION(ValueError) << "pooling_ratio_size of FractionalMaxPool must be 4, but got "
                             << std::to_string(pooling_ratio.size()) << ".";
  }
  if (pooling_ratio[kInputIndex0] != 1.0) {
    MS_EXCEPTION(ValueError) << "The first elements of pooling ratio must be 1.0, but got "
                             << std::to_string(pooling_ratio[0]) << ".";
  }
  if (pooling_ratio[kInputIndex1] < 1.0) {
    MS_EXCEPTION(ValueError) << "The elements of pooling ratio must be larger than 1.0, but pooling_ratio[1] = "
                             << std::to_string(pooling_ratio[kInputIndex1]) << ".";
  }
  if (pooling_ratio[kInputIndex2] < 1.0) {
    MS_EXCEPTION(ValueError) << "The elements of pooling ratio must be larger than 1.0, but pooling_ratio[2] = "
                             << std::to_string(pooling_ratio[kInputIndex2]) << ".";
  }
  if (pooling_ratio[kInputIndex3] != 1.0) {
    MS_EXCEPTION(ValueError) << "The last elements of pooling ratio must be 1.0, but got "
                             << std::to_string(pooling_ratio[kInputIndex3]) << ".";
  }
  std::vector<int64_t> out_shape(x_rank);
  for (int i = 0; i < x_rank; ++i) {
    out_shape[i] = static_cast<int64_t>(std::floor(in_shape[i] / pooling_ratio[i]));
  }
  if (std::any_of(out_shape.begin(), out_shape.end(), [](int64_t a) { return a <= 0; })) {
    MS_EXCEPTION(ValueError) << "output shape <=0, pooling_ratio is not valid.";
  }
  int64_t row = out_shape[kInputIndex1] + 1;
  int64_t col = out_shape[kInputIndex2] + 1;
  std::vector<int64_t> row_dim = {row};
  std::vector<int64_t> col_dim = {col};
  abstract::ShapePtr output0_shape = std::make_shared<abstract::Shape>(out_shape);
  abstract::ShapePtr output1_shape = std::make_shared<abstract::Shape>(row_dim);
  abstract::ShapePtr output2_shape = std::make_shared<abstract::Shape>(col_dim);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{output0_shape, output1_shape, output2_shape});
}

TuplePtr FractionalMaxPoolInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x_dtype = input_args[kInputIndex0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kInt32, kInt64};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("input", x_dtype, valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, kInt64, kInt64});
}
}  // namespace

MIND_API_OPERATOR_IMPL(FractionalMaxPool, BaseOperator);
AbstractBasePtr FractionalMaxPoolInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = FractionalMaxPoolInferType(primitive, input_args);
  auto infer_shape = FractionalMaxPoolInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(FractionalMaxPool, prim::kPrimFractionalMaxPool, FractionalMaxPoolInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
