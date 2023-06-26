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

#include "ops/grad/fractional_avg_pool_grad.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/conv_pool_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputSizes = 4;
constexpr size_t kInputDims = 1;
abstract::ShapePtr FractionalAvgPoolGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);

  auto shape_v = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (shape_v.size() > kInputDims) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the input 'orig_input_tensor_shape' tensor must be a 1-D tensor.";
  }
  std::vector<int64_t> output_shape = GetShapeValue(primitive, input_args[kInputIndex0]);
  if (IsDynamicRank(output_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>(kInputSizes, abstract::Shape::kShapeDimAny));
  }
  (void)CheckAndConvertUtils::CheckInteger("orig_input_tensor_shape", SizeToLong(output_shape.size()), kEqual,
                                           SizeToLong(kInputSizes), op_name);
  int64_t shape_m = SizeToLong(SizeOf(output_shape));
  if (shape_m > max_length) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the number of elements of output must be less than max length: " << max_length
                             << ", but got " << shape_m
                             << "! The shape of  output must be reduced or max_length must be increased";
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr FractionalAvgPoolGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto orig_input_shape_dtype = input_args[kInputIndex0]->BuildType();
  auto backprop_dtype = input_args[kInputIndex1]->BuildType();
  auto row_seq_dtype = input_args[kInputIndex2]->BuildType();
  auto col_seq_dtype = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kInt32, kInt64};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("backprop", backprop_dtype, valid_types, op_name);
  const std::set<TypePtr> seq_valid_types = {kInt64};
  std::map<std::string, TypePtr> seq_types;
  (void)seq_types.emplace("orig_input_shape", orig_input_shape_dtype);
  (void)seq_types.emplace("row_seq", row_seq_dtype);
  (void)seq_types.emplace("col_seq", col_seq_dtype);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(seq_types, seq_valid_types, op_name);
  return type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(FractionalAvgPoolGrad, BaseOperator);
AbstractBasePtr FractionalAvgPoolGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = FractionalAvgPoolGradInferType(primitive, input_args);
  auto infer_shape = FractionalAvgPoolGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool FractionalAvgPoolGrad::get_overlapping() const {
  auto value_ptr = GetAttr("overlapping");
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGFractionalAvgPoolGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalAvgPoolGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalAvgPoolGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalAvgPoolGradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FractionalAvgPoolGrad, prim::kPrimFractionalAvgPoolGrad, AGFractionalAvgPoolGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
