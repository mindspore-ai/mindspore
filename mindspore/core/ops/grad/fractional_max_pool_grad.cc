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

#include "ops/grad/fractional_max_pool_grad.h"

#include <string>
#include <memory>
#include <set>
#include <vector>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t x_rank = 4;
abstract::ShapePtr FractionalMaxPoolGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShapeTrack())[kShape];
  auto out_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShapeTrack())[kShape];
  auto backprop_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShapeTrack())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("orig_input_rank", SizeToLong(in_shape.size()), kEqual, x_rank, op_name);
  (void)CheckAndConvertUtils::CheckInteger("orig_output_rank", SizeToLong(out_shape.size()), kEqual, x_rank, op_name);
  (void)CheckAndConvertUtils::CheckInteger("backprop_rank", SizeToLong(backprop_shape.size()), kEqual, x_rank, op_name);
  auto infer_shape = std::make_shared<abstract::Shape>(in_shape);
  return infer_shape;
}

TypePtr FractionalMaxPoolGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto orig_input_dtype = input_args[kInputIndex0]->BuildType();
  auto orig_output_dtype = input_args[kInputIndex1]->BuildType();
  auto out_backprop_dtype = input_args[kInputIndex2]->BuildType();
  auto row_seq_dtype = input_args[kInputIndex3]->BuildType();
  auto col_seq_dtype = input_args[kInputIndex4]->BuildType();
  const std::set<TypePtr> input_valid_types = {kFloat32, kFloat64, kInt32, kInt64};
  const std::set<TypePtr> seq_valid_types = {kInt64};
  std::map<std::string, TypePtr> tensor_types;
  std::map<std::string, TypePtr> seq_types;
  (void)tensor_types.emplace("orig_input", orig_input_dtype);
  (void)tensor_types.emplace("orig_output", orig_output_dtype);
  (void)tensor_types.emplace("out_backprop", out_backprop_dtype);
  (void)seq_types.emplace("row_seq", row_seq_dtype);
  (void)seq_types.emplace("col_seq", col_seq_dtype);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(tensor_types, input_valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(seq_types, seq_valid_types, op_name);
  return orig_input_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(FractionalMaxPoolGrad, BaseOperator);
AbstractBasePtr FractionalMaxPoolGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = FractionalMaxPoolGradInferType(primitive, input_args);
  auto infer_shape = FractionalMaxPoolGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool FractionalMaxPoolGrad::get_overlapping() const {
  auto value_ptr = GetAttr("overlapping");
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGFractionalMaxPoolGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FractionalMaxPoolGrad, prim::kPrimFractionalMaxPoolGrad, AGFractionalMaxPoolGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
