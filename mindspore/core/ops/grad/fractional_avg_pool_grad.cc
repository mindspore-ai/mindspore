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
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInpuSizes = 4;
constexpr size_t kInpuDims = 1;
constexpr int64_t kDynamicRankValue = -2;
abstract::ShapePtr FractionalAvgPoolGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input 'orig_input_tensor_shape' must be a tensor.";
  }
  auto input_shape = input_args[kInputIndex0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto input_shape_value_ptr = input_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input_shape_value_ptr);
  auto input_shape_tensor = input_shape_value_ptr->cast<tensor::TensorPtr>();
  auto input_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_type_id = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type_id);
  auto input_type_element = input_type_id->element();
  MS_EXCEPTION_IF_NULL(input_type_element);
  auto shape_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape]);
  auto shape_v = shape_ptr->shape();
  if (shape_v.size() > kInpuDims) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the input 'orig_input_tensor_shape' tensor must be a 1-D tensor.";
  }

  std::vector<int64_t> output_shape;
  if (IsDynamicRank(shape_v)) {
    output_shape.push_back(kDynamicRankValue);
    return std::make_shared<abstract::Shape>(output_shape);
  }

  if (IsDynamic(shape_v)) {
    output_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                    abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(output_shape);
  }

  if (!input_args[kInputIndex0]->BuildValue()->isa<ValueAny>() &&
      !input_args[kInputIndex0]->BuildValue()->isa<None>()) {
    int64_t shape_m = 1;
    auto input_shape_ptr = reinterpret_cast<int64_t *>(input_shape_tensor->data_c());
    for (auto i = 0; i < shape_v[kInputIndex0]; ++i) {
      if (input_shape_ptr[i] > 0) {
        output_shape.push_back(input_shape_ptr[i]);
        shape_m *= static_cast<int64_t>(input_shape_ptr[i]);
      }
    }
    if (shape_m > max_length) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', the number of elements of output must be less than max length: " << max_length
                               << ", but got " << shape_m
                               << "! The shape of  output must be reduced or max_length must be increased";
    }
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    for (int i = 0; i < shape_v[kInputIndex0]; i++) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(output_shape);
  }
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
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FractionalAvgPoolGrad, prim::kPrimFractionalAvgPoolGrad, AGFractionalAvgPoolGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
