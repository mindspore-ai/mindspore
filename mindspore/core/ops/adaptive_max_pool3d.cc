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

#include "ops/adaptive_max_pool3d.h"

#include <algorithm>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputDims4 = 4;
constexpr int64_t kInputDims5 = 5;
constexpr int64_t kOutputSizeNumElem = 3;

abstract::TupleShapePtr AdaptiveMaxPool3DInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  std::shared_ptr<mindspore::abstract::Shape> out_shape_ptr;
  if (x_shape_ptr->IsDimUnknown()) {
    ShapeVector out_shape = {abstract::Shape::kShapeRankAny};
    out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  auto output_size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (IsDynamic(output_size_shape)) {
    std::vector<int64_t> out_shape = x_shape;
    for (int64_t i = out_shape.size() - kOutputSizeNumElem; i < SizeToLong(out_shape.size()); ++i) {
      out_shape[i] = abstract::Shape::kShapeDimAny;
    }
    out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
  }
  const int64_t input_num_dims = SizeToLong(x_shape.size());
  const int64_t output_size_dim = SizeToLong(output_size_shape.size());
  CheckAndConvertUtils::CheckInRange("rank of x", input_num_dims, kIncludeBoth, {kInputDims4, kInputDims5}, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rank of output_size", output_size_dim, kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("size of output_size", output_size_shape[0], kEqual, kOutputSizeNumElem,
                                           prim_name);

  auto output_size_value = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(output_size_value);
  if (input_args[1]->isa<abstract::AbstractTensor>() && !output_size_value->isa<None>() &&
      !output_size_value->isa<ValueAny>()) {
    auto output_size = CheckAndConvertUtils::CheckTensorIntValue("output_size", output_size_value, prim_name);

    ShapeVector out_shape = x_shape;
    for (int64_t i = 1; i <= kOutputSizeNumElem; ++i) {
      if (output_size[kOutputSizeNumElem - i] <= 0) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', 'output_size' should be a vector with all positive item, but got "
                                 << ShapeVectorToStr(output_size) << ".";
      }
      out_shape[input_num_dims - i] = output_size[kOutputSizeNumElem - i];
    }
    out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
  } else {
    const size_t kDHWDims = 3;
    std::vector<int64_t> out_shape = x_shape;
    for (int64_t i = out_shape.size() - kDHWDims; i < SizeToLong(out_shape.size()); ++i) {
      out_shape[i] = abstract::Shape::kShapeDimAny;
    }
    out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
  }

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
}

TuplePtr AdaptiveMaxPool3DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_dtype = input_args[0]->BuildType();
  auto output_size_dtype = input_args[1]->BuildType();
  const std::set<TypePtr> x_valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                           kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> output_size_valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, x_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("output_size", output_size_dtype, output_size_valid_types,
                                                   prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, output_size_dtype});
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveMaxPool3D, BaseOperator);
AbstractBasePtr AdaptiveMaxPool3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveMaxPool3DInferType(primitive, input_args);
  auto shapes = AdaptiveMaxPool3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGAdaptiveMaxPool3DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool3DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool3DInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AdaptiveMaxPool3DInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdaptiveMaxPool3D, prim::kPrimAdaptiveMaxPool3D, AGAdaptiveMaxPool3DInfer, false);
}  // namespace ops
}  // namespace mindspore
