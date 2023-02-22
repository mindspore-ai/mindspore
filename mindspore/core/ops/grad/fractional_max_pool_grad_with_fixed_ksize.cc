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

#include "ops/grad/fractional_max_pool_grad_with_fixed_ksize.h"

#include <string>
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
#include "ir/primitive.h"
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
constexpr size_t kInputsIndex0 = 0;
constexpr size_t kInputsIndex1 = 1;
constexpr size_t kInputsIndex2 = 2;
constexpr size_t kInputsIndex3 = 3;
constexpr size_t kInputsDimSize = 4;
constexpr size_t kInputIndexN = 0;
constexpr size_t kInputIndexC = 1;
constexpr int dyShape = -1;

std::vector<bool> InputDynamic(const std::vector<int64_t> &out_backprop_shape_,
                               const std::vector<int64_t> &argmax_shape_,
                               const std::vector<int64_t> &origin_input_shape_, bool out_backprop_shape_dy_,
                               bool argmax_shape_dy_, bool origin_input_shape_dy_) {
  std::vector<bool> dynamic_shape;
  out_backprop_shape_dy_ =
    out_backprop_shape_[kInputsIndex0] != dyShape && out_backprop_shape_[kInputsIndex1] != dyShape &&
    out_backprop_shape_[kInputsIndex2] != dyShape && out_backprop_shape_[kInputsIndex3] != dyShape;
  dynamic_shape.push_back(out_backprop_shape_dy_);
  argmax_shape_dy_ = argmax_shape_[kInputsIndex0] != dyShape && argmax_shape_[kInputsIndex1] != dyShape &&
                     argmax_shape_[kInputsIndex2] != dyShape && argmax_shape_[kInputsIndex3] != dyShape;
  dynamic_shape.push_back(argmax_shape_dy_);
  origin_input_shape_dy_ =
    origin_input_shape_[kInputsIndex0] != dyShape && origin_input_shape_[kInputsIndex1] != dyShape &&
    origin_input_shape_[kInputsIndex2] != dyShape && origin_input_shape_[kInputsIndex3] != dyShape;
  dynamic_shape.push_back(origin_input_shape_dy_);
  return dynamic_shape;
}
abstract::ShapePtr FractionalMaxPoolGradWithFixedKsizeInferShape(const PrimitivePtr &primitive,
                                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto data_format = GetValue<std::string>(primitive->GetAttr(kFormat));
  if (data_format != "NCHW") {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, attr data_format must be NCHW.";
  }

  auto origin_input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputsIndex0]->BuildShape())[kShape];
  if (IsDynamicRank(origin_input_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-1, -1, -1, -1});
  }
  auto out_backprop_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputsIndex1]->BuildShape())[kShape];
  auto argmax_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputsIndex2]->BuildShape())[kShape];
  if (origin_input_shape.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the dimension of origin_input must be 4.";
  }
  if (out_backprop_shape.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the dimension of out_backprop must be 4.";
  }
  if (argmax_shape.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the dimension of argmax must be 4.";
  }
  bool out_backprop_shape_dy = false;
  bool argmax_shape_dy = false;
  bool origin_input_shape_dy = false;
  std::vector<bool> shape_dy = InputDynamic(out_backprop_shape, argmax_shape, origin_input_shape, out_backprop_shape_dy,
                                            argmax_shape_dy, origin_input_shape_dy);
  for (size_t i = 0; i < kInputsDimSize; i++) {
    if (out_backprop_shape[i] != argmax_shape[i] && shape_dy[kInputsIndex0] && shape_dy[kInputsIndex1]) {
      MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, out_backprop and argmax must have "
                               << "the same shape.";
    }
  }
  if (origin_input_shape[kInputIndexN] != out_backprop_shape[kInputIndexN] && shape_dy[kInputsIndex0] &&
      shape_dy[kInputsIndex2]) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the first dimension size of three inputs "
                             << "must be equal.";
  }
  if (origin_input_shape[kInputIndexC] != out_backprop_shape[kInputIndexC] && shape_dy[kInputsIndex0] &&
      shape_dy[kInputsIndex2]) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the second dimension size of three inputs "
                             << "must be equal.";
  }
  return std::make_shared<abstract::Shape>(origin_input_shape);
}

TypePtr FractionalMaxPoolGradWithFixedKsizeInferType(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();

  const std::set<TypePtr> out_backprop_valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
  const std::set<TypePtr> argmax_valid_types = {kInt64};
  CheckAndConvertUtils::CheckTensorTypeValid("origin_input dtype", input_args[kInputsIndex0]->BuildType(),
                                             out_backprop_valid_types, prim_name);
  CheckAndConvertUtils::CheckTensorTypeValid("argmax dtype", input_args[kInputsIndex2]->BuildType(), argmax_valid_types,
                                             prim_name);
  auto y_dtype = CheckAndConvertUtils::CheckTensorTypeValid(
    "out_backprop dtype", input_args[kInputsIndex1]->BuildType(), out_backprop_valid_types, prim_name);
  return std::make_shared<TensorType>(y_dtype);
}
}  // namespace

MIND_API_OPERATOR_IMPL(FractionalMaxPoolGradWithFixedKsize, BaseOperator);
AbstractBasePtr FractionalMaxPoolGradWithFixedKsizeInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t inputs_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, inputs_num, primitive->name());

  auto infer_shape = FractionalMaxPoolGradWithFixedKsizeInferShape(primitive, input_args);
  auto infer_type = FractionalMaxPoolGradWithFixedKsizeInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

void FractionalMaxPoolGradWithFixedKsize::Init(const std::string data_format) { set_data_format(data_format); }

void FractionalMaxPoolGradWithFixedKsize::set_data_format(const std::string data_format) {
  (void)this->AddAttr(kFormat, api::MakeValue(data_format));
}

std::string FractionalMaxPoolGradWithFixedKsize::get_data_format() const {
  return GetValue<std::string>(GetAttr(kFormat));
}

// AG means auto generated
class MIND_API AGFractionalMaxPoolGradWithFixedKsizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolGradWithFixedKsizeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolGradWithFixedKsizeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolGradWithFixedKsizeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FractionalMaxPoolGradWithFixedKsize, prim::kPrimFractionalMaxPoolGradWithFixedKsize,
                                 AGFractionalMaxPoolGradWithFixedKsizeInfer, false);
}  // namespace ops
}  // namespace mindspore
