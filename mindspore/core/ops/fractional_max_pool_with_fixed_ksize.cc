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

#include "ops/fractional_max_pool_with_fixed_ksize.h"

#include <string>
#include <algorithm>
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
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
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
constexpr size_t kInputDimSize = 4;
constexpr size_t kRandomSamplesDimSize = 3;
constexpr size_t kRandomSamplesDimIndex2 = 2;
constexpr size_t kRandomSamplesLastDimSize = 2;
constexpr size_t kInputsDimIndex0 = 0;
constexpr size_t kInputsDimIndex1 = 1;
constexpr size_t kInputsDimIndex2 = 2;
constexpr size_t kInputsDimIndex3 = 3;
constexpr size_t kKsizeDimSize1 = 1;
constexpr size_t kKsizeDimSize2 = 2;
constexpr size_t kKsizeIndex0 = 0;
constexpr size_t kKsizeIndex1 = 1;
constexpr size_t kOutputShapeDimSize1 = 1;
constexpr size_t kOutputShapeDimSize2 = 2;
constexpr size_t kOutputShapeIndex0 = 0;
constexpr size_t kOutputShapeIndex1 = 1;
constexpr int kDynamicShape = -1;

void InputCheck(const PrimitivePtr &primitive, const std::vector<int64_t> &x_shape,
                const std::vector<int64_t> &random_samples_shape) {
  auto data_format = GetValue<std::string>(primitive->GetAttr(kFormat));
  if (data_format != "NCHW") {
    MS_EXCEPTION(ValueError) << "data_format must be NCHW, but got " << data_format;
  }

  if (x_shape.size() != kInputDimSize) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolWithFixedKsize, the dimension of input_x must be 4, but got "
                             << x_shape.size();
  }

  if (random_samples_shape.size() != kRandomSamplesDimSize) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolWithFixedKsize, the dimension of random_samples must be 3, "
                             << "but got " << random_samples_shape.size();
  }
  if (random_samples_shape[kRandomSamplesDimIndex2] != kRandomSamplesLastDimSize &&
      random_samples_shape[kRandomSamplesDimIndex2] != kDynamicShape) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolWithFixedKsize, the last dimension size of random_samples must "
                             << "be 2, but got " << random_samples_shape[kRandomSamplesDimIndex2];
  }

  if (x_shape[kInputsDimIndex0] != random_samples_shape[kInputsDimIndex0] &&
      x_shape[kInputsDimIndex0] != kDynamicShape && random_samples_shape[kInputsDimIndex0] != kDynamicShape) {
    MS_EXCEPTION(ValueError) << "The first dimension size of input_x and random_samples must be equal.";
  }
  if (x_shape[kInputsDimIndex1] != random_samples_shape[kInputsDimIndex1] &&
      x_shape[kInputsDimIndex1] != kDynamicShape && random_samples_shape[kInputsDimIndex1] != kDynamicShape) {
    MS_EXCEPTION(ValueError) << "The second dimension size of input_x and random_samples must be equal.";
  }

  auto ksize = GetValue<std::vector<int64_t>>(primitive->GetAttr("ksize"));
  if (std::any_of(ksize.begin(), ksize.end(), [](int64_t ksize) { return ksize <= 0; })) {
    MS_EXCEPTION(ValueError) << "invalid ksize, ksize items must be all positive.";
  }
}
abstract::TupleShapePtr FractionalMaxPoolWithFixedKsizeInferShape(const PrimitivePtr &primitive,
                                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    abstract::ShapePtr output0_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{-1, -1, -1, -1});
    abstract::ShapePtr output1_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{-1, -1, -1, -1});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{output0_shape, output1_shape});
  }
  auto random_samples_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto ksize = GetValue<std::vector<int64_t>>(primitive->GetAttr("ksize"));
  InputCheck(primitive, x_shape, random_samples_shape);
  int64_t ksize_h = 0;
  int64_t ksize_w = 0;
  if (ksize.size() == kKsizeDimSize1) {
    ksize_h = ksize[kKsizeIndex0];
    ksize_w = ksize[kKsizeIndex0];
  } else if (ksize.size() == kKsizeDimSize2) {
    ksize_h = ksize[kKsizeIndex0];
    ksize_w = ksize[kKsizeIndex1];
  } else {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolWithFixedKsize, the dimension of ksize must be 1 or 2, "
                             << "but got " << ksize.size();
  }
  auto output_shape = GetValue<std::vector<int64_t>>(primitive->GetAttr("output_shape"));
  if (std::any_of(output_shape.begin(), output_shape.end(), [](int64_t output_shape) { return output_shape <= 0; })) {
    MS_EXCEPTION(ValueError) << "invalid output_shape, output_shape items must be all positive.";
  }
  int64_t output_h = 0;
  int64_t output_w = 0;
  if (output_shape.size() == kOutputShapeDimSize1) {
    output_h = output_shape[kOutputShapeIndex0];
    output_w = output_shape[kOutputShapeIndex0];
  } else if (output_shape.size() == kOutputShapeDimSize2) {
    output_h = output_shape[kOutputShapeIndex0];
    output_w = output_shape[kOutputShapeIndex1];
  } else {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolWithFixedKsize, the dimension of output_shape must be 1 or 2, "
                             << "but got " << output_shape.size();
  }

  if (output_h + ksize_h - 1 > x_shape[kInputsDimIndex2] &&
      random_samples_shape[kRandomSamplesDimIndex2] != kDynamicShape) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolWithFixedKsize, ksize height [" << ksize_h
                             << "] + output_shape_h [" << output_h << "] too large relative to input height ["
                             << x_shape[kInputsDimIndex2]
                             << "], conflict with the rule: ksize_h + output_shape_h - 1 <= input_h";
  }
  if (output_w + ksize_w - 1 > x_shape[kInputsDimIndex3] &&
      random_samples_shape[kRandomSamplesDimIndex2] != kDynamicShape) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolWithFixedKsize, ksize width [" << ksize_w
                             << "] + output_shape_w [" << output_w << "] too large relative to input width ["
                             << x_shape[kInputsDimIndex3]
                             << "], conflict with the rule: ksize_w + output_shape_w - 1 <= input_w";
  }

  ShapeVector out_shape_vector = {x_shape[kInputsDimIndex0], x_shape[kInputsDimIndex1], output_h, output_w};
  abstract::ShapePtr out_shape = std::make_shared<abstract::Shape>(out_shape_vector);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
}

TuplePtr FractionalMaxPoolWithFixedKsizeInferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();

  const std::set<TypePtr> random_samples_valid_types = {kFloat16, kFloat32, kFloat64};
  auto random_samples_dtype = input_args[1]->BuildType();
  CheckAndConvertUtils::CheckTensorTypeValid("random_samples dtype", random_samples_dtype, random_samples_valid_types,
                                             prim_name);

  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
  auto x_dtype = input_args[0]->BuildType();
  auto y_dtype = CheckAndConvertUtils::CheckTensorTypeValid("input_x dtype", x_dtype, x_valid_types, prim_name);
  TypePtr argmax_dtype = kInt64;
  return std::make_shared<Tuple>(std::vector<TypePtr>{y_dtype, argmax_dtype});
}
}  // namespace

MIND_API_OPERATOR_IMPL(FractionalMaxPoolWithFixedKsize, BaseOperator);
AbstractBasePtr FractionalMaxPoolWithFixedKsizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t inputs_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, inputs_num, primitive->name());

  auto types = FractionalMaxPoolWithFixedKsizeInferType(primitive, input_args);
  auto shapes = FractionalMaxPoolWithFixedKsizeInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

void FractionalMaxPoolWithFixedKsize::Init(const std::vector<int64_t> ksize, const std::vector<int64_t> output_shape,
                                           const std::string data_format) {
  set_ksize(ksize);
  set_output_shape(output_shape);
  set_data_format(data_format);
}

void FractionalMaxPoolWithFixedKsize::set_ksize(const std::vector<int64_t> ksize) {
  (void)this->AddAttr("ksize", api::MakeValue(ksize));
}

void FractionalMaxPoolWithFixedKsize::set_output_shape(const std::vector<int64_t> output_shape) {
  (void)this->AddAttr("output_shape", api::MakeValue(output_shape));
}

void FractionalMaxPoolWithFixedKsize::set_data_format(const std::string data_format) {
  (void)this->AddAttr(kFormat, api::MakeValue(data_format));
}

std::vector<int64_t> FractionalMaxPoolWithFixedKsize::get_ksize() const {
  auto value_ptr = GetAttr("ksize");
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> FractionalMaxPoolWithFixedKsize::get_output_shape() const {
  auto value_ptr = GetAttr("output_shape");
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::string FractionalMaxPoolWithFixedKsize::get_data_format() const { return GetValue<std::string>(GetAttr(kFormat)); }

// AG means auto generated
class MIND_API AGFractionalMaxPoolWithFixedKsizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolWithFixedKsizeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolWithFixedKsizeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPoolWithFixedKsizeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FractionalMaxPoolWithFixedKsize, prim::kPrimFractionalMaxPoolWithFixedKsize,
                                 AGFractionalMaxPoolWithFixedKsizeInfer, false);
}  // namespace ops
}  // namespace mindspore
