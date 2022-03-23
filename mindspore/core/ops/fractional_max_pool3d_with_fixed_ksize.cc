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

#include "ops/fractional_max_pool3d_with_fixed_ksize.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kDimSize1 = 1;
constexpr size_t kDimSize2 = 2;
constexpr size_t kDimSize3 = 3;
constexpr size_t kDimSize4 = 4;
constexpr size_t kDimSize5 = 5;
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputshapeIndexD = 0;
constexpr size_t kOutputshapeIndexH = 1;
constexpr size_t kOutputshapeIndexW = 2;
constexpr size_t kDimSize4FormatNCDHWIndexN = 0;
constexpr size_t kDimSize4FormatNDHWCIndexC = 3;
constexpr size_t kDimSize5FormatNDHWCIndexN = 0;
constexpr size_t kDimSize5FormatNDHWCIndexC = 4;
constexpr size_t kDimSize5FormatNCDHWIndexN = 0;
constexpr size_t kDimSize5FormatNCDHWIndexC = 1;

void GetAttrs(const PrimitivePtr &primitive, std::vector<float> *ksize, std::vector<int64_t> *output_shape) {
  MS_EXCEPTION_IF_NULL(primitive);
  // attr kize
  MS_EXCEPTION_IF_NULL(primitive->GetAttr("ksize"));
  *ksize = GetValue<std::vector<float>>(primitive->GetAttr("ksize"));
  if (ksize->size() != kDimSize1 && ksize->size() != kDimSize3) {
    MS_EXCEPTION(ValueError) << "ksize of FractionalMaxPool3DWithFixedKsize should be 1 or 3, but got "
                             << ksize->size();
  }
  if (std::any_of(ksize->begin(), ksize->end(), [](float ksize) { return ksize <= 0; })) {
    MS_EXCEPTION(ValueError) << "invalid ksize, ksize must be all positive.";
  }
  // attr output_shape
  MS_EXCEPTION_IF_NULL(primitive->GetAttr("output_shape"));
  *output_shape = GetValue<std::vector<int64_t>>(primitive->GetAttr("output_shape"));
  if (output_shape->size() != kDimSize1 && output_shape->size() != kDimSize3) {
    MS_EXCEPTION(ValueError) << "output_shape of FractionalMaxPool3DWithFixedKsize should be 1 or 3, but got "
                             << output_shape->size();
  }
  if (std::any_of(output_shape->begin(), output_shape->end(), [](int64_t output_shape) { return output_shape <= 0; })) {
    MS_EXCEPTION(ValueError) << "invalid output_shape, output_shape must be all positive.";
  }
}

abstract::TupleShapePtr FractionalMaxPool3DWithFixedKsizeInferShape(const PrimitivePtr &primitive,
                                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto data_format = GetValue<std::string>(primitive->GetAttr(kFormat));
  if (data_format != "NCDHW" && data_format != "NDHWC") {
    MS_EXCEPTION(ValueError) << "data_format is neither NCDHW nor NDHWC." << data_format;
  }
  (void)CheckAndConvertUtils::CheckInteger("input_number", SizeToLong(input_args.size()), kEqual, kInputsNum, op_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  auto random_samples_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack())[kShape];
  if (input_shape.size() != kDimSize4 && input_shape.size() != kDimSize5) {
    MS_EXCEPTION(TypeError) << "input_shape of FractionalMaxPool3DWithFixedKsize "
                               "must be 4 or 5, but got"
                            << input_shape.size();
  }
  if (random_samples_shape.size() != kDimSize3) {
    MS_EXCEPTION(TypeError) << "random_samples_shape of "
                               "FractionalMaxPool3DWithFixedKsize must be 3, but got"
                            << random_samples_shape.size();
  }
  std::vector<float> ksize;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> output_size;
  GetAttrs(primitive, &ksize, &output_shape);
  int64_t outputD = output_shape[kOutputshapeIndexD];
  int64_t outputH = output_shape[kOutputshapeIndexH];
  int64_t outputW = output_shape[kOutputshapeIndexW];

  if (input_shape.size() == kDimSize4) {
    if (data_format == "NCDHW") {
      int64_t c_dim = input_shape[kDimSize4FormatNCDHWIndexN];
      output_size.push_back(c_dim);
      output_size.push_back(outputD);
      output_size.push_back(outputH);
      output_size.push_back(outputW);
    } else {
      int64_t c_dim = input_shape[kDimSize4FormatNDHWCIndexC];
      output_size.push_back(outputD);
      output_size.push_back(outputH);
      output_size.push_back(outputW);
      output_size.push_back(c_dim);
    }
  } else {
    if (data_format == "NCDHW") {
      int64_t n_dim = input_shape[kDimSize5FormatNCDHWIndexN];
      int64_t c_dim = input_shape[kDimSize5FormatNCDHWIndexC];
      output_size.push_back(n_dim);
      output_size.push_back(c_dim);
      output_size.push_back(outputD);
      output_size.push_back(outputH);
      output_size.push_back(outputW);
    } else {
      int64_t n_dim = input_shape[kDimSize5FormatNDHWCIndexN];
      int64_t c_dim = input_shape[kDimSize5FormatNDHWCIndexC];
      output_size.push_back(n_dim);
      output_size.push_back(outputD);
      output_size.push_back(outputH);
      output_size.push_back(outputW);
      output_size.push_back(c_dim);
    }
  }
  if (std::any_of(output_size.begin(), output_size.end(), [](int64_t shp_v) { return shp_v <= 0; })) {
    MS_LOG(EXCEPTION) << "output_size is not valid.";
  }
  if (input_shape.size() == kDimSize4) {
    if (random_samples_shape[0] != input_shape[0]) {
      MS_EXCEPTION(ValueError)
        << "If input_x is 4 dimensional, the first dimension size of input_x and random_samples must be equal.";
    }
    if (random_samples_shape[kDimSize2] != kDimSize3) {
      MS_EXCEPTION(ValueError)
        << "If input_x is 4 dimensional, the second dimension size of random_samples must be equal to 3.";
    }
  } else {
    if (random_samples_shape[0] != input_shape[0]) {
      MS_EXCEPTION(ValueError)
        << "If input_x is 5 dimensional, the first dimension size of input_x and random_samples must be equal.";
    }
    if (random_samples_shape[1] != input_shape[1]) {
      MS_EXCEPTION(ValueError)
        << "If input_x is 5 dimensional, the second dimension size of input_x and random_samples must be equal.";
    }
    if (random_samples_shape[kDimSize2] != kDimSize3) {
      MS_EXCEPTION(ValueError)
        << "If input_x is 5 dimensional, the second dimension size of random_samples must be equal to 3.";
    }
  }
  abstract::ShapePtr output0_shape = std::make_shared<abstract::Shape>(output_size);
  abstract::ShapePtr output1_shape = std::make_shared<abstract::Shape>(output_size);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{output0_shape, output1_shape});
}

TuplePtr FractionalMaxPool3DWithFixedKsizeInferType(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input_number", SizeToLong(input_args.size()), kEqual, kInputsNum, op_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 1);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
  const std::set<TypePtr> random_samples_valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("random_samples", input_args[1]->BuildType(),
                                                   random_samples_valid_types, op_name);
  auto x_dtype = CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), x_valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, kInt64});
}
}  // namespace

MIND_API_BASE_IMPL(FractionalMaxPool3DWithFixedKsize, PrimitiveC, BaseOperator);
AbstractBasePtr FractionalMaxPool3DWithFixedKsizeInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = FractionalMaxPool3DWithFixedKsizeInferType(primitive, input_args);
  auto infer_shape = FractionalMaxPool3DWithFixedKsizeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(FractionalMaxPool3DWithFixedKsize, prim::kPrimFractionalMaxPool3DWithFixedKsize,
                             FractionalMaxPool3DWithFixedKsizeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
