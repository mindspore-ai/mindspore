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

#include "ops/max_unpool2d.h"
#include <string>
#include <set>
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MaxUnpool2DInferShapeCompute(const std::string &data_format, const ShapeVector &in_shape,
                                                const std::vector<int64_t> &ksize, const std::vector<int64_t> &strides,
                                                const std::vector<int64_t> &pads,
                                                const std::vector<int64_t> &attr_output_shape,
                                                const std::string &op_name) {
  if (data_format == "NCHW") {
    int64_t out_h = static_cast<int64_t>((in_shape[kInputIndex2] - 1) * strides[kInputIndex2] - 2 * pads[kInputIndex2] +
                                         ksize[kInputIndex2]);
    int64_t out_w = static_cast<int64_t>((in_shape[kInputIndex3] - 1) * strides[kInputIndex3] - 2 * pads[kInputIndex3] +
                                         ksize[kInputIndex3]);
    std::vector<int64_t> out_shape = {in_shape[kInputIndex0], in_shape[kInputIndex1], out_h, out_w};
    if (attr_output_shape.size() == kDim4) {
      (void)CheckAndConvertUtils::CheckInteger("output_shape[0]", attr_output_shape[kInputIndex0], kEqual,
                                               in_shape[kInputIndex0], op_name);
      (void)CheckAndConvertUtils::CheckInteger("output_shape[1]", attr_output_shape[kInputIndex1], kEqual,
                                               in_shape[kInputIndex1], op_name);
      auto min_size_h = out_h - strides[kInputIndex2];
      auto max_size_h = out_h + strides[kInputIndex2];
      auto min_size_w = out_w - strides[kInputIndex3];
      auto max_size_w = out_w + strides[kInputIndex3];

      if ((min_size_h < attr_output_shape[kInputIndex2] && attr_output_shape[kInputIndex2] < max_size_h) &&
          (min_size_w < attr_output_shape[kInputIndex3] && attr_output_shape[kInputIndex3] < max_size_w)) {
        out_shape = attr_output_shape;
      } else {
        std::vector<int64_t> max_output_shape = {in_shape[kInputIndex0], in_shape[kInputIndex1], max_size_h,
                                                 max_size_w};
        std::vector<int64_t> min_output_shape = {in_shape[kInputIndex0], in_shape[kInputIndex1], min_size_h,
                                                 min_size_w};
        MS_EXCEPTION(ValueError) << "MaxUnpool2D: The dim 2, 3 of output_shape : " << attr_output_shape
                                 << " must be between " << min_output_shape << " and " << max_output_shape << "."
                                 << std::endl;
      }
    }
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    int64_t out_h = static_cast<int64_t>((in_shape[kInputIndex1] - 1) * strides[kInputIndex1] - 2 * pads[kInputIndex1] +
                                         ksize[kInputIndex1]);
    int64_t out_w = static_cast<int64_t>((in_shape[kInputIndex2] - 1) * strides[kInputIndex2] - 2 * pads[kInputIndex2] +
                                         ksize[kInputIndex2]);
    std::vector<int64_t> out_shape = {in_shape[kInputIndex0], out_h, out_w, in_shape[kInputIndex3]};

    if (attr_output_shape.size() == kDim4) {
      (void)CheckAndConvertUtils::CheckInteger("output_shape[0]", attr_output_shape[kInputIndex0], kEqual,
                                               in_shape[kInputIndex0], op_name);
      (void)CheckAndConvertUtils::CheckInteger("output_shape[3]", attr_output_shape[kInputIndex3], kEqual,
                                               in_shape[kInputIndex3], op_name);
      auto min_size_h = out_h - strides[kInputIndex1];
      auto max_size_h = out_h + strides[kInputIndex1];
      auto min_size_w = out_w - strides[kInputIndex2];
      auto max_size_w = out_w + strides[kInputIndex2];
      if ((min_size_h < attr_output_shape[kInputIndex1] && attr_output_shape[kInputIndex1] < max_size_h) &&
          (min_size_w < attr_output_shape[kInputIndex2] && attr_output_shape[kInputIndex2] < max_size_w)) {
        out_shape = attr_output_shape;
      } else {
        std::vector<int64_t> max_output_shape = {in_shape[kInputIndex0], max_size_h, max_size_w,
                                                 in_shape[kInputIndex3]};
        std::vector<int64_t> min_output_shape = {in_shape[kInputIndex0], min_size_h, min_size_w,
                                                 in_shape[kInputIndex3]};
        MS_EXCEPTION(ValueError) << "MaxUnpool2D: The dim 1, 2 of output_shape : " << attr_output_shape
                                 << " must be between " << min_output_shape << " and " << max_output_shape << "."
                                 << std::endl;
      }
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }
}

abstract::ShapePtr MaxUnpool2DInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShapeTrack())[kShape];
  auto argmax_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShapeTrack())[kShape];
  auto data_format = GetValue<std::string>(primitive->GetAttr("format"));
  auto attr_output_shape = GetValue<std::vector<int64_t>>(primitive->GetAttr("output_shape"));
  if (attr_output_shape.size() != kDim4 && attr_output_shape.size() != kDim0) {
    MS_EXCEPTION(ValueError) << "MaxUnpool2D: Output_shape size must be 0 or 4.";
  }

  if (IsDynamic(in_shape)) {
    if (attr_output_shape.size() == kDim4) {
      return std::make_shared<abstract::Shape>(attr_output_shape);
    }

    std::vector<int64_t> out_shape = {-1, -1, -1, -1};
    if (IsDynamicRank(in_shape)) {
      return std::make_shared<abstract::Shape>(out_shape);
    }

    (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, SizeToLong(kDim4), op_name);
    if (data_format == "NCHW") {
      out_shape = {in_shape[kInputIndex0], in_shape[kInputIndex1], -1, -1};
    } else {
      out_shape = {in_shape[kInputIndex0], -1, -1, in_shape[kInputIndex3]};
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }

  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, SizeToLong(kDim4), op_name);
  (void)CheckAndConvertUtils::CheckInteger("argmax_rank", SizeToLong(argmax_shape.size()), kEqual, SizeToLong(kDim4),
                                           op_name);
  CheckAndConvertUtils::Check("x_shape", in_shape, kEqual, argmax_shape, op_name, ValueError);

  auto ksize = GetValue<std::vector<int64_t>>(primitive->GetAttr("ksize"));
  auto strides = GetValue<std::vector<int64_t>>(primitive->GetAttr("strides"));
  auto pads = GetValue<std::vector<int64_t>>(primitive->GetAttr("pads"));

  (void)CheckAndConvertUtils::CheckInteger("ksize_rank", SizeToLong(ksize.size()), kEqual, SizeToLong(kDim4), op_name);
  (void)CheckAndConvertUtils::CheckInteger("strides_rank", SizeToLong(strides.size()), kEqual, SizeToLong(kDim4),
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("pads_rank", SizeToLong(pads.size()), kEqual, SizeToLong(kDim4), op_name);

  return MaxUnpool2DInferShapeCompute(data_format, in_shape, ksize, strides, pads, attr_output_shape, op_name);
}

TypePtr MaxUnpool2DInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> argmax_valid_types = {kInt32, kInt64};
  auto input_x_type = input_args[kInputIndex0]->BuildType();
  auto argmax_type = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, common_valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax", argmax_type, argmax_valid_types, prim->name());
  return input_x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(MaxUnpool2D, BaseOperator);
AbstractBasePtr MaxUnpool2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MaxUnpool2DInferType(primitive, input_args);
  auto infer_shape = MaxUnpool2DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

std::string MaxUnpool2D::get_format() const {
  auto value_ptr = GetAttr("format");
  return GetValue<std::string>(value_ptr);
}
REGISTER_PRIMITIVE_EVAL_IMPL(MaxUnpool2D, prim::kPrimMaxUnpool2D, MaxUnpool2DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
