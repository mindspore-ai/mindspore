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

#include "ops/max_unpool3d.h"
#include <string>
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MaxUnpool3DInferShapeCompute(const std::string &data_format, const ShapeVector &in_shape,
                                                const std::vector<int64_t> &ksize, const std::vector<int64_t> &strides,
                                                const std::vector<int64_t> &pads,
                                                const std::vector<int64_t> &attr_output_shape,
                                                const std::string &op_name) {
  if (data_format == "NCDHW") {
    int64_t out_d = static_cast<int64_t>((in_shape[kInputIndex2] - 1) * strides[kInputIndex2] - 2 * pads[kInputIndex2] +
                                         ksize[kInputIndex2]);
    int64_t out_h = static_cast<int64_t>((in_shape[kInputIndex3] - 1) * strides[kInputIndex3] - 2 * pads[kInputIndex3] +
                                         ksize[kInputIndex3]);
    int64_t out_w = static_cast<int64_t>((in_shape[kInputIndex4] - 1) * strides[kInputIndex4] - 2 * pads[kInputIndex4] +
                                         ksize[kInputIndex4]);
    std::vector<int64_t> out_shape = {in_shape[kInputIndex0], in_shape[kInputIndex1], out_d, out_h, out_w};
    if (attr_output_shape.size() == kDim5) {
      (void)CheckAndConvertUtils::CheckInteger("output_shape[0]", attr_output_shape[kInputIndex0], kEqual,
                                               in_shape[kInputIndex0], op_name);
      (void)CheckAndConvertUtils::CheckInteger("output_shape[1]", attr_output_shape[kInputIndex1], kEqual,
                                               in_shape[kInputIndex1], op_name);
      auto min_size_d = out_d - strides[kInputIndex2];
      auto max_size_d = out_d + strides[kInputIndex2];
      auto min_size_h = out_h - strides[kInputIndex3];
      auto max_size_h = out_h + strides[kInputIndex3];
      auto min_size_w = out_w - strides[kInputIndex4];
      auto max_size_w = out_w + strides[kInputIndex4];
      if ((min_size_d < attr_output_shape[kInputIndex2] && attr_output_shape[kInputIndex2] < max_size_d) &&
          (min_size_h < attr_output_shape[kInputIndex3] && attr_output_shape[kInputIndex3] < max_size_h) &&
          (min_size_w < attr_output_shape[kInputIndex4] && attr_output_shape[kInputIndex4] < max_size_w)) {
        out_shape = attr_output_shape;
      } else {
        std::vector<int64_t> max_output_shape = {in_shape[kInputIndex0], in_shape[kInputIndex1], max_size_d, max_size_h,
                                                 max_size_w};
        std::vector<int64_t> min_output_shape = {in_shape[kInputIndex0], in_shape[kInputIndex1], min_size_d, min_size_h,
                                                 min_size_w};
        MS_EXCEPTION(ValueError) << "MaxUnpool3D: The dim 2, 3, 4 of output_shape : " << attr_output_shape
                                 << " must be between " << min_output_shape << " and " << max_output_shape << "."
                                 << std::endl;
      }
    }
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    int64_t out_d = static_cast<int64_t>((in_shape[kInputIndex1] - 1) * strides[kInputIndex1] - 2 * pads[kInputIndex1] +
                                         ksize[kInputIndex1]);
    int64_t out_h = static_cast<int64_t>((in_shape[kInputIndex2] - 1) * strides[kInputIndex2] - 2 * pads[kInputIndex2] +
                                         ksize[kInputIndex2]);
    int64_t out_w = static_cast<int64_t>((in_shape[kInputIndex3] - 1) * strides[kInputIndex3] - 2 * pads[kInputIndex3] +
                                         ksize[kInputIndex3]);
    std::vector<int64_t> out_shape = {in_shape[kInputIndex0], out_d, out_h, out_w, in_shape[kInputIndex4]};
    if (attr_output_shape.size() == kDim5) {
      (void)CheckAndConvertUtils::CheckInteger("output_shape[0]", attr_output_shape[kInputIndex0], kEqual,
                                               in_shape[kInputIndex0], op_name);
      (void)CheckAndConvertUtils::CheckInteger("output_shape[4]", attr_output_shape[kInputIndex4], kEqual,
                                               in_shape[kInputIndex4], op_name);
      auto min_size_d = out_d - strides[kInputIndex1];
      auto max_size_d = out_d + strides[kInputIndex1];
      auto min_size_h = out_h - strides[kInputIndex2];
      auto max_size_h = out_h + strides[kInputIndex2];
      auto min_size_w = out_w - strides[kInputIndex3];
      auto max_size_w = out_w + strides[kInputIndex3];
      if ((min_size_d < attr_output_shape[kInputIndex1] && attr_output_shape[kInputIndex1] < max_size_d) &&
          (min_size_h < attr_output_shape[kInputIndex2] && attr_output_shape[kInputIndex2] < max_size_h) &&
          (min_size_w < attr_output_shape[kInputIndex3] && attr_output_shape[kInputIndex3] < max_size_w)) {
        out_shape = attr_output_shape;
      } else {
        std::vector<int64_t> max_output_shape = {in_shape[kInputIndex0], max_size_d, max_size_h, max_size_w,
                                                 in_shape[kInputIndex4]};
        std::vector<int64_t> min_output_shape = {in_shape[kInputIndex0], min_size_d, min_size_h, min_size_w,
                                                 in_shape[kInputIndex4]};
        MS_EXCEPTION(ValueError) << "MaxUnpool3D: The dim 1, 2, 3 of output_shape : " << attr_output_shape
                                 << " must be between " << min_output_shape << " and " << max_output_shape << "."
                                 << std::endl;
      }
    }
    return std::make_shared<abstract::Shape>(out_shape);
  }
}

abstract::ShapePtr MaxUnpool3DInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShapeTrack())[kShape];
  auto argmax_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShapeTrack())[kShape];
  auto data_format = GetValue<std::string>(primitive->GetAttr("format"));
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, SizeToLong(kDim5), op_name);
  (void)CheckAndConvertUtils::CheckInteger("argmax_rank", SizeToLong(argmax_shape.size()), kEqual, SizeToLong(kDim5),
                                           op_name);
  CheckAndConvertUtils::Check("x_shape", in_shape, kEqual, argmax_shape, op_name, ValueError);
  auto ksize = GetValue<std::vector<int64_t>>(primitive->GetAttr("ksize"));
  auto strides = GetValue<std::vector<int64_t>>(primitive->GetAttr("strides"));
  auto pads = GetValue<std::vector<int64_t>>(primitive->GetAttr("pads"));
  auto attr_output_shape = GetValue<std::vector<int64_t>>(primitive->GetAttr("output_shape"));
  (void)CheckAndConvertUtils::CheckInteger("ksize_rank", SizeToLong(ksize.size()), kEqual, SizeToLong(kDim5), op_name);
  (void)CheckAndConvertUtils::CheckInteger("strides_rank", SizeToLong(strides.size()), kEqual, SizeToLong(kDim5),
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("pads_rank", SizeToLong(pads.size()), kEqual, SizeToLong(kDim5), op_name);

  if (attr_output_shape.size() != kDim5 && attr_output_shape.size() != kDim0) {
    MS_EXCEPTION(ValueError) << "MaxUnpool3D: Output_shape size must be 0 or 5.";
  }

  return MaxUnpool3DInferShapeCompute(data_format, in_shape, ksize, strides, pads, attr_output_shape, op_name);
}

TypePtr MaxUnpool3DInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
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

MIND_API_OPERATOR_IMPL(MaxUnpool3D, BaseOperator);
AbstractBasePtr MaxUnpool3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MaxUnpool3DInferType(primitive, input_args);
  auto infer_shape = MaxUnpool3DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
std::string MaxUnpool3D::get_format() const { return GetValue<std::string>(GetAttr(kFormat)); }

REGISTER_PRIMITIVE_EVAL_IMPL(MaxUnpool3D, prim::kPrimMaxUnpool3D, MaxUnpool3DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
