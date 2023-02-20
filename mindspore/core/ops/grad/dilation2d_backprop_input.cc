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

#include "ops/grad/dilation2d_backprop_input.h"

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <cmath>
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
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
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
void CheckDilation2DBackpropInputWindowAndStrideShape(const std::string &prim_name, int64_t window_h, int64_t window_w,
                                                      int64_t stride_h, int64_t stride_w) {
  const int64_t wLengthMaxLimit = 255;
  const int64_t wSizeMaxLimit = 512;
  if (window_w < 1 || window_w > wLengthMaxLimit || window_h < 1 || window_h > wLengthMaxLimit ||
      window_h * window_w > wSizeMaxLimit) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", size of window which is equal to (filter-1)*dilation+1"
                                "is not supported, the range of window should be [1, 255] and window_h * window_w "
                                "should be less than or equal to 512, but window_w is "
                             << window_w << " and window_h is" << window_h;
  }
  if (stride_h < 1 || stride_h > wLengthMaxLimit || stride_w < 1 || stride_w > wLengthMaxLimit) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", size of stride is not supported, the range of "
                                "stride should be [1, 255], but stride_h is "
                             << stride_h << " and stride_w is" << stride_w;
  }
}

abstract::ShapePtr Dilation2DBackpropInputInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Get input shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto filter_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto out_backprop_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  // Check inputs' dimension
  const int64_t x_shape_size = 4;
  const int64_t filter_shape_size = 3;
  const int64_t out_backprop_shape_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, x_shape_size,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("filter shape size", SizeToLong(filter_shape.size()), kEqual,
                                           filter_shape_size, primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("out_backprop shape size", SizeToLong(out_backprop_shape.size()), kEqual,
                                           out_backprop_shape_size, primitive->name());
  std::string data_format = GetValue<std::string>(primitive->GetAttr("format"));
  std::string pad_mode = GetValue<std::string>(primitive->GetAttr("pad_mode"));
  ShapeVector stride = GetValue<std::vector<int64_t>>(primitive->GetAttr("stride"));
  ShapeVector dilation = GetValue<std::vector<int64_t>>(primitive->GetAttr("dilation"));
  const uint64_t n_axis = 0;
  const uint64_t shapeIndex1 = 1;
  const uint64_t shapeIndex2 = 2;
  const uint64_t shapeIndex3 = 3;
  auto h_axis = shapeIndex1;
  auto w_axis = shapeIndex2;
  auto c_axis = shapeIndex3;
  if (data_format == "NCHW") {
    c_axis = shapeIndex1;
    h_axis = shapeIndex2;
    w_axis = shapeIndex3;
  }
  // Check filter depth
  if (x_shape[c_axis] != filter_shape[c_axis - 1]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x and filter must have the same depth: " << x_shape[c_axis]
                             << " vs " << filter_shape[c_axis];
  }
  auto x_h = x_shape[h_axis];
  auto x_w = x_shape[w_axis];
  auto stride_h = stride[h_axis];
  auto stride_w = stride[w_axis];
  auto dilation_h = dilation[h_axis];
  auto dilation_w = dilation[w_axis];
  auto filter_h = filter_shape[h_axis - 1];
  auto filter_w = filter_shape[w_axis - 1];
  auto window_h = (filter_h - 1) * dilation_h + 1;
  auto window_w = (filter_w - 1) * dilation_w + 1;
  // Check window,stride and dilation
  CheckDilation2DBackpropInputWindowAndStrideShape(prim_name, window_h, window_w, stride_h, stride_w);
  if (window_w > x_w || window_h > x_h) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", size of window which is equal to (filter-1)*dilation+1 "
                                "should not be bigger than size of x, but (window_h, window_w) is ("
                             << window_h << ", " << window_w << ") and (x_h, x_w) is (" << x_h << ", " << x_w << ")";
  }
  // Check out_backprop
  int64_t out_h = -1;
  int64_t out_w = -1;
  if (pad_mode == "VALID") {
    out_h = static_cast<int64_t>(std::ceil(((x_h * 1.0) - dilation_h * (filter_h - 1)) / stride_h));
    out_w = static_cast<int64_t>(std::ceil(((x_w * 1.0) - dilation_w * (filter_w - 1)) / stride_w));
  } else if (pad_mode == "SAME") {
    out_h = static_cast<int64_t>(std::ceil((x_h * 1.0) / stride_h));
    out_w = static_cast<int64_t>(std::ceil((x_w * 1.0) / stride_w));
  }
  out_h = out_h >= 1 ? out_h : 1L;
  out_w = out_w >= 1 ? out_w : 1L;
  const uint64_t outShapeLength = 4;
  ShapeVector e_out_backprop(outShapeLength);
  e_out_backprop[n_axis] = x_shape[n_axis];
  e_out_backprop[h_axis] = out_h;
  e_out_backprop[w_axis] = out_w;
  e_out_backprop[c_axis] = x_shape[c_axis];
  if (out_backprop_shape != e_out_backprop) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the shape of out_backprop should be ["
                             << e_out_backprop[n_axis] << ", " << e_out_backprop[shapeIndex1] << ", "
                             << e_out_backprop[shapeIndex2] << ", " << e_out_backprop[shapeIndex3] << "], but got ["
                             << out_backprop_shape[n_axis] << ", " << out_backprop_shape[shapeIndex1] << ", "
                             << out_backprop_shape[shapeIndex2] << ", " << out_backprop_shape[shapeIndex3] << "]";
  }
  // Return out_backprop shape
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr Dilation2DBackpropInputInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("filter", input_args[kInputIndex1]->BuildType());
  std::map<std::string, TypePtr> out_backprop_types;
  (void)out_backprop_types.emplace("out_backprop", input_args[kInputIndex2]->BuildType());
  std::set<TypePtr> valid_type = {kUInt8, kUInt16, kUInt32,  kUInt64,  kInt8,   kInt16,
                                  kInt32, kInt64,  kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_type, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(out_backprop_types, valid_type, prim_name);
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(Dilation2DBackpropInput, BaseOperator);
AbstractBasePtr Dilation2DBackpropInputInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = Dilation2DBackpropInputInferType(primitive, input_args);
  auto infer_shape = Dilation2DBackpropInputInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

std::vector<int64_t> Dilation2DBackpropInput::get_stride() const {
  auto value_ptr = GetAttr("stride");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<int64_t> Dilation2DBackpropInput::get_dilation() const {
  auto value_ptr = GetAttr("dilation");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::string Dilation2DBackpropInput::get_pad_mode() const {
  auto value_ptr = GetAttr("pad_mode");
  return GetValue<string>(value_ptr);
}
std::string Dilation2DBackpropInput::get_format() const {
  auto value_ptr = GetAttr("format");
  return GetValue<std::string>(value_ptr);
}

// AG means auto generated
class MIND_API AGDilation2DBackpropInputInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DBackpropInputInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DBackpropInputInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DBackpropInputInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Dilation2DBackpropInput, prim::kPrimDilation2DBackpropInput,
                                 AGDilation2DBackpropInputInfer, false);
}  // namespace ops
}  // namespace mindspore
