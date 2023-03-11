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

#include "ops/grad/dilation2d_backprop_filter.h"

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
void CheckDilation2DBackpropFilterWindowAndStrideShape(const std::string &prim_name, int64_t window_h, int64_t window_w,
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

abstract::ShapePtr Dilation2DBackpropFilterInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto filter_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto out_backprop_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(filter_shape) || IsDynamicRank(out_backprop_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
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
  if (IsDynamicShape(x_shape) || IsDynamicShape(filter_shape) || IsDynamicShape(out_backprop_shape)) {
    return std::make_shared<abstract::Shape>(
      ShapeVector({abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny}));
  }
  // Get Attributes
  std::string data_format = GetValue<std::string>(primitive->GetAttr("format"));
  std::string pad_mode = GetValue<std::string>(primitive->GetAttr("pad_mode"));
  std::vector<int64_t> stride = GetValue<std::vector<int64_t>>(primitive->GetAttr("stride"));
  std::vector<int64_t> dilation = GetValue<std::vector<int64_t>>(primitive->GetAttr("dilation"));
  // Convert NHWC to NCHW
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
  // Check input depth
  if (x_shape[c_axis] != filter_shape[c_axis - 1]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", x and filter must have the same depth: " << x_shape[c_axis]
                             << " , but got " << filter_shape[c_axis - 1];
  }
  auto x_h = x_shape[h_axis];
  auto x_w = x_shape[w_axis];
  auto depth = x_shape[c_axis];
  auto stride_h = stride[h_axis];
  auto stride_w = stride[w_axis];
  auto dilation_h = dilation[h_axis];
  auto dilation_w = dilation[w_axis];
  auto filter_h = filter_shape[h_axis - 1];
  auto filter_w = filter_shape[w_axis - 1];
  auto window_h = (filter_h - 1) * dilation_h + 1;
  auto window_w = (filter_w - 1) * dilation_w + 1;
  // Check window,stride and dilation
  CheckDilation2DBackpropFilterWindowAndStrideShape(prim_name, window_h, window_w, stride_h, stride_w);
  if (window_w > x_w || window_h > x_h) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", size of window which is equal to (filter-1)*dilation+1 "
                                "should not be bigger than size of x, but (window_h, window_w) is ("
                             << window_h << ", " << window_w << ") and (x_h, x_w) is (" << x_h << ", " << x_w << ")";
  }
  // Check out_backprop shape
  std::vector<int64_t> out_backprop_expect_shape;
  if (pad_mode == "VALID") {
    out_backprop_expect_shape.push_back(
      static_cast<int64_t>(std::ceil(((x_h * 1.0) - dilation_h * (filter_h - 1)) / stride_h)));
    out_backprop_expect_shape.push_back(
      static_cast<int64_t>(std::ceil(((x_w * 1.0) - dilation_w * (filter_w - 1)) / stride_w)));
  } else if (pad_mode == "SAME") {
    out_backprop_expect_shape.push_back(static_cast<int64_t>(std::ceil((x_h * 1.0) / stride_h)));
    out_backprop_expect_shape.push_back(static_cast<int64_t>(std::ceil((x_w * 1.0) / stride_w)));
  }
  ShapeVector out_backprop_expect_shape_;
  if (data_format == "NHWC") {
    out_backprop_expect_shape_ = {x_shape[n_axis], out_backprop_expect_shape[0], out_backprop_expect_shape[1], depth};
  } else {
    out_backprop_expect_shape_ = {x_shape[n_axis], depth, out_backprop_expect_shape[0], out_backprop_expect_shape[1]};
  }
  if (out_backprop_expect_shape_ != out_backprop_shape) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the shape of out_backprop should be ["
                             << out_backprop_expect_shape_[n_axis] << ", " << out_backprop_expect_shape_[shapeIndex1]
                             << ", " << out_backprop_expect_shape_[shapeIndex2] << ", "
                             << out_backprop_expect_shape_[shapeIndex3] << "], but got [" << out_backprop_shape[n_axis]
                             << ", " << out_backprop_shape[shapeIndex1] << ", " << out_backprop_shape[shapeIndex2]
                             << ", " << out_backprop_shape[shapeIndex3] << "]";
  }
  // Return out_backprop shape
  return std::make_shared<abstract::Shape>(filter_shape);
}

TypePtr Dilation2DBackpropFilterInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
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

MIND_API_OPERATOR_IMPL(Dilation2DBackpropFilter, BaseOperator);
AbstractBasePtr Dilation2DBackpropFilterInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = Dilation2DBackpropFilterInferType(primitive, input_args);
  auto infer_shape = Dilation2DBackpropFilterInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

std::vector<int64_t> Dilation2DBackpropFilter::get_stride() const {
  auto value_ptr = GetAttr("stride");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<int64_t> Dilation2DBackpropFilter::get_dilation() const {
  auto value_ptr = GetAttr("dilation");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::string Dilation2DBackpropFilter::get_pad_mode() const {
  auto value_ptr = GetAttr("pad_mode");
  return GetValue<string>(value_ptr);
}
std::string Dilation2DBackpropFilter::get_format() const {
  auto value_ptr = GetAttr("format");
  return GetValue<std::string>(value_ptr);
}

// AG means auto generated
class MIND_API AGDilation2DBackpropFilterInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DBackpropFilterInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DBackpropFilterInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DBackpropFilterInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Dilation2DBackpropFilter, prim::kPrimDilation2DBackpropFilter,
                                 AGDilation2DBackpropFilterInfer, false);
}  // namespace ops
}  // namespace mindspore
