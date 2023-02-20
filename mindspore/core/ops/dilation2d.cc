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

#include "ops/dilation2d.h"

#include <set>
#include <cmath>
#include <map>
#include <memory>

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
#include "mindapi/base/format.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void CheckDilation2DShapeAnyAndPositive(const std::string &op, const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if ((shape[i] < 0) && (shape[i] != abstract::Shape::kShapeDimAny)) {
      MS_EXCEPTION(ValueError) << op << " shape element [" << i
                               << "] must be positive integer or kShapeDimAny, but got " << shape[i];
    }
  }
}

abstract::ShapePtr Dilation2DInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto filter_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto filter_shape = filter_shape_map[kShape];
  const uint64_t n_axis = 0;
  ShapeVector output_shape = {x_shape[n_axis], -1, -1, -1};
  if (IsDynamic(x_shape) || IsDynamic(filter_shape)) {
    return std::make_shared<abstract::Shape>(output_shape);
  }
  const int64_t x_shape_size = 4;
  const int64_t filter_shape_size = 3;
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, x_shape_size,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("filter shape size", SizeToLong(filter_shape.size()), kEqual,
                                           filter_shape_size, primitive->name());
  const uint64_t shapeIndex1 = 1;
  const uint64_t shapeIndex2 = 2;
  const uint64_t shapeIndex3 = 3;
  uint64_t h_axis = shapeIndex1;
  uint64_t w_axis = shapeIndex2;
  uint64_t c_axis = shapeIndex3;
  Format data_format = Format(CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format")));
  if (data_format == Format::NCHW) {
    c_axis = shapeIndex1;
    h_axis = shapeIndex2;
    w_axis = shapeIndex3;
  }

  std::string pad_mode = GetValue<std::string>(primitive->GetAttr("pad_mode"));
  std::vector<int64_t> kernel_size{filter_shape[h_axis - 1], filter_shape[w_axis - 1]};
  int64_t depth = filter_shape[c_axis - 1];
  std::vector<int64_t> stride = GetValue<std::vector<int64_t>>(primitive->GetAttr("stride"));
  std::vector<int64_t> dilation = GetValue<std::vector<int64_t>>(primitive->GetAttr("dilation"));

  if (filter_shape[c_axis - 1] != x_shape[c_axis]) {
    MS_EXCEPTION(ValueError)
      << "For Dilation2D, the C dim value of `x` shape and `filter` shape must be equal, but got x_shape: " << x_shape
      << ", filter_shape: " << filter_shape;
  }
  std::vector<int64_t> output_hw;
  if (pad_mode == "VALID") {
    output_hw.push_back(static_cast<int64_t>(
      std::ceil(((x_shape[h_axis] * 1.0) - dilation[h_axis] * (kernel_size[0] - 1)) / stride[h_axis])));
    output_hw.push_back(static_cast<int64_t>(
      std::ceil(((x_shape[w_axis] * 1.0) - dilation[w_axis] * (kernel_size[1] - 1)) / stride[w_axis])));
  } else if (pad_mode == "SAME") {
    output_hw.push_back(static_cast<int64_t>(std::ceil((x_shape[h_axis] * 1.0) / stride[h_axis])));
    output_hw.push_back(static_cast<int64_t>(std::ceil((x_shape[w_axis] * 1.0) / stride[w_axis])));
  }
  if (data_format == Format::NHWC) {
    output_shape = {x_shape[n_axis], output_hw[0], output_hw[1], depth};
  } else {
    output_shape = {x_shape[n_axis], depth, output_hw[0], output_hw[1]};
  }
  CheckDilation2DShapeAnyAndPositive(prim_name + " output_shape", output_shape);
  if (data_format == Format::NHWC) {
    output_shape = {x_shape[n_axis], output_hw[0], output_hw[1], depth};
  } else {
    output_shape = {x_shape[n_axis], depth, output_hw[0], output_hw[1]};
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr Dilation2DInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kUInt16, kUInt32,
                                         kUInt64,  kInt8,    kInt16,   kInt32, kInt64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("filter", input_args[kInputIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(Dilation2D, BaseOperator);
AbstractBasePtr Dilation2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = Dilation2DInferType(primitive, input_args);
  auto infer_shape = Dilation2DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

std::vector<int64_t> Dilation2D::get_stride() const {
  auto value_ptr = GetAttr("stride");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<int64_t> Dilation2D::get_dilation() const {
  auto value_ptr = GetAttr("dilation");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::string Dilation2D::get_pad_mode() const {
  auto value_ptr = GetAttr("pad_mode");
  return GetValue<string>(value_ptr);
}
std::string Dilation2D::get_format() const {
  auto value_ptr = GetAttr("format");
  return GetValue<std::string>(value_ptr);
}

// AG means auto generated
class MIND_API AGDilation2DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return Dilation2DInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Dilation2D, prim::kPrimDilation2D, AGDilation2DInfer, false);
}  // namespace ops
}  // namespace mindspore
