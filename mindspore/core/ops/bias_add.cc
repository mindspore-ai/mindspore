/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/bias_add.h"
#include <map>
#include <string>
#include <vector>
#include <set>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/ms_context.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t x_min_rank = 2;
constexpr int64_t x_max_rank = 5;
inline bool IsNHWCorNCHW(int64_t data_format) {
  return (data_format == static_cast<int64_t>(Format::NHWC) || data_format == static_cast<int64_t>(Format::NCHW));
}
inline bool IsNCHWorNCDHW(int64_t data_format) {
  return (data_format == static_cast<int64_t>(Format::NCHW) || data_format == static_cast<int64_t>(Format::NCDHW));
}
inline bool IsShapeSizeOutOfRange(int64_t size) { return (size > x_max_rank || size < x_min_rank); }

abstract::ShapePtr BiasAddInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto bias = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(bias);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_shape = shape_map[kShape];
  auto bias_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (IsDynamicRank(input_shape) || IsDynamicRank(bias_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }

  (void)CheckAndConvertUtils::CheckInteger("bias rank", SizeToLong(bias_shape.size()), kEqual, 1, prim_name);
  const int64_t x_size = 2;
  (void)CheckAndConvertUtils::CheckInteger("x rank", SizeToLong(input_shape.size()), kGreaterEqual, x_size, prim_name);
  auto data_format_ptr = primitive->GetAttr("format");
  int64_t data_format = static_cast<int64_t>(Format::NCHW);
  if (data_format_ptr != nullptr) {
    data_format = CheckAndConvertUtils::GetAndCheckFormat(data_format_ptr);
  }
  auto attr_value_str = FormatEnumToString(Format(data_format));
  (void)primitive->AddAttr("data_format", data_format_ptr);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  auto is_cpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
  if ((data_format == static_cast<int64_t>(Format::NCDHW)) && input_shape.size() != x_max_rank &&
      (is_ascend || is_cpu)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', NCDHW format only supports 5-D input on Ascend or CPU, but got a "
                             << input_shape.size() << "-D input.";
  }
  if (IsNHWCorNCHW(data_format) && IsShapeSizeOutOfRange(input_shape.size())) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the dimension of 'input_x' tensor must be 2D-5D when data_format is "
                             << attr_value_str << ".";
  }

  if ((data_format == static_cast<int64_t>(Format::NHWC)) && bias_shape[0] != input_shape[input_shape.size() - 1]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', bias[0] shape should be equal to input_x["
                             << (input_shape.size() - 1) << "] shape when data_format is " << attr_value_str << ".";
  }
  if (IsNCHWorNCDHW(data_format) && input_shape[1] != abstract::Shape::kShapeDimAny &&
      bias_shape[0] != input_shape[1]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', bias[0] shape should be equal to input_x[1] "
                                "shape when data_format is "
                             << attr_value_str << ".";
  }
  if ((data_format == static_cast<int64_t>(Format::NHWC)) && (input_shape.size() == x_max_rank) && is_ascend) {
    MS_EXCEPTION(ValueError) << "For 5-D input, '" << prim_name << "', only supports NCHW and NCDHW on Ascend, "
                             << "but got an invalidC.";
  }
  return std::make_shared<abstract::Shape>(input_shape);
}
TypePtr BiasAddInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  std::set<TypePtr> valid_x_type = {common_valid_types_with_complex};

  auto x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_x_type, prim_name);
  auto bias_type = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(bias_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("bias", bias_type, valid_x_type, prim_name);

  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_x", input_args[0]->BuildType());
  (void)types.emplace("bias", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_x_type, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(BiasAdd, BaseOperator);
void BiasAdd::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}
Format BiasAdd::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}
std::string BiasAdd::get_str_format() const {
  auto value_ptr = GetAttr("format");
  return GetValue<std::string>(value_ptr);
}
void BiasAdd::Init(const Format &format) { this->set_format(format); }
AbstractBasePtr BiasAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  auto infertype = BiasAddInferType(primitive, input_args);
  auto infershape = BiasAddInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

// AG means auto generated
class MIND_API AGBiasAddInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BiasAddInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BiasAddInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BiasAddInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BiasAdd, prim::kPrimBiasAdd, AGBiasAddInfer, false);
}  // namespace ops
}  // namespace mindspore
