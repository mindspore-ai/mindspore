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

#include "ops/grad/bias_add_grad.h"
#include <string>
#include <algorithm>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/ms_context.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> GetFormatShape(const int64_t &format, const std::vector<int64_t> &input_shape) {
  std::vector<int64_t> output_shape;
  if (format == NHWC) {
    output_shape.push_back(input_shape.back());
  } else {
    output_shape.push_back(input_shape[1]);
  }
  return output_shape;
}
abstract::ShapePtr BiasAddGradInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamic(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  const int64_t x_min_rank = 2;
  const int64_t x_max_rank = 5;
  const int64_t last_dims = 2;
  const int64_t three_dims = 3;
  const int64_t error_size = 1;
  CheckAndConvertUtils::CheckInRange("dims of input_x", input_shape.size(), kIncludeBoth, {x_min_rank, x_max_rank},
                                     prim_name);
  auto data_format_ptr = primitive->GetAttr("format");
  (void)primitive->AddAttr("data_format", data_format_ptr);

  int64_t data_format = static_cast<int64_t>(Format::NCHW);
  if (data_format_ptr == nullptr) {
    data_format = static_cast<int64_t>(Format::NCHW);
  }
  auto attr_value_str = GetValue<std::string>(data_format_ptr);

  if (attr_value_str == "NCHW") {
    data_format = static_cast<int64_t>(Format::NCHW);
  } else if (attr_value_str == "NHWC") {
    data_format = static_cast<int64_t>(Format::NHWC);
  } else if (attr_value_str == "NCDHW") {
    data_format = static_cast<int64_t>(Format::NCDHW);
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the data_format must be NCHW, NHWC, or NCDHW, but got: " << attr_value_str << ".";
  }
  auto input_shape_ = GetFormatShape(data_format, input_shape);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  auto is_cpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
  if ((data_format == static_cast<int64_t>(Format::NCDHW)) && input_shape.size() != x_max_rank &&
      (is_ascend || is_cpu)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', NCDHW format only support 5-dims input in Ascend or CPU target, but got "
                             << attr_value_str << ".";
  }
  if (data_format == static_cast<int64_t>(Format::NCHW) && input_shape.size() == three_dims &&
      input_shape[last_dims] == error_size && (is_ascend || is_cpu)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', input tensor's dimension is 3, when data_format is NCHW "
                                "the last dimension size should greater than 1, but got "
                             << error_size << ".";
  }

  return std::make_shared<abstract::Shape>(input_shape_);
}
TypePtr BiasAddGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("BiasAddGrad infer", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type_map = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type_map);
  auto x_type = x_type_map->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_type);
  std::set<TypePtr> valid_x_type = {kTensorType};
  return CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_x_type, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(BiasAddGrad, BaseOperator);
std::string BiasAddGrad::get_str_format() const {
  auto value_ptr = GetAttr("format");
  return GetValue<std::string>(value_ptr);
}
AbstractBasePtr BiasAddGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(BiasAddGradInferShape(primitive, input_args),
                                BiasAddGradInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(BiasAddGrad, prim::kPrimBiasAddGrad, BiasAddGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
