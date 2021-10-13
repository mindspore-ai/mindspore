/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "abstract/primitive_infer_map.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto bias = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(bias);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("arg size", SizeToLong(input_args.size()), kEqual, input_num, prim_name);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  const int64_t x_min_rank = 2;
  const int64_t x_max_rank = 5;
  CheckAndConvertUtils::CheckInRange("dims of input_x", input_shape.size(), kIncludeBoth, {x_min_rank, x_max_rank},
                                     prim_name);
  auto bias_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("bias rank", SizeToLong(bias_shape.size()), kEqual, 1, prim_name);
  const int64_t x_size = 2;
  (void)CheckAndConvertUtils::CheckInteger("x rank", SizeToLong(input_shape.size()), kGreaterEqual, x_size, prim_name);
  auto data_format_ptr = primitive->GetAttr("format");
  int64_t data_format = Format::NCHW;
  if (data_format_ptr != nullptr) {
    data_format = CheckAndConvertUtils::GetAndCheckFormat(data_format_ptr);
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (data_format == Format::NCDHW && input_shape.size() != 5 && is_ascend) {
    MS_EXCEPTION(ValueError) << "NCDHW format only support 5-dims input in Ascend target.";
  }
  auto x_channel = data_format == Format::NHWC ? input_shape[input_shape.size() - 1] : input_shape[1];
  bool x_not_dyn = std::all_of(input_shape.begin(), input_shape.end(),
                               [](int64_t value) { return value != abstract::Shape::SHP_ANY; });
  if (x_not_dyn && bias_shape[0] != x_channel) {
    MS_EXCEPTION(ValueError) << "BiasAdd shape error, data format is " << data_format
                             << ", got bias_shape[0]: " << bias_shape[0] << ", x_channel: " << x_channel << ".";
  }
  CheckAndConvertUtils::CheckMinMaxShape(input_shape, &min_shape, &max_shape);
  if (min_shape.size() != 0 && max_shape.size() != 0) {
    return std::make_shared<abstract::Shape>(input_shape, min_shape, max_shape);
  }
  return std::make_shared<abstract::Shape>(input_shape);
}
TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("biasadd_infer", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  for (size_t i = 0; i < input_args.size(); i++) {
    auto x_type = input_args[i]->BuildType();
    MS_EXCEPTION_IF_NULL(x_type);
    std::set<TypePtr> valid_x_type = {kTensorType};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_type, valid_x_type, prim_name);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_x", input_args[0]->BuildType());
  (void)types.emplace("bias", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim_name);
}
}  // namespace
void BiasAdd::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, MakeValue(f));
}
Format BiasAdd::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}
void BiasAdd::Init(const Format &format) { this->set_format(format); }
AbstractBasePtr BiasAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  auto infertype = InferType(primitive, input_args);
  auto infershape = InferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(BiasAdd, prim::kPrimBiasAdd, BiasAddInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
