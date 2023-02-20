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

#include "ops/padding.h"

#include <vector>
#include <set>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/ms_context.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "ir/primitive.h"
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
TypePtr PaddingInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_gpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  bool is_cpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
  std::set<TypePtr> valid_types{};
  if (is_gpu || is_cpu) {
    valid_types = common_valid_types_with_complex_and_bool;
  } else {
    valid_types = common_valid_types_with_bool;
  }
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, name);
}

abstract::ShapePtr PaddingInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t kInputSize = 1;
  constexpr int64_t kNumber1 = 1;
  constexpr int64_t kNumber2 = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_x_shape_ptr = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(input_x_shape_ptr);
  if (input_x_shape_ptr->IsDynamic()) {
    return input_args[0]->BuildShape()->cast<abstract::ShapePtr>();
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("x rank", x_rank, kGreaterEqual, kNumber2, prim_name);
  int64_t x_last_dim = x_shape[x_shape.size() - 1];
  if (x_last_dim != kNumber1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the last dimension of 'x' must be 1, but got: " << x_last_dim << ".";
  }
  auto value_ptr = primitive->GetAttr(kPadDimSize);
  auto pad_dim_size = GetValue<int64_t>(value_ptr);
  (void)CheckAndConvertUtils::CheckInteger("pad_dim_size", pad_dim_size, kGreaterEqual, kNumber1, prim_name);
  // Extends the last dimension of the input tensor from 1 to pad_dim_size.
  x_shape[x_shape.size() - 1] += pad_dim_size - 1;
  return std::make_shared<abstract::Shape>(x_shape);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Padding, BaseOperator);

void Padding::Init(int64_t pad_dim_size) { set_pad_dim_size(pad_dim_size); }

void Padding::set_pad_dim_size(int64_t pad_dim_size) { (void)AddAttr(kPadDimSize, api::MakeValue(pad_dim_size)); }

int64_t Padding::get_pad_dim_size() const {
  auto value_ptr = GetAttr(kPadDimSize);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr PaddingInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  abstract::ShapePtr output_shape = PaddingInferShape(primitive, input_args);
  TypePtr output_type = PaddingInferType(primitive, input_args);
  return abstract::MakeAbstract(output_shape, output_type);
}

// AG means auto generated
class MIND_API AGPaddingInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PaddingInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PaddingInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PaddingInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Padding, prim::kPrimPadding, AGPaddingInfer, false);
}  // namespace ops
}  // namespace mindspore
