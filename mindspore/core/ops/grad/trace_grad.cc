/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/trace_grad.h"
#include <string>
#include <set>
#include <vector>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr TraceGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args[1]->isa<abstract::AbstractTensor>() && !input_args[1]->BuildValue()->isa<AnyValue>() &&
      !input_args[1]->BuildValue()->isa<None>()) {
    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
    auto x_shape = shape_map[kShape];
    // TraceGrad x_shape must be 2
    (void)CheckAndConvertUtils::CheckInteger("x shape size", x_shape[0], kEqual, 2, primitive->name());
    // build Trace output shape
    auto x = input_args[1]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(x);
    auto x_ptr = x->BuildValue();
    MS_EXCEPTION_IF_NULL(x_ptr);
    auto x_tensor = x_ptr->cast<tensor::TensorPtr>();

    MS_EXCEPTION_IF_NULL(x_tensor);
    auto data_size = x_tensor->DataSize();
    auto type_id = x_tensor->data_type();
    ShapeVector out_shape = {};
    switch (type_id) {
      case kNumberTypeInt32: {
        int32_t *x_data = reinterpret_cast<int32_t *>(x_tensor->data_c());
        for (size_t i = 0; i < data_size; ++i) {
          out_shape.push_back(static_cast<int64_t>(x_data[i]));
        }
        break;
      }
      case kNumberTypeInt64: {
        int64_t *x_data = reinterpret_cast<int64_t *>(x_tensor->data_c());
        for (size_t i = 0; i < data_size; ++i) {
          out_shape.push_back(static_cast<int64_t>(x_data[i]));
        }
        break;
      }
      default: {
        MS_EXCEPTION(TypeError) << "For TraceGrad, the type of shape must be int32 or int64";
      }
    }
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
    auto x_shape = shape_map[kShape];
    if (!IsDynamic(x_shape)) {
      // TraceGrad x_shape must be 2
      (void)CheckAndConvertUtils::CheckInteger("x shape size", x_shape[0], kEqual, 2, primitive->name());
    }
    auto infer_shape_max = shape_map[kMaxShape];
    std::vector<int64_t> out_shape = {abstract::Shape::SHP_ANY};
    std::vector<int64_t> infer_shape_min = {0};
    return std::make_shared<abstract::Shape>(out_shape, infer_shape_min, infer_shape_max);
  }
}

TypePtr TraceGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_grad_types = {kInt8,  kInt16,  kInt32,  kInt64,  kFloat16,   kFloat32,   kFloat64,
                                              kUInt8, kUInt16, kUInt32, kUInt64, kComplex64, kComplex128};
  const std::set<TypePtr> valid_shape_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("x_shape", input_args[1]->BuildType(), valid_shape_types, prim->name());
  return CheckAndConvertUtils::CheckTypeValid("y_grad", input_args[0]->BuildType(), valid_grad_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(TraceGrad, BaseOperator);
AbstractBasePtr TraceGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = TraceGradInferType(primitive, input_args);
  auto infer_shape = TraceGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(TraceGrad, prim::kPrimTraceGrad, TraceGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
