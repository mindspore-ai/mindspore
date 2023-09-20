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

#include "ops/add_v2.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AddV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto is_gpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  if (!is_gpu) {
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex0]->BuildShape())[kShape];
    auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex1]->BuildShape())[kShape];
    CheckAndConvertUtils::Check("input_shape", x_shape, kEqual, y_shape, primitive->name(), ValueError);
  }
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr AddV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  MS_EXCEPTION_IF_NULL(prim);
  constexpr int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim->name());
  const std::set<TypePtr> valid_types = {kInt8,   kInt16, kInt32,   kInt64,   kUInt8,   kUInt16,    kUInt32,
                                         kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)types.emplace("x", input_args[kIndex0]->BuildType());
  (void)types.emplace("y", input_args[kIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[kIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(AddV2, BaseOperator);
AbstractBasePtr AddV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  auto output_type = AddV2InferType(primitive, input_args);
  auto output_shape = AddV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(output_shape, output_type);
}

// AG means auto generated
class MIND_API AGAddV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AddV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AddV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AddV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AddV2, prim::kPrimAddV2, AGAddV2Infer, false);
}  // namespace ops
}  // namespace mindspore
