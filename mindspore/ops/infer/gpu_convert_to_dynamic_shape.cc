/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/helper.h"
#include "primitive/structure_ops.h"
#include "infer/gpu_convert_to_dynamic_shape.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GpuConvertToDynamicShapeInferShape(const PrimitivePtr &,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape())[kShape];
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  ShapeVector output_shape_dyn;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    output_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
  }
  return std::make_shared<abstract::Shape>(output_shape_dyn);
}

TypePtr GpuConvertToDynamicShapeInferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[0]->GetType();
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(GpuConvertToDynamicShape, BaseOperator);
AbstractBasePtr GpuConvertToDynamicShapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = GpuConvertToDynamicShapeInferType(primitive, input_args);
  auto infer_shape = GpuConvertToDynamicShapeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class OPS_API AGGpuConvertToDynamicShapeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(input_args[0]);
    const auto &input_shape = input_args[0]->GetShape()->GetShapeVector();
    if (IsDynamic(input_shape)) {
      MS_LOG(EXCEPTION) << "Got dynamic input shape: " << input_shape;
    }
    return std::make_shared<abstract::TensorShape>(input_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GpuConvertToDynamicShapeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GpuConvertToDynamicShapeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GpuConvertToDynamicShape, prim::kPrimGpuConvertToDynamicShape,
                                 AGGpuConvertToDynamicShapeInfer, false);
}  // namespace ops
}  // namespace mindspore
