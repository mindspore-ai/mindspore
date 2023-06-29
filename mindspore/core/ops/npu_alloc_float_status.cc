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
#include "ops/npu_alloc_float_status.h"

#include <memory>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/other_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr NPUAllocFloatStatusInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &) {
  ShapeVector output_shape;
  constexpr int64_t kFloatStatusNum = 8;
  output_shape.push_back(kFloatStatusNum);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr NPUAllocFloatStatusInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.size() != 0) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "' op, input num should be 0, bug gets "
                            << input_args.size();
  }
  return kTensorTypeFP32;
}

AbstractBasePtr NPUAllocFloatStatusInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(NPUAllocFloatStatusInferShape(primitive, input_args),
                                NPUAllocFloatStatusInferType(primitive, input_args));
}
}  // namespace

MIND_API_OPERATOR_IMPL(NPUAllocFloatStatus, BaseOperator);
class MIND_API AGNPUAllocFloatStatusInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUAllocFloatStatusInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUAllocFloatStatusInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NPUAllocFloatStatusInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NPUAllocFloatStatus, prim::kPrimNPUAllocFloatStatus, AGNPUAllocFloatStatusInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
