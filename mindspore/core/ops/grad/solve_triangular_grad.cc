/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/grad/solve_triangular_grad.h"

#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kSolveTriangularInputsNum = 6;
constexpr int64_t kIndexA = 0;
constexpr int64_t kIndexDX = 2;

abstract::TupleShapePtr SolveTriangularGradInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kIndexA]->GetShape());
  auto a_shape = input_args[kIndexA]->GetShape();
  MS_EXCEPTION_IF_NULL(input_args[kIndexDX]->GetShape());
  auto dx_shape = input_args[kIndexDX]->GetShape();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{a_shape, dx_shape});
}

TypePtr SolveTriangularGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto a_type = input_args[kIndexA]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(a_type);
  auto input_a_type_id = a_type->element()->type_id();
  static const std::vector<TypeId> type_to_float32 = {
    kNumberTypeFloat16, kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64,
  };
  bool is_type_to_float32 =
    std::any_of(type_to_float32.begin(), type_to_float32.end(),
                [&input_a_type_id](const TypeId &type_id) { return input_a_type_id == type_id; });
  TypePtr grad_type;
  if (is_type_to_float32) {
    grad_type = kFloat32;
  } else {
    grad_type = input_args[kIndexDX]->GetType()->Clone();
  }
  return std::make_shared<Tuple>(std::vector<TypePtr>{grad_type, grad_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SolveTriangularGrad, BaseOperator);
AbstractBasePtr SolveTriangularGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SolveTriangularGradInferType(primitive, input_args);
  auto infer_shape = SolveTriangularGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSolveTriangularGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SolveTriangularGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SolveTriangularGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SolveTriangularGradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SolveTriangularGrad, prim::kPrimSolveTriangularGrad, AGSolveTriangularGradInfer,
                                 false);

}  // namespace ops
}  // namespace mindspore
