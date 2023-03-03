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

#include "ops/lu_scipy.h"

#include <algorithm>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kLUInputsNum = 1;
constexpr size_t kXDim = 2;
constexpr size_t kLastDim = 1;
constexpr size_t kPenultimateDim = 2;
abstract::TupleShapePtr LUInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];

  auto x_output = std::make_shared<abstract::Shape>(x_shape);
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_output, x_output, x_output});
  }

  size_t x_shape_size = x_shape.size();
  if (x_shape_size < kXDim) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "',"
                             << " the dimension of hashmap must be greater than or equal to 2, but got: "
                             << x_shape_size << ".";
  }

  auto k_shape = std::min(x_shape[x_shape_size - kLastDim], x_shape[x_shape_size - kPenultimateDim]);
  ShapeVector top_k_shape(x_shape.begin(), x_shape.end() - SizeToLong(kPenultimateDim));
  ShapeVector pivots_shape = top_k_shape;
  pivots_shape.push_back(k_shape);
  ShapeVector permutation_shape = pivots_shape;
  permutation_shape.push_back(k_shape);

  auto pivots_output = std::make_shared<abstract::Shape>(pivots_shape);
  auto permutation_output = std::make_shared<abstract::Shape>(permutation_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{x_output, pivots_output, permutation_output});
}

TuplePtr LUInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_type = input_args[0]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, kInt32, kInt32});
}
}  // namespace

AbstractBasePtr LUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, SizeToLong(kLUInputsNum), primitive->name());
  auto infer_type = LUInferType(primitive, input_args);
  auto infer_shape = LUInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(LU, BaseOperator);

// AG means auto generated
class MIND_API AGLUInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LUInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LUInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LUInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LU, prim::kPrimLU, AGLUInfer, false);
}  // namespace ops
}  // namespace mindspore
