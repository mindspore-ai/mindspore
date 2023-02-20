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
#include "ops/lu_solve_.h"

#include <map>
#include <ostream>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

#define LuSolve_for(shape)    \
  do {                        \
    for (auto item : shape) { \
      buffer << item << " ";  \
    }                         \
  } while (0)

#define LuSolve_pop()                \
  do {                               \
    for (size_t j = 0; j < 2; j++) { \
      x_shape.pop_back();            \
      lu_data_shape.pop_back();      \
    }                                \
  } while (0)

#define LuSolve_buffer(x_shape, lu_data_shape)                                                                      \
  do {                                                                                                              \
    LuSolve_pop();                                                                                                  \
    buffer << "For LuSolve x's batch dimension does not match lu_data's batch dimension, x's batch dimension is ["; \
    LuSolve_for(x_shape);                                                                                           \
    buffer << "], lu_data's batch dimension is [";                                                                  \
    LuSolve_for(lu_data_shape);                                                                                     \
    buffer << "], the batch dimensions may have different sizes, ";                                                 \
    buffer << "from right to left, the corresponding dimensions must be equal.";                                    \
  } while (0)

namespace mindspore {
namespace ops {
namespace {
void CheckInputsShape(const ShapeVector &x_shape, const ShapeVector &lu_data_shape, const ShapeVector &lu_pivots_shape,
                      const std::string &op_name) {
  const int64_t kDimNum = 2;
  if (lu_data_shape.size() < kDimNum) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', lu_data's dimension must be greater than or equal to 2, but got: "
                             << lu_data_shape.size() << ".";
  }
  if (x_shape.size() < kDimNum) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', x's dimension must be greater than or equal to 2, but got: " << x_shape.size()
                             << ".";
  }
  if (lu_pivots_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', lu_pivots's dimension must be greater than or equal to 1, but got: "
                             << lu_pivots_shape.size() << ".";
  }
  if (lu_data_shape[lu_data_shape.size() - 1] != lu_data_shape[lu_data_shape.size() - kDimNum]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', input lu_data must be a square matrix, "
                             << "but got row: " << lu_data_shape[lu_data_shape.size() - kDimNum]
                             << ", col: " << lu_data_shape[lu_data_shape.size() - 1] << ".";
  }
  if (x_shape[x_shape.size() - kDimNum] != lu_data_shape[lu_data_shape.size() - kDimNum]) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', x's col rank must be the same as lu_data's col rank, "
                             << "but got x's: " << x_shape[x_shape.size() - kDimNum]
                             << ", lu_data's: " << lu_data_shape[lu_data_shape.size() - kDimNum] << ".";
  }
}

abstract::ShapePtr LuSolveInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t kDimNum = 2;
  std::ostringstream buffer;
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto lu_data_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto lu_data_shape = lu_data_shape_map[kShape];
  auto lu_pivots_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape());
  auto lu_pivots_shape = lu_pivots_shape_map[kShape];

  if (IsDynamicRank(x_shape) || IsDynamicRank(lu_data_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  if (!IsDynamicShape(x_shape) && !IsDynamicShape(lu_data_shape) && !IsDynamic(lu_pivots_shape)) {
    CheckInputsShape(x_shape, lu_data_shape, lu_pivots_shape, op_name);
    if (x_shape.size() == lu_data_shape.size()) {
      for (size_t i = 0; i <= x_shape.size() - kDimNum; i++) {
        if (x_shape[i] != lu_data_shape[i]) {
          LuSolve_buffer(x_shape, lu_data_shape);
          MS_EXCEPTION(ValueError) << buffer.str();
        }
      }
    } else if (lu_data_shape.size() > x_shape.size()) {
      for (size_t i = 0; i < x_shape.size() - kDimNum; i++) {
        if (x_shape[i] != lu_data_shape[lu_data_shape.size() - x_shape.size() + i]) {
          LuSolve_buffer(x_shape, lu_data_shape);
          MS_EXCEPTION(ValueError) << buffer.str();
        }
      }
    } else {
      for (size_t i = 0; i < lu_data_shape.size() - kDimNum; i++) {
        if (lu_data_shape[i] != x_shape[x_shape.size() - lu_data_shape.size() + i]) {
          LuSolve_buffer(x_shape, lu_data_shape);
          MS_EXCEPTION(ValueError) << buffer.str();
        }
      }
    }
    if (lu_pivots_shape[lu_pivots_shape.size() - 1] != lu_data_shape[lu_data_shape.size() - 1]) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', the last dim of lu_pivots must be the same as lu_data's last dim, "
                               << "but got lu_pivots' last dim: " << lu_pivots_shape[lu_pivots_shape.size() - 1]
                               << ", lu_data's last dim: " << lu_data_shape[lu_data_shape.size() - 1] << ".";
    }
    for (size_t i = 0; i < lu_pivots_shape.size(); i++) {
      if (lu_data_shape[i] != lu_pivots_shape[i]) {
        x_shape.pop_back();
        x_shape.pop_back();
        lu_pivots_shape.pop_back();
        buffer
          << "For " << op_name
          << " lu_data's batch dimension does not match lu_pivots's batch dimension, lu_data's batch dimension is [";
        LuSolve_for(x_shape);
        buffer << "], lu_pivots's batch dimension is [";
        LuSolve_for(lu_pivots_shape);
        buffer << "], the size of the dimension and the number of each dimension must be the same.";
        MS_EXCEPTION(ValueError) << buffer.str();
      }
    }
  }

  auto dim_vector = lu_data_shape;
  if (x_shape.size() >= lu_data_shape.size()) {
    return std::make_shared<abstract::Shape>(x_shape);
  } else {
    dim_vector[lu_data_shape.size() - 1] = x_shape[x_shape.size() - 1];
    return std::make_shared<abstract::Shape>(dim_vector);
  }
}

TypePtr LuSolveInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kDimNum = 2;
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> type;
  (void)type.emplace("x", input_args[0]->BuildType());
  (void)type.emplace("lu_data", input_args[1]->BuildType());
  const std::set<TypePtr> valid_types = {kFloat64, kFloat32, kFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lu_data", input_args[1]->BuildType(), valid_types, prim->name());
  auto out_type = CheckAndConvertUtils::CheckTensorTypeSame(type, valid_types, prim->name());
  const std::set<TypePtr> valid_lu_pivots_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lu_pivots", input_args[kDimNum]->BuildType(), valid_lu_pivots_types,
                                                   prim->name());
  return out_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(LuSolve, BaseOperator);
AbstractBasePtr LuSolveInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = LuSolveInferType(primitive, input_args);
  auto infer_shape = LuSolveInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLuSolveInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LuSolveInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LuSolveInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LuSolveInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LuSolve, prim::kPrimLuSolve, AGLuSolveInfer, false);
}  // namespace ops
}  // namespace mindspore
