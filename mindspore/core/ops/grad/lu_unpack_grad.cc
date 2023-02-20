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

#include "ops/grad/lu_unpack_grad.h"

#include <algorithm>
#include <vector>
#include <set>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr LuUnpackGradInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto L_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto U_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto LU_data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];

  auto LU_data_ouput = std::make_shared<abstract::Shape>(LU_data_shape);
  if (IsDynamic(LU_data_shape) || IsDynamic(L_shape) || IsDynamic(U_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{LU_data_ouput, LU_data_ouput});
  }

  auto L_data_dim1 = L_shape[L_shape.size() - 2];
  auto L_data_dim2 = L_shape[L_shape.size() - 1];
  auto U_data_dim1 = U_shape[U_shape.size() - 2];
  auto U_data_dim2 = U_shape[U_shape.size() - 1];
  auto LU_data_dim1 = LU_data_shape[LU_data_shape.size() - 2];
  auto LU_data_dim2 = LU_data_shape[LU_data_shape.size() - 1];
  int64_t LU_data_min = std::min(LU_data_dim1, LU_data_dim2);

  if (LU_data_dim1 != L_data_dim1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', L_grad's data dim[-2] and LU_data's dim[-2] should be same.";
  }
  if (LU_data_min != L_data_dim2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', L_grad's data dim[-1] and LU_data's minimum dim should be same.";
  }
  if (U_data_dim2 != LU_data_dim2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', U_grad's data dim[-1] and LU_data's dim[-1] should be same.";
  }
  if (LU_data_min != U_data_dim1) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', U_grad's data dim[-2] and LU_data's minimum dim should be same.";
  }

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{LU_data_ouput, LU_data_ouput});
}

TuplePtr LuUnpackGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::set<TypePtr> valid_types = {kFloat64, kFloat32, kFloat16, kInt64, kInt32, kInt16, kInt8, kUInt8};
  auto LU_type = input_args[2]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("LU_data", LU_type, valid_types, primitive->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{LU_type, LU_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(LuUnpackGrad, BaseOperator);
AbstractBasePtr LuUnpackGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = LuUnpackGradInferType(primitive, input_args);
  auto infer_shape = LuUnpackGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLuUnpackGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LuUnpackGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LuUnpackGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LuUnpackGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LuUnpackGrad, prim::kPrimLuUnpackGrad, AGLuUnpackGradInfer, false);
}  // namespace ops
}  // namespace mindspore
