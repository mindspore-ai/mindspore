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

#include "ops/sparse_apply_adadelta.h"

#include <set>
#include <map>
#include <type_traits>
#include <utility>

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
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseApplyAdadeltaInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Indices and grad must be tensor
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex5);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex6);
  // Get input shape
  auto var_shape_ptr = input_args[0]->BuildShape();
  auto accum_shape_ptr = input_args[1]->BuildShape();
  auto accum_updata_shape_ptr = input_args[2]->BuildShape();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape_ptr)[kShape];
  auto accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(accum_shape_ptr)[kShape];
  auto accum_updata_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(accum_updata_shape_ptr)[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  auto rho_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->BuildShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->BuildShape())[kShape];
  auto grad_shape_rank = SizeToLong(grad_shape.size());
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->BuildShape())[kShape];
  // Args lr rho must be scalar
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape.size()), kEqual, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rho_shape size", SizeToLong(rho_shape.size()), kEqual, 0, prim_name);
  // Args var,accum,accum_update and grad shape must be same
  std::map<std::string, ShapeVector> same_shape_args_map;
  (void)same_shape_args_map.insert(std::make_pair("accum shape", accum_shape));
  (void)same_shape_args_map.insert(std::make_pair("accum_updata shape", accum_updata_shape));
  (void)same_shape_args_map.insert(std::make_pair("grad shape", grad_shape));
  for (auto &elem : same_shape_args_map) {
    CheckAndConvertUtils::Check(elem.first, elem.second, kEqual, var_shape, prim_name);
  }
  // Indices must be rank 1
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", SizeToLong(indices_shape.size()), kEqual, 1, prim_name);
  // Grad dimension must be equal or greater than 1
  (void)CheckAndConvertUtils::CheckInteger("grad dimension", grad_shape_rank, kGreaterEqual, 1, prim_name);
  // Indices size must equal with grad first dimension size
  if (indices_shape[0] != grad_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the indices size must be equal to grad first dimension size. But got indices size: "
                             << indices_shape[0] << ", grad first dimension size: " << grad_shape[0] << ".";
  }
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr, accum_updata_shape_ptr});
}

TuplePtr SparseApplyAdadeltaInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Get all inputs's type
  auto var_type = input_args[0]->BuildType();
  auto accum_type = input_args[1]->BuildType();
  auto accum_updata_type = input_args[2]->BuildType();
  auto lr_type = input_args[3]->BuildType();
  auto rho_type = input_args[4]->BuildType();
  auto grad_type = input_args[5]->BuildType();
  auto indices_type = input_args[6]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // Args accum accum_updata and grad must have the same type as var
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var", var_type));
  (void)args.insert(std::make_pair("accum", accum_type));
  (void)args.insert(std::make_pair("accum_updata", accum_updata_type));
  (void)args.insert(std::make_pair("grad", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // Args lr rho must be a scalar type
  std::map<std::string, TypePtr> args2;
  (void)args2.insert(std::make_pair("lr", lr_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args2, valid_types, prim_name);
  std::map<std::string, TypePtr> args3;
  (void)args3.insert(std::make_pair("rho", rho_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args3, valid_types, prim_name);
  // Check indices_type
  std::map<std::string, TypePtr> args4;
  (void)args4.insert(std::make_pair("indices", indices_type));
  const std::set<TypePtr> valid_types2 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args4, valid_types2, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type, accum_updata_type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseApplyAdadelta, BaseOperator);
AbstractBasePtr SparseApplyAdadeltaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 7;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SparseApplyAdadeltaInferType(primitive, input_args);
  auto infer_shape = SparseApplyAdadeltaInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSparseApplyAdadeltaInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyAdadeltaInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyAdadeltaInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyAdadeltaInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseApplyAdadelta, prim::kPrimSparseApplyAdadelta, AGSparseApplyAdadeltaInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
