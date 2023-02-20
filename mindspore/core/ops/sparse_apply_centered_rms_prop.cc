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

#include "ops/sparse_apply_centered_rms_prop.h"

#include <algorithm>
#include <set>
#include <map>
#include <memory>
#include <utility>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseApplyCenteredRMSPropInferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto mg_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto ms_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto mom_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->BuildShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShapeTrack())[kShape];
  auto rho_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShapeTrack())[kShape];
  auto momentum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->GetShapeTrack())[kShape];
  auto epsilon_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[7]->GetShapeTrack())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[8]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[9]->BuildShape())[kShape];

  const int64_t scalar_shape = 0;
  if (IsDynamicRank(var_shape) || IsDynamicRank(mg_shape) || IsDynamicRank(ms_shape) || IsDynamicRank(mom_shape) ||
      IsDynamicRank(grad_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kEqual, scalar_shape, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("rho_shape size", rho_shape.size(), kEqual, scalar_shape, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("momentum_shape size", momentum_shape.size(), kEqual, scalar_shape,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("epsilon_shape size", epsilon_shape.size(), kEqual, scalar_shape, prim_name);

  std::vector<ShapeVector> tensor_shapes = {var_shape, mg_shape, ms_shape, mom_shape};
  auto is_dynamic_tensor = std::any_of(tensor_shapes.begin(), tensor_shapes.end(), IsDynamic);
  if (!is_dynamic_tensor) {
    std::map<std::string, ShapeVector> same_shape_args_map;
    (void)same_shape_args_map.emplace("shape of mg", mg_shape);
    (void)same_shape_args_map.emplace("ms", ms_shape);
    (void)same_shape_args_map.emplace("mom", mom_shape);
    for (auto &elem : same_shape_args_map) {
      CheckAndConvertUtils::Check(elem.first, elem.second, kEqual, var_shape, prim_name);
    }
  }
  // Var dimension must be equal or greater than 1.
  (void)CheckAndConvertUtils::CheckInteger("var dimension", SizeToLong(var_shape.size()), kGreaterEqual, 1, prim_name);
  // Indices must be rank 1.
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", SizeToLong(indices_shape.size()), kEqual, 1, prim_name);
  auto is_dynamic = IsDynamic(var_shape) || IsDynamic(grad_shape) || IsDynamic(indices_shape);
  if (!is_dynamic) {
    if (var_shape.size() != grad_shape.size()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', rank(grad) should be same as rank(var), but got rank(grad): " << grad_shape.size()
                               << ", rank(var): " << var_shape.size() << ".";
    }
    for (size_t i = 1; i < var_shape.size(); ++i) {
      if (var_shape[i] != grad_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the shape of var and grad must equal in dimension " << i
                                 << ".";
      }
    }
    if (indices_shape[0] != grad_shape[0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', grad.shape[0] must be equal to indices.shape[0], but got grad_shape[0]: "
                               << grad_shape[0] << " indices_shape[0]: " << indices_shape[0] << ".";
    }
  }

  return std::make_shared<abstract::Shape>(var_shape);
}

TypePtr SparseApplyCenteredRMSPropInferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var = input_args[0]->BuildType();
  auto mg = input_args[1]->BuildType();
  auto ms = input_args[2]->BuildType();
  auto mom = input_args[3]->BuildType();
  auto lr = input_args[4]->BuildType();
  auto rho = input_args[5]->BuildType();
  auto momentum = input_args[6]->BuildType();
  auto epsilon = input_args[7]->BuildType();
  auto grad = input_args[8]->BuildType();
  auto indices = input_args[9]->BuildType();
  std::map<std::string, TypePtr> args;
  (void)args.emplace("var", var);
  (void)args.emplace("ms", mg);
  (void)args.emplace("mg", ms);
  (void)args.emplace("mom", mom);
  (void)args.emplace("lr", lr);
  (void)args.emplace("rho", rho);
  (void)args.emplace("momentum", momentum);
  (void)args.emplace("epsilon", epsilon);
  (void)args.emplace("grad", grad);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args, common_valid_types, prim_name);
  const std::set<TypePtr> valids1 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices, valids1, prim_name);

  return var;
}
}  // namespace

void SparseApplyCenteredRMSProp::Init(bool use_locking) { set_use_locking(use_locking); }

void SparseApplyCenteredRMSProp::set_use_locking(bool use_locking) {
  (void)AddAttr(kUseLocking, api::MakeValue(use_locking));
}
bool SparseApplyCenteredRMSProp::get_use_locking() { return GetValue<bool>(GetAttr(kUseLocking)); }

AbstractBasePtr SparseApplyCenteredRMSPropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int INPUTS_NUM = 10;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, INPUTS_NUM, primitive->name());
  auto infer_type = SparseApplyCenteredRMSPropInferType(primitive, input_args);
  auto infer_shape = SparseApplyCenteredRMSPropInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseApplyCenteredRMSProp, BaseOperator);

// AG means auto generated
class MIND_API AGSparseApplyCenteredRMSPropInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyCenteredRMSPropInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyCenteredRMSPropInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyCenteredRMSPropInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseApplyCenteredRMSProp, prim::kPrimSparseApplyCenteredRMSProp,
                                 AGSparseApplyCenteredRMSPropInfer, false);
}  // namespace ops
}  // namespace mindspore
