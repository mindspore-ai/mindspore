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
#include "ops/sparse_apply_adagrad.h"

#include <set>
#include <map>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "include/common/utils/utils.h"
#include "ir/dtype/container.h"
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
#include "utils/overload.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SparseApplyAdagradInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Indices and grad must be tensor
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex2);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex3);
  // Get input shape
  auto var_shape_ptr = input_args[0]->BuildShape();
  auto accum_shape_ptr = input_args[1]->BuildShape();
  auto grad_shape_ptr = input_args[2]->BuildShape();
  auto indices_shape_ptr = input_args[3]->BuildShape();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape_ptr)[kShape];
  auto accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(accum_shape_ptr)[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(grad_shape_ptr)[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];

  if (var_shape_ptr->IsDynamic() || accum_shape_ptr->IsDynamic() || grad_shape_ptr->IsDynamic() ||
      indices_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
  }
  (void)CheckAndConvertUtils::CheckValue("var_shape", var_shape, kEqual, "accum_shape", accum_shape, prim_name);

  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    batch_rank = GetValue<int64_t>(primitive->GetAttr(kBatchRank));
  }
  // Indices must be rank 1
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", indices_shape.size(), kEqual, batch_rank + 1,
                                           prim_name);
  // Grad dimension must be equal or greater than 1
  (void)CheckAndConvertUtils::CheckInteger("grad dimension", grad_shape.size(), kGreaterEqual, batch_rank + 1,
                                           prim_name);

  // Indices size must equal with grad first dimension size
  if (indices_shape[batch_rank] != grad_shape[batch_rank]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the indices size must be equal to grad first dimension size. But got indices size: "
                             << indices_shape[batch_rank] << ", grad first dimension size: " << grad_shape[batch_rank]
                             << ".";
  }

  if (grad_shape.size() > LongToSize(batch_rank) + 1) {
    ShapeVector var_indexed_shape(var_shape.begin() + batch_rank + 1, var_shape.end());
    ShapeVector grad_indexed_shape(grad_shape.begin() + batch_rank + 1, grad_shape.end());
    (void)CheckAndConvertUtils::CheckValue("var_shape[1:]", var_indexed_shape, kEqual, "grad_shape[1:]",
                                           grad_indexed_shape, prim_name);
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
}

TuplePtr SparseApplyAdagradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // Get all inputs's type
  auto var_type = input_args[0]->BuildType();
  auto accum_type = input_args[1]->BuildType();
  auto grad_type = input_args[2]->BuildType();
  auto indices_type = input_args[3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // Args accum and grad must have the same type as var
  std::map<std::string, TypePtr> args;
  args.insert({"var", var_type});
  args.insert({"accum", accum_type});
  args.insert({"grad", grad_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // Check indices_type
  std::map<std::string, TypePtr> args2;
  args2.insert({"indices", indices_type});
  const std::set<TypePtr> valid_types2 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args2, valid_types2, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

void SparseApplyAdagrad::Init(float lr, bool update_slots, bool use_locking) {
  set_lr(lr);
  set_update_slots(update_slots);
  set_use_locking(use_locking);
}

void SparseApplyAdagrad::set_lr(float lr) { (void)AddAttr(kAttrLr, api::MakeValue(lr)); }

float SparseApplyAdagrad::get_lr() const { return GetValue<float>(GetAttr(kAttrLr)); }

void SparseApplyAdagrad::set_update_slots(bool update_slots) {
  (void)AddAttr(kAttrUpdateSlots, api::MakeValue(update_slots));
}

bool SparseApplyAdagrad::get_update_slots() const { return GetValue<bool>(GetAttr(kAttrUpdateSlots)); }

void SparseApplyAdagrad::set_use_locking(bool use_locking) {
  (void)AddAttr(kAttrUseLocking, api::MakeValue(use_locking));
}

bool SparseApplyAdagrad::get_use_locking() const { return GetValue<bool>(GetAttr(kAttrUseLocking)); }

AbstractBasePtr SparseApplyAdagradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = SparseApplyAdagradInferType(primitive, input_args);
  auto infer_shape = SparseApplyAdagradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(SparseApplyAdagrad, BaseOperator);

// AG means auto generated
class MIND_API AGSparseApplyAdagradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyAdagradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyAdagradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SparseApplyAdagradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SparseApplyAdagrad, prim::kPrimSparseApplyAdagrad, AGSparseApplyAdagradInfer, false);
}  // namespace ops
}  // namespace mindspore
