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

#include "ops/eigh.h"

#include <algorithm>
#include <memory>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
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
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::BaseShapePtr EighInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto input_x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  auto x_shape = input_x->shape();
  MS_EXCEPTION_IF_NULL(x_shape);
  constexpr size_t kDefaultRank = 2;
  constexpr size_t kRowIndex = 2;
  constexpr size_t kColIndex = 1;
  auto const &x_shape_list = x_shape->shape();
  const size_t x_rank = x_shape_list.size();
  if (x_rank < kDefaultRank) {
    MS_EXCEPTION(ValueError) << "For Eig, x should be at least rank 2"
                             << ", but got a " << x_rank << "-D Tensor.";
  }
  if (x_shape_list[x_rank - kRowIndex] != x_shape_list[x_rank - kColIndex]) {
    MS_EXCEPTION(ValueError) << "For Eig, x should be square(squares)"
                             << ", but got " << x_shape_list[x_rank - kRowIndex] << " × "
                             << x_shape_list[x_rank - kColIndex] << " matrix(matrices).";
  }
  auto compute_eigenvectors = GetValue<bool>(primitive->GetAttr("compute_eigenvectors"));
  std::vector<BaseShapePtr> shapes_list;
  if (compute_eigenvectors) {
    ShapeVector val_shape_list_0;
    val_shape_list_0.push_back(x_shape_list[0]);
    ShapeVector val_shape_list_1;
    val_shape_list_1.push_back(x_shape_list[0]);
    val_shape_list_1.push_back(x_shape_list[0]);
    (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(val_shape_list_0));
    (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(val_shape_list_1));
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  ShapeVector val_shape_list_0{x_shape_list[0]};
  return std::make_shared<abstract::Shape>(val_shape_list_0);
}

TypePtr EighInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, op_name);
  std::vector<TypePtr> types_list;
  auto compute_eigenvectors = GetValue<bool>(primitive->GetAttr("compute_eigenvectors"));
  if (!compute_eigenvectors) {
    return x_type;
  }
  types_list = {x_type, x_type};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Eigh, BaseOperator);
AbstractBasePtr EighInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto infer_type = EighInferType(primitive, input_args);
  auto infer_shape = EighInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool Eigh::get_compute_eigen_vectors() const {
  auto value_ptr = this->GetAttr(KComputeEigenvectors);
  return GetValue<bool>(value_ptr);
}

bool Eigh::get_lower() const {
  auto value_ptr = this->GetAttr(kLower);
  return GetValue<bool>(value_ptr);
}

// AG means auto generated
class MIND_API AGEighInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return EighInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EighInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return EighInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Eigh, prim::kPrimEigh, AGEighInfer, false);
}  // namespace ops
}  // namespace mindspore
