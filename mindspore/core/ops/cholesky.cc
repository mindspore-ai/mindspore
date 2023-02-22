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

#include "ops/cholesky.h"

#include <memory>
#include <set>

#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CholeskyInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto input_x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  auto x_shape = input_x->shape();
  MS_EXCEPTION_IF_NULL(x_shape);
  auto const &x_shape_list = x_shape->shape();
  // support dynamic rank
  if (IsDynamicRank(x_shape_list)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  const size_t x_dim = x_shape_list.size();
  constexpr size_t kDefaultRank = 2;
  constexpr size_t kRowIndex = 2;
  constexpr size_t kColIndex = 1;
  if (x_dim < kDefaultRank) {
    MS_EXCEPTION(ValueError) << "For Cholesky, the dimension of input x must be greater than or "
                             << "equal to 2"
                             << ", but got a " << x_dim << "-D Tensor.";
  }
  // support dynamic shape
  if (IsDynamic(x_shape_list)) {
    return input_x->BuildShape()->cast<abstract::ShapePtr>();
  }
  if (x_shape_list[x_dim - kColIndex] != x_shape_list[x_dim - kRowIndex]) {
    MS_EXCEPTION(ValueError) << "For Cholesky, input x must be batch squares"
                             << ", but got batch " << x_shape_list[x_dim - kRowIndex] << " x "
                             << x_shape_list[x_dim - kColIndex] << " matrices.";
  }
  return std::make_shared<abstract::Shape>(x_shape_list);
}

TypePtr CholeskyInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, op_name);
}
}  // namespace

void Cholesky::set_upper(const bool upper) { (void)this->AddAttr("upper", api::MakeValue(upper)); }

bool Cholesky::get_upper() const { return GetValue<bool>(GetAttr("upper")); }

void Cholesky::Init(const bool upper) { set_upper(upper); }

MIND_API_OPERATOR_IMPL(Cholesky, BaseOperator);
AbstractBasePtr CholeskyInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto infer_type = CholeskyInferType(primitive, input_args);
  auto infer_shape = CholeskyInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGCholeskyInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CholeskyInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Cholesky, prim::kPrimCholesky, AGCholeskyInfer, false);
}  // namespace ops
}  // namespace mindspore
