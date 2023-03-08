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

#include "ops/self_adjoint_eig.h"
#include <complex>
#include <map>
#include <string>
#include <set>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SelfAdjointEigInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kNumber = 2;
  auto x = input_args[0]->BuildShape();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x)[kShape];
  auto input_rank = SizeToLong(input_shape.size());

  if (IsDynamicRank(input_shape)) {
    auto unknow_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknow_shape_ptr, unknow_shape_ptr});
  }

  CheckAndConvertUtils::CheckInteger("input rank", input_rank, kGreaterEqual, kNumber, prim_name);
  int64_t last_shape_input = input_shape[input_rank - 1];
  int64_t last_second__shape_input = input_shape[input_rank - 2];
  // Check whether the innermost matrix is square
  if (last_shape_input != last_second__shape_input) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the last dimension of the input "
                             << "and the last second dimension of the input "
                             << "should be the same in this case, but got " << last_shape_input << " and "
                             << last_second__shape_input << ".";
  }
  // Determine the output shape based on the attributes
  std::vector<int64_t> out_shape_e;
  auto compute_v_ptr = primitive->GetAttr("compute_v");
  MS_EXCEPTION_IF_NULL(compute_v_ptr);
  for (int64_t i = 0; i < input_rank - 1; i++) {
    out_shape_e.push_back(input_shape[i]);
  }
  abstract::ShapePtr Out_shape_e = std::make_shared<abstract::Shape>(out_shape_e);
  abstract::ShapePtr Out_shape_v = x->cast<abstract::ShapePtr>();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{Out_shape_e, Out_shape_v});
}

TuplePtr SelfAdjointEigInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  // Determine the output shape based on the attributes
  auto compute_v_ptr = prim->GetAttr("compute_v");
  MS_EXCEPTION_IF_NULL(compute_v_ptr);
  // Determine the number of outputs according to the attribute
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type});
}
}  // namespace

AbstractBasePtr SelfAdjointEigInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto shape = SelfAdjointEigInferShape(primitive, input_args);
  auto type = SelfAdjointEigInferType(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
MIND_API_OPERATOR_IMPL(SelfAdjointEig, BaseOperator);

// AG means auto generated
class MIND_API AGSelfAdjointEigInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SelfAdjointEigInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SelfAdjointEigInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SelfAdjointEigInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SelfAdjointEig, prim::kPrimSelfAdjointEig, AGSelfAdjointEigInfer, false);
}  // namespace ops
}  // namespace mindspore
