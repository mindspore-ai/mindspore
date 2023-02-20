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

#include "ops/lu.h"

#include <set>

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
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
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
abstract::TupleShapePtr LuInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr int64_t number1 = 1;
  constexpr int64_t number2 = 2;
  const int64_t input_num = 1;
  const int64_t rank = 2;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           prim_name);
  auto input_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kInputIndex0);
  auto input_shape = input_shape_ptr->shape();
  if (IsDynamicRank(input_shape)) {
    abstract::ShapePtr rank_shape = std::make_shared<abstract::Shape>(ShapeVector({-2}));
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{rank_shape, rank_shape});
  }
  std::vector<int64_t> p_shape(input_shape.begin(), (input_shape.end() - number1));
  abstract::ShapePtr p_shape_ptr = std::make_shared<abstract::Shape>(p_shape);
  auto input_rank = SizeToLong(input_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("input rank", input_rank, kGreaterEqual, rank, prim_name);
  int64_t size1 = input_shape[input_shape.size() - number1];
  int64_t size2 = input_shape[input_shape.size() - number2];
  if (size1 != size2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', input_shape[-1] and input_shape[-2] must be same, but got " << size1 << " vs "
                             << size2;
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{input_shape_ptr, p_shape_ptr});
}

TypePtr LuInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> lu_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  auto input_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input type", input_type, lu_types, prim->name());
  const std::set<TypePtr> out_valid_types = {kInt32, kInt64};
  ValuePtr out_type_value = prim->GetAttr("output_idx_type");
  TypePtr type = dyn_cast<Type>(out_type_value);
  (void)CheckAndConvertUtils::CheckTypeValid("p type", type, out_valid_types, prim->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_type, type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(Lu, BaseOperator);
AbstractBasePtr LuInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                        const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = LuInferType(primitive, input_args);
  auto infer_shape = LuInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLuInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LuInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LuInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LuInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Lu, prim::kPrimLu, AGLuInfer, false);
}  // namespace ops
}  // namespace mindspore
