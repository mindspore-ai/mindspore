/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/apply_came_part1.h"

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/type_id.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/nn_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kInputsNumPart1 = 2;
const int64_t kOutPutNumPart1 = 3;

std::vector<int64_t> CheckInputsShapePart1(const string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  int64_t m = abstract::Shape::kShapeDimAny;
  int64_t n = abstract::Shape::kShapeDimAny;
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (!IsDynamicRank(grad_shape)) {
    size_t expect_rank = 2;
    CheckAndConvertUtils::CheckInteger("rank of grad", grad_shape.size(), kEqual, expect_rank, op_name);
    m = grad_shape[grad_shape.size() - 1];
    n = grad_shape[grad_shape.size() - 2];
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(n);
  out_shape.push_back(m);
  return out_shape;
}

abstract::TupleShapePtr ApplyCamePart1InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNumPart1, op_name);
  size_t expect_rank = 2;
  ShapeVector out_shape = CheckInputsShapePart1(op_name, input_args);

  ShapeVector sum_grad_r_vec = out_shape;
  sum_grad_r_vec.pop_back();

  ShapeVector sum_grad_c_vec = out_shape;
  sum_grad_c_vec.erase(sum_grad_c_vec.begin() + expect_rank - 2);

  ShapeVector sum_grad_rc_vec = out_shape;
  sum_grad_rc_vec.erase(sum_grad_rc_vec.begin() + expect_rank - 2, sum_grad_rc_vec.end());

  abstract::BaseShapePtrList output_shape_ptr_list(kOutPutNumPart1);
  output_shape_ptr_list[0] = std::make_shared<abstract::Shape>(sum_grad_r_vec);
  output_shape_ptr_list[1] = std::make_shared<abstract::Shape>(sum_grad_c_vec);
  output_shape_ptr_list[2] = std::make_shared<abstract::Shape>(sum_grad_rc_vec);

  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TypePtr ApplyCamePart1InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  auto grad_type = input_args[kInputIndex0]->BuildType();
  auto eps_type = input_args[kInputIndex1]->BuildType();
  (void)types.emplace("grad", grad_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{eps_type, eps_type, eps_type});
}
}  // namespace

AbstractBasePtr ApplyCamePart1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNumPart1, primitive->name());
  auto infer_type = ApplyCamePart1InferType(primitive, input_args);
  auto infer_shape = ApplyCamePart1InferShape(primitive, input_args);
  auto shape_tuple = infer_shape->cast_ptr<abstract::TupleShape>();
  auto type_tuple = infer_type->cast_ptr<Tuple>();
  AbstractBasePtrList ptr_list;
  for (size_t i = 0; i < shape_tuple->size() - 1; i++) {
    auto tensor_it = abstract::MakeAbstract((*shape_tuple)[i], (*type_tuple)[i]);
    ptr_list.push_back(tensor_it);
  }
  auto sum_grad_rc_any = std::make_shared<abstract::AbstractScalar>(kValueAny, (*type_tuple)[type_tuple->size() - 1]);
  auto sum_grad_rc_tensor =
    std::make_shared<abstract::AbstractTensor>(sum_grad_rc_any, std::make_shared<abstract::Shape>());
  ptr_list.push_back(sum_grad_rc_tensor);
  return std::make_shared<abstract::AbstractTuple>(ptr_list);
}

MIND_API_OPERATOR_IMPL(ApplyCamePart1, BaseOperator);

// AG means auto generated
class MIND_API AGApplyCamePart1Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyCamePart1InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyCamePart1InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ApplyCamePart1Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApplyCamePart1, prim::kPrimApplyCamePart1, AGApplyCamePart1Infer, false);
}  // namespace ops
}  // namespace mindspore
