/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ops/reduce_all.h"
#include "ops/reduce_any.h"
#include "ops/reduce_max.h"
#include "ops/reduce_min.h"
#include "ops/reduce_sum.h"
#include "ops/reduce_prod.h"
#include "ops/reduce_mean.h"
#include <string>
#include <set>
#include <memory>
#include "ops/op_utils.h"
#include "abstract/ops/op_infer.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ReduceAll, Reduce);
MIND_API_OPERATOR_IMPL(ReduceAny, Reduce);
MIND_API_OPERATOR_IMPL(ReduceMax, Reduce);
MIND_API_OPERATOR_IMPL(ReduceMin, Reduce);
MIND_API_OPERATOR_IMPL(ReduceSum, Reduce);
MIND_API_OPERATOR_IMPL(ReduceProd, Reduce);
MIND_API_OPERATOR_IMPL(ReduceMean, Reduce);
class ReduceArithmeticInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const int64_t input_num = 1;
    MS_EXCEPTION_IF_NULL(primitive);
    CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                       primitive->name());
    return ReduceBaseInferShape(primitive, input_args, kNameReduceAll);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    std::set<TypePtr> bool_types = {kBool};
    const std::string &op_name = primitive->name();
    static const std::map<std::string, std::set<TypePtr>> check_list_map{
      {prim::kPrimReduceAll->name(), bool_types},
      {prim::kPrimReduceAny->name(), bool_types},
      {prim::kPrimReduceMax->name(), common_valid_types_with_complex_and_bool},
      {prim::kPrimReduceMin->name(), common_valid_types_with_complex_and_bool},
      {prim::kPrimReduceSum->name(), common_valid_types_with_complex_and_bool},
      {prim::kPrimReduceProd->name(), common_valid_types_with_complex},
      {prim::kPrimReduceMean->name(), common_valid_types_with_complex},
    };
    if (check_list_map.find(op_name) == check_list_map.end()) {
      MS_EXCEPTION(TypeError) << "For Primitive[" << op_name << "], the current ops do not support this operation.";
    }
    return ReduceBaseInferType(primitive, input_args, check_list_map.at(op_name));
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceAll, prim::kPrimReduceAll, ReduceArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceAny, prim::kPrimReduceAny, ReduceArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceMax, prim::kPrimReduceMax, ReduceArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceMin, prim::kPrimReduceMin, ReduceArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceSum, prim::kPrimReduceSum, ReduceArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceProd, prim::kPrimReduceProd, ReduceArithmeticInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceMean, prim::kPrimReduceMean, ReduceArithmeticInfer, false);
}  // namespace ops
}  // namespace mindspore
