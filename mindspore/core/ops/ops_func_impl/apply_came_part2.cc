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

#include "ops/ops_func_impl/apply_came_part2.h"

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
#include "ops/auto_generate/gen_ops_name.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {

const int64_t kInputsNumOfPart2 = 9;
const int64_t kOutPutNumOfPart2 = 4;
const int kConstNumberZeroPart2 = 0;
const int kConstNumberOnePart2 = 1;
const int kConstNumberTwoPart2 = 2;
const int kConstNumberThreePart2 = 3;

int64_t CheckInputDimPart2(int64_t dim, int64_t goldValue, string dimStr, string opName) {
  if (dim == abstract::Shape::kShapeDimAny) {
    return goldValue;
  }
  if (goldValue == abstract::Shape::kShapeDimAny) {
    goldValue = dim;
  } else {
    CheckAndConvertUtils::CheckInteger(dimStr, dim, kEqual, goldValue, opName);
  }
  return goldValue;
}

std::vector<int64_t> CheckInputsShapePart2(const string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  int64_t m = abstract::Shape::kShapeDimAny;
  int64_t n = abstract::Shape::kShapeDimAny;
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (!IsDynamicRank(grad_shape)) {
    size_t expect_rank = 2;
    CheckAndConvertUtils::CheckInteger("rank of grad", grad_shape.size(), kEqual, expect_rank, op_name);
    m = grad_shape[grad_shape.size() - 1];
    n = grad_shape[grad_shape.size() - 2];
  }
  auto sum_grad_r_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  if (!IsDynamicRank(sum_grad_r_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of sum_grad_r", sum_grad_r_shape.size(), kEqual, 1, op_name);
    n = CheckInputDimPart2(sum_grad_r_shape[sum_grad_r_shape.size() - 1], n, "last dim of sum_grad_r", op_name);
  }
  auto sum_grad_c_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (!IsDynamicRank(sum_grad_c_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of sum_grad_c", sum_grad_c_shape.size(), kEqual, 1, op_name);
    m = CheckInputDimPart2(sum_grad_c_shape[sum_grad_c_shape.size() - 1], m, "last dim of sum_grad_c", op_name);
  }

  auto r_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  if (!IsDynamicRank(r_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of r", r_shape.size(), kEqual, 1, op_name);
    n = CheckInputDimPart2(r_shape[r_shape.size() - 1], n, "last dim of r", op_name);
  }
  auto c_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  if (!IsDynamicRank(c_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of c", c_shape.size(), kEqual, 1, op_name);
    m = CheckInputDimPart2(c_shape[c_shape.size() - 1], m, "last dim of c", op_name);
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(n);
  out_shape.push_back(m);
  return out_shape;
}

BaseShapePtr ApplyCamePart2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNumOfPart2, op_name);
  size_t expect_rank = 2;
  ShapeVector out_shape = CheckInputsShapePart2(op_name, input_args);
  ShapeVector r_vec = out_shape;
  r_vec.pop_back();
  ShapeVector c_vec = out_shape;
  c_vec.erase(c_vec.begin() + expect_rank - 2);
  ShapeVector u_vec = out_shape;
  ShapeVector sum_square_u_vec = {1};
  abstract::BaseShapePtrList output_shape_ptr_list(kOutPutNumOfPart2);
  output_shape_ptr_list[kConstNumberZeroPart2] = std::make_shared<abstract::Shape>(r_vec);
  output_shape_ptr_list[kConstNumberOnePart2] = std::make_shared<abstract::Shape>(c_vec);
  output_shape_ptr_list[kConstNumberTwoPart2] = std::make_shared<abstract::Shape>(u_vec);
  output_shape_ptr_list[kConstNumberThreePart2] = std::make_shared<abstract::Shape>(sum_square_u_vec);
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TypePtr ApplyCamePart2FuncImpl::InferType(const PrimitivePtr &prim,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = prim->name();
  auto grad_type = input_args[kInputIndex0]->BuildType();
  auto r_type = input_args[kInputIndex4]->BuildType();
  auto c_type = input_args[kInputIndex4]->BuildType();
  std::map<std::string, TypePtr> types = {{"grad", grad_type}, {"r", r_type}, {"c", c_type}};
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  auto sum_grad_r_type = input_args[kInputIndex1]->BuildType();
  auto sum_grad_c_type = input_args[kInputIndex2]->BuildType();
  auto sum_grad_rc_type = input_args[kInputIndex3]->BuildType();
  std::map<std::string, TypePtr> other_types = {
    {"sum_grad_r", sum_grad_r_type}, {"sum_grad_c", sum_grad_c_type}, {"sum_grad_rc", sum_grad_rc_type}};
  const std::set<TypePtr> other_valid_types = {kFloat32};
  auto other_type = CheckAndConvertUtils::CheckTensorTypeSame(other_types, other_valid_types, op_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type, other_type, other_type});
}

}  // namespace ops
}  // namespace mindspore
