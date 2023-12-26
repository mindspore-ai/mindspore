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

#include "ops/ops_func_impl/apply_came_part4.h"

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
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"

namespace mindspore {
namespace ops {

const int64_t kInputsNumOfPart4 = 12;
const int64_t kOutPutNumOfPart4 = 3;
const int kConstNumberZeroPart4 = 0;
const int kConstNumberOnePart4 = 1;
const int kConstNumberTwoPart4 = 2;

int64_t CheckInputDimPart4(int64_t dim, int64_t goldValue, string dimStr, string opName) {
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

std::vector<int64_t> CheckInputsShapePart4(const string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  int64_t m = abstract::Shape::kShapeDimAny;
  int64_t n = abstract::Shape::kShapeDimAny;

  auto param_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (!IsDynamicRank(param_shape)) {
    size_t expect_rank = 2;
    CheckAndConvertUtils::CheckInteger("rank of param", param_shape.size(), kEqual, expect_rank, op_name);
    m = param_shape[param_shape.size() - 1];
    n = param_shape[param_shape.size() - 2];
  }
  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  if (!IsDynamicRank(m_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of m", m_shape.size(), kEqual, 2, op_name);
    m = CheckInputDimPart4(m_shape[m_shape.size() - 1], m, "last dim of m", op_name);
    n = CheckInputDimPart4(m_shape[m_shape.size() - 2], n, "penultimate dim of m", op_name);
  }
  auto r_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (!IsDynamicRank(r_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of r", r_shape.size(), kEqual, 1, op_name);
    n = CheckInputDimPart4(r_shape[r_shape.size() - 1], n, "last dim of r", op_name);
  }
  auto c_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  if (!IsDynamicRank(c_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of c", c_shape.size(), kEqual, 1, op_name);
    m = CheckInputDimPart4(c_shape[c_shape.size() - 1], m, "last dim of c", op_name);
  }
  auto sum_u_r_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex8]->BuildShape())[kShape];
  if (!IsDynamicRank(sum_u_r_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of sum_u_r", sum_u_r_shape.size(), kEqual, 1, op_name);
    n = CheckInputDimPart4(sum_u_r_shape[sum_u_r_shape.size() - 1], n, "last dim of sum_u_r_shape", op_name);
  }
  auto sum_u_c_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex9]->BuildShape())[kShape];
  if (!IsDynamicRank(sum_u_c_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of sum_u_c", sum_u_c_shape.size(), kEqual, 1, op_name);
    m = CheckInputDimPart4(sum_u_c_shape[sum_u_c_shape.size() - 1], m, "last dim of sum_u_c_shape", op_name);
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(n);
  out_shape.push_back(m);
  return out_shape;
}

BaseShapePtr ApplyCamePart4FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNumOfPart4, op_name);
  size_t expect_rank = 2;
  ShapeVector out_shape = CheckInputsShapePart4(op_name, input_args);
  ShapeVector param_shape = out_shape;
  ShapeVector r_shape = out_shape;
  r_shape.pop_back();
  ShapeVector c_shape = out_shape;
  c_shape.erase(c_shape.begin() + expect_rank - 2);
  abstract::BaseShapePtrList output_shape_ptr_list(kOutPutNumOfPart4);
  output_shape_ptr_list[kConstNumberZeroPart4] = std::make_shared<abstract::Shape>(param_shape);
  output_shape_ptr_list[kConstNumberOnePart4] = std::make_shared<abstract::Shape>(r_shape);
  output_shape_ptr_list[kConstNumberTwoPart4] = std::make_shared<abstract::Shape>(c_shape);
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TypePtr ApplyCamePart4FuncImpl::InferType(const PrimitivePtr &prim,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = prim->name();

  auto param_type = input_args[kInputIndex0]->BuildType();
  auto m_type = input_args[kInputIndex1]->BuildType();
  auto r_type = input_args[kInputIndex2]->BuildType();
  auto c_type = input_args[kInputIndex3]->BuildType();
  std::map<std::string, TypePtr> types{{"param", param_type}, {"m", m_type}, {"r", r_type}, {"c", c_type}};
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  auto sum_u_r_type = input_args[kInputIndex8]->BuildType();
  auto sum_u_c_type = input_args[kInputIndex9]->BuildType();
  auto sum_u_rc_type = input_args[kInputIndex10]->BuildType();
  std::map<std::string, TypePtr> other_types{
    {"sum_u_r", sum_u_r_type}, {"sum_u_c", sum_u_c_type}, {"sum_u_rc", sum_u_rc_type}};
  const std::set<TypePtr> other_valid_types = {kFloat32};
  CheckAndConvertUtils::CheckTensorTypeSame(other_types, other_valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type, type});
}

}  // namespace ops
}  // namespace mindspore
