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

#include "ops/ops_func_impl/apply_came_part3.h"

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
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"

namespace mindspore {
namespace ops {

const int64_t kInputsNumPart3 = 8;
const int64_t kOutPutNumPart3 = 4;
const int kConstNumberZeroPart3 = 0;
const int kConstNumberOnePart3 = 1;
const int kConstNumberTwoPart3 = 2;
const int kConstNumberThreePart3 = 3;

int64_t CheckInputDimPart3(int64_t dim, int64_t goldValue, string dimStr, string opName) {
  if (dim == abstract::Shape::kShapeDimAny) {
    return dim;
  }
  if (goldValue == abstract::Shape::kShapeDimAny) {
    goldValue = dim;
  } else {
    CheckAndConvertUtils::CheckInteger(dimStr, dim, kEqual, goldValue, opName);
  }
  return goldValue;
}

std::vector<int64_t> CheckInputsShapePart3(const string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  int64_t m = abstract::Shape::kShapeDimAny;
  int64_t n = abstract::Shape::kShapeDimAny;
  size_t expect_rank = 2;
  auto u_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (!IsDynamicRank(u_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of u", u_shape.size(), kEqual, expect_rank, op_name);
    m = u_shape[u_shape.size() - 1];
    n = u_shape[u_shape.size() - 2];
  }
  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  if (!IsDynamicRank(m_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of m", m_shape.size(), kEqual, expect_rank, op_name);
    m = CheckInputDimPart3(m_shape[m_shape.size() - 1], m, "last dim of m", op_name);
    n = CheckInputDimPart3(m_shape[m_shape.size() - 2], n, "last but onedim of m", op_name);
  }
  auto sum_square_u_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  if (!IsDynamicRank(sum_square_u_shape)) {
    CheckAndConvertUtils::CheckInteger("rank of sum_square_u", sum_square_u_shape.size(), kEqual, 1, op_name);
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(n);
  out_shape.push_back(m);
  return out_shape;
}

BaseShapePtr ApplyCamePart3FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNumPart3, op_name);
  size_t expect_rank = 2;
  ShapeVector out_shape = CheckInputsShapePart3(op_name, input_args);

  ShapeVector m_vec = out_shape;
  ShapeVector sum_u_r_vec = out_shape;
  sum_u_r_vec.pop_back();
  ShapeVector sum_u_c_vec = out_shape;
  sum_u_c_vec.erase(sum_u_c_vec.begin() + expect_rank - 2);

  ShapeVector sum_u_rc_vec = out_shape;
  sum_u_rc_vec.erase(sum_u_rc_vec.begin() + expect_rank - 2, sum_u_rc_vec.end());
  abstract::BaseShapePtrList output_shape_ptr_list(kOutPutNumPart3);
  output_shape_ptr_list[kConstNumberZeroPart3] = std::make_shared<abstract::Shape>(m_vec);
  output_shape_ptr_list[kConstNumberOnePart3] = std::make_shared<abstract::Shape>(sum_u_r_vec);
  output_shape_ptr_list[kConstNumberTwoPart3] = std::make_shared<abstract::Shape>(sum_u_c_vec);
  output_shape_ptr_list[kConstNumberThreePart3] = std::make_shared<abstract::Shape>(sum_u_rc_vec);
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TypePtr ApplyCamePart3FuncImpl::InferType(const PrimitivePtr &prim,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  auto u_type = input_args[kInputIndex0]->BuildType();
  auto m_type = input_args[kInputIndex1]->BuildType();
  auto sum_square_u_type = input_args[kInputIndex5]->BuildType();
  (void)types.emplace("u", u_type);
  (void)types.emplace("m", m_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{u_type, sum_square_u_type, sum_square_u_type, sum_square_u_type});
}

}  // namespace ops
}  // namespace mindspore
