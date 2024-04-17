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

#include "ops/ops_func_impl/apply_came_part1.h"
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

const int64_t kInputsNumPart1 = 2;
const int64_t kOutPutNumPart1 = 3;
const int kConstNumberZeroPart1 = 0;
const int kConstNumberOnePart1 = 1;
const int kConstNumberTwoPart1 = 2;

std::vector<int64_t> CheckInputsShapePart1(const string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  int64_t m = abstract::Shape::kShapeDimAny;
  int64_t n = abstract::Shape::kShapeDimAny;
  auto grad_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto grad_shape = grad_shape_ptr->GetShapeVector();
  if (!IsDynamicRank(grad_shape) && grad_shape.size() != 0) {
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

BaseShapePtr ApplyCamePart1FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
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
  output_shape_ptr_list[kConstNumberZeroPart1] = std::make_shared<abstract::Shape>(sum_grad_r_vec);
  output_shape_ptr_list[kConstNumberOnePart1] = std::make_shared<abstract::Shape>(sum_grad_c_vec);
  output_shape_ptr_list[kConstNumberTwoPart1] = std::make_shared<abstract::Shape>(sum_grad_rc_vec);
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TypePtr ApplyCamePart1FuncImpl::InferType(const PrimitivePtr &prim,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;
  auto grad_type = input_args[kInputIndex0]->BuildType();
  auto eps_type = input_args[kInputIndex1]->BuildType();
  (void)types.emplace("grad", grad_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  auto eps_tensor_type = std::make_shared<TensorType>(eps_type);
  return std::make_shared<Tuple>(std::vector<TypePtr>{eps_tensor_type, eps_tensor_type, eps_tensor_type});
}

}  // namespace ops
}  // namespace mindspore
