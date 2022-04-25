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

#include "ops/reduce_std.h"
#include <set>

#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr ReduceStdInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto input_rank = SizeToLong(input_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("input_rank", input_rank, kGreaterEqual, 1, prim_name);
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(primitive->GetAttr("keep_dims"));
  auto temp_shape = input_shape;
  (void)CheckAndConvertUtils::CheckInRange("axis size", axis.size(), kIncludeLeft, {0, input_rank + 1}, prim_name);
  if (axis.size() == 0) {
    for (size_t i = 0; i < input_shape.size(); i++) {
      axis.push_back(i);
    }
  } else {
    for (size_t i = 0; i < axis.size(); ++i) {
      (void)CheckAndConvertUtils::CheckInRange("axis value", axis[i], kIncludeLeft, {-input_rank, input_rank},
                                               prim_name);
      if (axis[i] < 0) {
        axis[i] += input_rank;
      }
    }
    for (size_t i = 0; i < axis.size(); ++i) {
      auto temp = axis;
      auto idx = std::find(temp.begin(), temp.end(), axis[i]);
      temp.erase(idx);
      auto re_idx = std::find(temp.begin(), temp.end(), axis[i]);
      if (re_idx != temp.end()) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the elements in attribute axis should be different.";
      }
    }
  }
  for (size_t i = 0; i < axis.size(); ++i) {
    if (!keep_dims) {
      temp_shape[axis[i]] = -1;
    } else {
      temp_shape[axis[i]] = 1;
    }
  }
  if (!keep_dims) {
    for (std::vector<int64_t>::iterator iter = temp_shape.begin(); iter != temp_shape.end(); ++iter) {
      if (*iter == -1) {
        iter = temp_shape.erase(iter);
        iter -= 1;
      }
    }
  }
  abstract::ShapePtr output_shape = std::make_shared<abstract::Shape>(temp_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{output_shape, output_shape});
}

TuplePtr ReduceStdInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16};
  auto type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTypeValid("input_x", type, valid_types, name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(ReduceStd, BaseOperator);
AbstractBasePtr ReduceStdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = ReduceStdInferType(primitive, input_args);
  auto infer_shape = ReduceStdInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ReduceStd, prim::kPrimReduceStd, ReduceStdInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
