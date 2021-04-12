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

#include <functional>
#include "ops/where.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr WhereInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto Where_prim = primitive->cast<PrimWherePtr>();
  MS_EXCEPTION_IF_NULL(Where_prim);
  for (auto input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  auto op_name = Where_prim->name();
  CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kGreaterEqual, 3, op_name);
  auto input0_type_ = input_args[0]->BuildType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input0_type_);
  auto input0_type = input0_type_->element();
  auto input0_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input0_shape", input_args[0]->BuildShape(), op_name);
  auto num = input_args[0]->BuildValue()->cast<tensor::TensorPtr>()->ElementsNum();
  auto input1_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input1_shape", input_args[1]->BuildShape(), op_name);
  auto num1 = input_args[1]->BuildValue()->cast<tensor::TensorPtr>()->ElementsNum();
  auto input2_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input2_shape", input_args[2]->BuildShape(), op_name);
  auto num2 = input_args[2]->BuildValue()->cast<tensor::TensorPtr>()->ElementsNum();
  int64_t nummax = num > num1 ? num : (num1 > num2 ? num1 : num2);
  int64_t axisout = 0;
  int64_t temp = 0;
  for (int64_t j = 0; j < (int64_t)input0_shape.size(); j++) {
    if (input0_shape[j] == input1_shape[j] && input0_shape[j] != input2_shape[j]) {
      axisout = j;
      break;
    }
    if (input0_shape[j] == input2_shape[j] && input0_shape[j] != input1_shape[j]) {
      axisout = j;
      break;
    }
    if (input1_shape[j] != input2_shape[j] && input0_shape[j] == input1_shape[j]) {
      axisout = j;
      break;
    }
    temp += 1;
    if (temp == (int64_t)input0_shape.size()) {
      return std::make_shared<abstract::AbstractTensor>(input0_type, input0_shape);
    }
  }
  input0_shape[axisout] = nummax;
  return std::make_shared<abstract::AbstractTensor>(input0_type, input0_shape);
}
REGISTER_PRIMITIVE_C(kNameWhere, Where);
}  // namespace ops
}  // namespace mindspore
