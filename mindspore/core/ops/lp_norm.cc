/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include "ops/lp_norm.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LpNormInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto output_shape = input_shape;
  auto input_rank = SizeToLong(input_shape.size());
  auto axis = GetValue<std::vector<int64_t>>(primitive->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(primitive->GetAttr("keep_dims"));
  if (input_rank == 0) {
    CheckAndConvertUtils::CheckInteger("axis size", axis.size(), kEqual, input_rank + 1, prim_name);
    return std::make_shared<abstract::Shape>(input_shape);
  } else {
    CheckAndConvertUtils::CheckInRange("axis size", axis.size(), kIncludeNeither, {0, input_rank + 1}, prim_name);
  }
  if (axis.size() > 1) {
    for (size_t i = 0; i < axis.size(); ++i) {
      CheckAndConvertUtils::CheckInRange("axis value", axis[i], kIncludeLeft, {-input_rank, input_rank}, prim_name);
      if (axis[i] < 0) {
        axis[i] += input_rank;
      }
    }
    for (size_t i = 0; i < axis.size(); ++i) {
      auto temp = axis;
      auto idx = std::find(temp.begin(), temp.end(), axis[i]);
      (void)temp.erase(idx);
      auto re_idx = std::find(temp.begin(), temp.end(), axis[i]);
      if (re_idx != temp.end()) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', The element of the axis should be different";
      }
      if (keep_dims == false) {
        output_shape[axis[i]] = -1;
      } else {
        output_shape[axis[i]] = 1;
      }
    }
    if (keep_dims == false) {
      for (std::vector<int64_t>::iterator iter = output_shape.begin(); iter != output_shape.end(); ++iter) {
        if (*iter == -1) {
          iter = output_shape.erase(iter);
          iter -= 1;
        }
      }
    }
  } else {
    if (axis[0] < 0) {
      axis[0] += input_rank;
    }
    if (keep_dims == false) {
      (void)output_shape.erase(output_shape.begin() + axis[0]);
    } else {
      output_shape[axis[0]] = 1;
    }
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr LpNormInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  CheckAndConvertUtils::CheckTensorTypeValid("input", infer_type, valid_types, prim->name());
  return infer_type;
}
}  // namespace

AbstractBasePtr LpNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = LpNormInferType(primitive, input_args);
  auto infer_shape = LpNormInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(LpNorm, prim::kPrimLpNorm, LpNormInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
