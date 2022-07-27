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
#include "ops/grad/sparse_slice_grad.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <map>
#include <vector>
#include "abstract/param_validator.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {

enum DimNum : size_t {
  dim0Num = 0,
  dim1Num,
  dim2Num,
};

void CheckInputTensor(const std::vector<AbstractBasePtr> &input_args) {
  auto backprop_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto start_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto new_indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  if (indices_shape.size() != dim2Num) {
    MS_EXCEPTION(ValueError) << "For SparseSliceGrad, indices should be a 2-D tensor"
                             << ", while input_indices dim num is " << indices_shape.size() << ".";
  }
  if (indices_shape[1] != dim2Num) {
    MS_EXCEPTION(ValueError) << "For SparseSliceGrad, indices shape should be (2, n)"
                             << ", while input_indices shape dim0 is " << indices_shape[0] << ".";
  }
  if (backprop_shape.size() != dim1Num) {
    MS_EXCEPTION(ValueError) << "For SparseSliceGrad, backprop_val_grad should be a 1-D tensor"
                             << ",  while input_backprop_val_grad dim num is " << backprop_shape.size() << ".";
  }
  if (start_shape[0] != dim2Num) {
    MS_EXCEPTION(ValueError) << "For SparseSliceGrad, start should be a 2-D tensor"
                             << ", while dim num is " << start_shape.size() << ".";
  }
  if (new_indices_shape.size() != dim2Num) {
    MS_EXCEPTION(ValueError) << "For SparseSliceGrad, new_indices should be a 2-D tensor"
                             << ", while input_new_indices dim num is " << new_indices_shape.size() << ".";
  }
  if (new_indices_shape[1] != dim2Num) {
    MS_EXCEPTION(ValueError) << "For SparseSliceGrad, new_indices shape should be (2, n)"
                             << ", while new_indices_indices shape dim0 is " << new_indices_shape[0] << ".";
  }
}

bool IsDynamic(const ShapeVector &shape) {
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    return true;
  }
  return false;
}

abstract::ShapePtr SparseSliceGradInferShape(const PrimitivePtr &primitive, 
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto new_indices_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("SparseSliceGrad", input_args, 3);
  MS_EXCEPTION_IF_NULL(new_indices_shape_ptr);
  auto new_indices_shape = new_indices_shape_ptr->shape();
  auto backprop_shape_ptr =CheckAndConvertUtils::GetTensorInputShape("SparseSliceGrad", input_args, 0);
  auto backprop_shape  = backprop_shape_ptr->shape();
  if (!(IsDynamic(backprop_shape)) && !(IsDynamic(new_indices_shape))) {
    CheckInputTensor(input_args);
  } else {
    backprop_shape = {abstract::Shape::SHP_ANY};
    new_indices_shape = {abstract::Shape::SHP_ANY, 2};
  }
  auto indices_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto indices_shape = indices_shape_map[kShape];
  int64_t output_shape = indices_shape[0];
  std::vector<int64_t> output_values_shape = {output_shape};
  return std::make_shared<abstract::Shape>(output_values_shape);
}

TypePtr SparseSliceGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("backprop_val_grad", input_args[kInputIndex0]->BuildType(),
                                                   {kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat32, kFloat64},
                                                   prim_name);                                             
  std::map<std::string, TypePtr> in_args = {{"indices", input_args[kInputIndex1]->BuildType()},
                                            {"start", input_args[kInputIndex2]->BuildType()},
                                            {"new_indices", input_args[kInputIndex3]->BuildType()}};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(in_args, {kInt64, kInt32}, prim_name);
  auto output_type = input_args[kInputIndex0]->BuildType();
  return output_type;
}
}  // namespace

AbstractBasePtr SparseSliceGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive, 
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = SparseSliceGradInferType(primitive, input_args);
  auto infer_shape = SparseSliceGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(SparseSliceGrad, prim::kPrimSparseSliceGrad, SparseSliceGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore