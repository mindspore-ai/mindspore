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

#include "ops/grad/sparse_add_grad.h"
#include <set>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;

namespace {
constexpr size_t kSparseAddGradIndex0 = 0;
constexpr size_t kSparseAddGradIndex1 = 1;
constexpr size_t kSparseAddGradIndex2 = 2;
constexpr size_t kSparseAddGradIndex3 = 3;

inline void CheckSparseAddGradShape(const size_t sparse_shape_size, const size_t expected_dim,
                                    const std::string &arg_name, const std::string &op_name) {
  if (sparse_shape_size != expected_dim) {
    MS_EXCEPTION(mindspore::ValueError) << "For '" << op_name << "', " << arg_name << " must be a " << expected_dim
                                        << "-dimensional tensor, but got a " << sparse_shape_size
                                        << "-dimensional tensor.";
  }
}

mindspore::TypePtr SparseAddGradInferType(const std::string &op_name, const AbstractBasePtrList &args_spec_list,
                                          size_t index) {
  auto tensor = mindspore::abstract::CheckArg<AbstractTensor>(op_name, args_spec_list, index);
  return tensor->element()->BuildType();
}
}  // namespace

void SparseAddGrad::Init() {}

AbstractBasePtr SparseAddGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const size_t kInputNum = 4;
  constexpr size_t kIndicesShapeSize = 2;
  constexpr size_t kValuesShapeSize = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, name);

  auto type = SparseAddGradInferType(name, input_args, 0);
  const std::set<TypePtr> indices_valid_types = {kInt64};
  const std::set<TypePtr> values_valid_types = {kInt8,    kInt16,   kInt32,     kInt64,
                                                kFloat32, kFloat64, kComplex64, kComplex128};
  CheckAndConvertUtils::CheckTensorTypeValid("backprop_val_grad", type, values_valid_types, name);
  auto x1_type = SparseAddGradInferType(name, input_args, kSparseAddGradIndex1);
  CheckAndConvertUtils::CheckTensorTypeValid("x1_indices", x1_type, indices_valid_types, name);
  auto x2_type = SparseAddGradInferType(name, input_args, kSparseAddGradIndex2);
  CheckAndConvertUtils::CheckTensorTypeValid("x2_indices", x2_type, indices_valid_types, name);
  auto sum_type = SparseAddGradInferType(name, input_args, kSparseAddGradIndex3);
  CheckAndConvertUtils::CheckTensorTypeValid("sum_indices", sum_type, indices_valid_types, name);
  std::shared_ptr<AbstractTensor> dx1 = nullptr;
  std::shared_ptr<AbstractTensor> dx2 = nullptr;
  auto val_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  (void)CheckSparseAddGradShape(val_grad_shape[kShape].size(), kValuesShapeSize, "backprop_val_grad", name);
  auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSparseAddGradIndex1]->BuildShape());
  bool x1_is_dyn_shape = !x1_shape[kMaxShape].empty();
  auto dx1_shape = x1_shape[kShape];
  (void)CheckSparseAddGradShape(dx1_shape.size(), kIndicesShapeSize, "x1_indices", name);
  ShapeVector shp = {dx1_shape.at(0)};
  if (x1_is_dyn_shape) {
    auto dx1_min_shape = x1_shape[kMinShape];
    auto dx1_max_shape = x1_shape[kMaxShape];
    ShapeVector min_shp = {dx1_min_shape.at(0)};
    ShapeVector max_shp = {dx1_max_shape.at(0)};
    dx1 = std::make_shared<AbstractTensor>(type, std::make_shared<mindspore::abstract::Shape>(shp, min_shp, max_shp));
  } else {
    dx1 = std::make_shared<AbstractTensor>(type, std::make_shared<mindspore::abstract::Shape>(shp));
  }

  auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSparseAddGradIndex2]->BuildShape());
  bool x2_is_dyn_shape = !x2_shape[kMaxShape].empty();
  ShapeVector dx2_shape = x2_shape[kShape];
  (void)CheckSparseAddGradShape(dx2_shape.size(), kIndicesShapeSize, "x2_indices", name);
  shp = {dx2_shape.at(0)};
  if (x2_is_dyn_shape) {
    auto dx2_min_shape = x2_shape[kMinShape];
    auto dx2_max_shape = x2_shape[kMaxShape];
    ShapeVector min_shp = {dx2_min_shape.at(0)};
    ShapeVector max_shp = {dx2_max_shape.at(0)};
    dx2 = std::make_shared<AbstractTensor>(type, std::make_shared<mindspore::abstract::Shape>(shp, min_shp, max_shp));
  } else {
    dx2 = std::make_shared<AbstractTensor>(type, std::make_shared<mindspore::abstract::Shape>(shp));
  }
  auto sum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSparseAddGradIndex3]->BuildShape());
  (void)CheckSparseAddGradShape(sum_shape.size(), kIndicesShapeSize, "sum_indices", name);
  AbstractBasePtrList ret = {dx1, dx2};
  return std::make_shared<AbstractTuple>(ret);
}

MIND_API_OPERATOR_IMPL(SparseAddGrad, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseAddGrad, prim::kPrimSparseAddGrad, SparseAddGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
