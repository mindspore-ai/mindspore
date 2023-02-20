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

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
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
                                    const std::string &arg_name, const std::string &op_name, bool is_dyn_rank) {
  if (!is_dyn_rank && sparse_shape_size != expected_dim) {
    MS_EXCEPTION(mindspore::ValueError) << "For '" << op_name << "', " << arg_name << " must be a " << expected_dim
                                        << "-dimensional tensor, but got a " << sparse_shape_size
                                        << "-dimensional tensor.";
  }
}

inline void CheckSparseAddGradNNZ(const int64_t indices_nnz, const int64_t value_nnz, const std::string &indices_name,
                                  const std::string &value_name, const std::string &op_name) {
  if (indices_nnz != value_nnz) {
    MS_EXCEPTION(mindspore::ValueError) << "For " << op_name << ", the length of " << indices_name << " and "
                                        << value_name << " must be same, but got length of " << indices_name << " is "
                                        << indices_nnz << ", and length of " << value_name << " is " << value_nnz;
  }
}

mindspore::TypePtr SparseAddGradInferType(const std::string &op_name, const AbstractBasePtrList &args_spec_list,
                                          size_t index) {
  auto tensor = mindspore::abstract::CheckArg<AbstractTensor>(op_name, args_spec_list, index);
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor->element()->BuildType();
}
}  // namespace

void SparseAddGrad::Init() {}

AbstractBasePtr SparseAddGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  for (auto &ptr : input_args) {
    MS_EXCEPTION_IF_NULL(ptr);
  }
  const size_t kInputNum = 4;
  constexpr size_t kIndicesShapeSize = 2;
  constexpr size_t kValuesShapeSize = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, SizeToLong(kInputNum), name);
  auto bvg = input_args.at(kSparseAddGradIndex0);
  auto x1_indices = input_args.at(kSparseAddGradIndex1);
  auto x2_indices = input_args.at(kSparseAddGradIndex2);
  auto sum_indices = input_args.at(kSparseAddGradIndex3);

  const std::set<TypePtr> indices_valid_types = {kInt64};
  const std::set<TypePtr> values_valid_types = {kInt8,    kInt16,   kInt32,     kInt64,
                                                kFloat32, kFloat64, kComplex64, kComplex128};
  CheckAndConvertUtils::CheckTensorTypeValid("backprop_val_grad", bvg->BuildType(), values_valid_types, name);
  CheckAndConvertUtils::CheckTensorTypeValid("x1_indices", x1_indices->BuildType(), indices_valid_types, name);
  CheckAndConvertUtils::CheckTensorTypeValid("x2_indices", x2_indices->BuildType(), indices_valid_types, name);
  CheckAndConvertUtils::CheckTensorTypeValid("sum_indices", sum_indices->BuildType(), indices_valid_types, name);
  std::shared_ptr<AbstractTensor> dx1 = nullptr;
  std::shared_ptr<AbstractTensor> dx2 = nullptr;
  auto val_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto val_grad_shape_dyn_rank = IsDynamicRank(val_grad_shape[kShape]);
  CheckSparseAddGradShape(val_grad_shape[kShape].size(), kValuesShapeSize, "backprop_val_grad", name,
                          val_grad_shape_dyn_rank);
  auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSparseAddGradIndex1]->BuildShape());
  auto dx1_shape = x1_shape[kShape];
  auto dx1_shape_dyn_rank = IsDynamicRank(dx1_shape);
  CheckSparseAddGradShape(dx1_shape.size(), kIndicesShapeSize, "x1_indices", name, dx1_shape_dyn_rank);
  auto type = SparseAddGradInferType(name, input_args, 0);
  ShapeVector shp = {dx1_shape.at(0)};
  dx1 = std::make_shared<AbstractTensor>(type, std::make_shared<mindspore::abstract::Shape>(shp));
  auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSparseAddGradIndex2]->BuildShape());
  ShapeVector dx2_shape = x2_shape[kShape];
  auto dx2_shape_dyn_rank = IsDynamicRank(dx2_shape);
  CheckSparseAddGradShape(dx2_shape.size(), kIndicesShapeSize, "x2_indices", name, dx2_shape_dyn_rank);
  shp = {dx2_shape.at(0)};
  dx2 = std::make_shared<AbstractTensor>(type, std::make_shared<mindspore::abstract::Shape>(shp));
  auto sum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kSparseAddGradIndex3]->BuildShape());
  auto sum_shape_dyn_rank = IsDynamicRank(sum_shape[kShape]);
  CheckSparseAddGradShape(sum_shape[kShape].size(), kIndicesShapeSize, "sum_indices", name, sum_shape_dyn_rank);
  if (!sum_shape_dyn_rank && !val_grad_shape_dyn_rank && sum_shape[kShape][0] >= 0 && val_grad_shape[kShape][0] >= 0) {
    CheckSparseAddGradNNZ(sum_shape[kShape][0], val_grad_shape[kShape][0], "sum_indices", "backprop_val_grad", name);
  }
  AbstractBasePtrList ret = {dx1, dx2};
  return std::make_shared<AbstractTuple>(ret);
}

MIND_API_OPERATOR_IMPL(SparseAddGrad, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseAddGrad, prim::kPrimSparseAddGrad, SparseAddGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
