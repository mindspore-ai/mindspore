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

#include "ops/sparse_add.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
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
constexpr auto kADenseShape = "a_shape";
constexpr auto kBDenseShape = "b_shape";
constexpr auto kThresh = "thresh";

inline void CheckSparseAddShape(const size_t sparse_shape_size, const size_t expected_dim,
                                const std::string &arg_name) {
  if (sparse_shape_size != expected_dim) {
    MS_EXCEPTION(mindspore::ValueError) << arg_name << " must be a " << expected_dim
                                        << "-dimensional tensor, but got a " << sparse_shape_size
                                        << "-dimensional tensor.";
  }
}

inline void CheckSparseAddIndicesDtype(const mindspore::TypePtr dtype, const std::string &arg_name) {
  if (!(dtype->equal(mindspore::kInt32))) {
    MS_EXCEPTION(mindspore::TypeError) << "The dtype of " << arg_name << " must be Int16 or Int32 or Int64, but got "
                                       << dtype->ToString() << ".";
  }
}
}  // namespace

void SparseAdd::set_a_dense_shape(const std::vector<int64_t> &shape) {
  (void)this->AddAttr(kADenseShape, api::MakeValue(shape));
}
void SparseAdd::set_b_dense_shape(const std::vector<int64_t> &shape) {
  (void)this->AddAttr(kBDenseShape, api::MakeValue(shape));
}
void SparseAdd::set_thresh(const float &thresh) { (void)this->AddAttr(kThresh, api::MakeValue(thresh)); }
std::vector<int64_t> SparseAdd::get_a_dense_shape() const {
  auto value_ptr = GetAttr(kADenseShape);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<int64_t> SparseAdd::get_b_dense_shape() const {
  auto value_ptr = GetAttr(kBDenseShape);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
float SparseAdd::get_thresh() const {
  auto value_ptr = GetAttr(kThresh);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

void SparseAdd::Init(const std::vector<int64_t> &a_shape, const std::vector<int64_t> &b_shape, const float &thresh) {
  auto op_name = this->name();
  if (a_shape.size() != b_shape.size()) {
    MS_LOG(EXCEPTION) << "For " << op_name
                      << " the rank of two dense shape should be the same, but got the rank of a_shape is "
                      << a_shape.size() << ", and b_shape is " << b_shape.size();
  }
  if (a_shape != b_shape) {
    MS_LOG(EXCEPTION) << "For " << op_name << " two dense shape should be the same, but got a_shape is " << a_shape
                      << ", and b_shape is " << b_shape;
  }
  this->set_a_dense_shape(a_shape);
  this->set_b_dense_shape(b_shape);
  this->set_thresh(thresh);
}

AbstractBasePtr SparseAddInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto a_shape = GetValue<std::vector<int64_t>>(primitive->GetAttr(kADenseShape));
  const std::string op_name = primitive->name();

  constexpr size_t kAIndicesIdx = 0;
  constexpr size_t kAValuesIdx = 1;
  constexpr size_t kBIndicesIdx = 2;
  constexpr size_t kBValuesIdx = 3;
  constexpr size_t kNumOfInputs = 4;
  constexpr size_t kIndicesShape = 2;

  mindspore::abstract::CheckArgsSize(op_name, input_args, kNumOfInputs);
  auto a_indices = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAIndicesIdx);
  auto a_values = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kAValuesIdx);
  auto b_indices = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kBIndicesIdx);
  auto b_values = mindspore::abstract::CheckArg<AbstractTensor>(op_name, input_args, kBValuesIdx);

  MS_EXCEPTION_IF_NULL(a_indices);
  MS_EXCEPTION_IF_NULL(a_values);
  MS_EXCEPTION_IF_NULL(b_indices);
  MS_EXCEPTION_IF_NULL(b_values);

  // 2-D indices
  auto a_indices_shape = a_indices->shape()->shape();
  auto b_indices_shape = b_indices->shape()->shape();
  CheckSparseAddShape(a_indices_shape.size(), kIndicesShape, "a_indices");
  CheckSparseAddShape(b_indices_shape.size(), kIndicesShape, "b_indices");
  // 1-D values
  auto a_values_shape = a_values->shape()->shape();
  auto b_values_shape = b_values->shape()->shape();
  CheckSparseAddShape(a_values_shape.size(), 1, "a_values");
  CheckSparseAddShape(b_values_shape.size(), 1, "b_values");

  auto a_type = a_values->element()->BuildType();
  auto b_type = b_values->element()->BuildType();
  // Input a_value and b_value should be the same data type
  if (a_type->type_id() != b_type->type_id()) {
    MS_LOG(EXCEPTION) << "For " << op_name
                      << ", the two input value should be the same data type, but got type of a_value is "
                      << a_type->type_id() << ", and type of b_value is " << b_type->type_id();
  }
  // a_indices and b_indices should be int16, int32 or int64
  CheckSparseAddIndicesDtype(a_indices->element()->BuildType(), op_name);
  CheckSparseAddIndicesDtype(b_indices->element()->BuildType(), op_name);

  int64_t max_indices_shape_ = std::min(a_indices_shape[0] + b_indices_shape[0], a_shape[0] * a_shape[1]);
  int64_t min_indices_shape_ = std::max(a_indices_shape[0], b_indices_shape[0]);
  ShapeVector out_indices_shape{-1, 2};
  ShapeVector out_value_shape{-1};
  ShapeVector min_value_shape{min_indices_shape_};
  ShapeVector max_value_shape{max_indices_shape_};
  ShapeVector min_indices_shape{min_indices_shape_, 2};
  ShapeVector max_indices_shape{max_indices_shape_, 2};

  auto out_indices = std::make_shared<AbstractTensor>(
    a_indices->element()->BuildType(),
    std::make_shared<mindspore::abstract::Shape>(out_indices_shape, min_indices_shape, max_indices_shape));
  auto out_values = std::make_shared<AbstractTensor>(
    a_type, std::make_shared<mindspore::abstract::Shape>(out_value_shape, min_value_shape, max_value_shape));

  AbstractBasePtrList ret = {out_indices, out_values};
  return std::make_shared<AbstractTuple>(ret);
}
MIND_API_OPERATOR_IMPL(SparseAdd, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseAdd, prim::kPrimSparseAdd, SparseAddInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
