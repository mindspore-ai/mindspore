/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/gather_d.h"
#include <memory>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
// gather_d
namespace {
int64_t GetGatherDimValue(const AbstractBasePtr dim_ptr) {
  MS_EXCEPTION_IF_NULL(dim_ptr);
  auto dim_value_ptr = dim_ptr->BuildValue();
  MS_EXCEPTION_IF_NULL(dim_value_ptr);
  auto dim_type_ptr = dim_ptr->BuildType();
  MS_EXCEPTION_IF_NULL(dim_type_ptr);
  int64_t dim_v = 0;
  if (dim_value_ptr->isa<tensor::Tensor>()) {
    auto dim_tensor = dim_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(dim_tensor);
    size_t data_size = dim_tensor->DataSize();
    MS_EXCEPTION_IF_CHECK_FAIL(data_size == 1, "dim value is not equal to one!");
    auto dim_type_id = dim_type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(dim_type_id);
    auto element = dim_type_id->element();
    MS_EXCEPTION_IF_NULL(element);
    if (element->type_id() == kNumberTypeInt32) {
      auto dim_data32 = reinterpret_cast<int *>(dim_tensor->data_c());
      MS_EXCEPTION_IF_NULL(dim_data32);
      dim_v = static_cast<int64_t>(*dim_data32);
    } else {
      auto dim_data64 = reinterpret_cast<int64_t *>(dim_tensor->data_c());
      MS_EXCEPTION_IF_NULL(dim_data64);
      dim_v = static_cast<int64_t>(*dim_data64);
    }
  } else {
    if (dim_value_ptr->isa<Int32Imm>() || dim_value_ptr->isa<Int64Imm>()) {
      dim_v = GetValue<int64_t>(dim_value_ptr);
    } else {
      MS_LOG(EXCEPTION) << "For GatherD, 'dim' must be one of these types: [int32/int64].";
    }
  }

  return dim_v;
}

bool IsShapeInValid(const ShapeVector &shape) {
  return std::any_of(shape.cbegin(), shape.cend(), [](int64_t s) { return s < 0; });
}

void CheckGatherShapeEqual(const std::string &prim_name, const ShapeVector &x_shape, int64_t dim_v,
                           const ShapeVector &index_shape) {
  if (IsShapeInValid(x_shape) || IsShapeInValid(index_shape)) {
    return;
  }

  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (SizeToLong(i) == dim_v) continue;
    MS_LOG(INFO) << "For '" << prim_name << "', it's now checking " << i << "th x shape.";
    CheckAndConvertUtils::Check("x shape", x_shape[i], kEqual, index_shape[i], prim_name);
  }
}

abstract::ShapePtr GatherDInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  const size_t gather_d_input_num = 3;
  MS_EXCEPTION_IF_CHECK_FAIL(input_args.size() == gather_d_input_num,
                             "GatherD's input size should be 3 but got " + std::to_string(input_args.size()));

  MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex0]->BuildShape()->isa<abstract::Shape>(), "x's shape wrong.");
  auto shape_element = input_args[kInputIndex0]->BuildShape()->cast<abstract::ShapePtr>();
  auto x_shape = shape_element->shape();
  auto x_min_shape = shape_element->min_shape();
  auto x_max_shape = shape_element->max_shape();
  MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex2]->BuildShape()->isa<abstract::Shape>(), "index's shape wrong.");
  auto index_shape_element = input_args[kInputIndex2]->BuildShape()->cast<abstract::ShapePtr>();
  auto index_shape = index_shape_element->shape();
  auto index_min_shape = index_shape_element->min_shape();
  auto index_max_shape = index_shape_element->max_shape();
  int64_t x_rank = SizeToLong(x_shape.size());
  CheckAndConvertUtils::Check("x_rank", x_rank, kEqual, SizeToLong(index_shape.size()), prim_name);
  auto dim_v = GetGatherDimValue(input_args[kInputIndex1]);
  CheckAndConvertUtils::Check("dim value", dim_v, kGreaterEqual, -x_rank, prim_name);
  CheckAndConvertUtils::Check("dim value", dim_v, kLessThan, x_rank, prim_name);

  if (dim_v < 0) {
    dim_v = dim_v + x_rank;
  }

  // For Ascend, only support x.shape[d] == index.shape[d] when d != dim. So limit it.
  CheckGatherShapeEqual(prim_name, x_shape, dim_v, index_shape);
  CheckGatherShapeEqual(prim_name, x_min_shape, dim_v, index_min_shape);
  CheckGatherShapeEqual(prim_name, x_max_shape, dim_v, index_max_shape);
  return std::make_shared<abstract::Shape>(index_shape, index_min_shape, index_max_shape);
}

TypePtr GatherDInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  std::set<TypePtr> valid_x_type = {kTensorType};
  auto x_type = CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_x_type, prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(GatherD, BaseOperator);
AbstractBasePtr GatherDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  // check
  std::set<TypePtr> index_valid_types = {kInt32, kInt64};
  std::set<TypePtr> dim_valid_types = {kInt32, kInt64, std::make_shared<TensorType>(kInt32),
                                       std::make_shared<TensorType>(kInt64)};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("index", input_args[kInputIndex2]->BuildType(), index_valid_types,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckSubClass("dim", input_args[kInputIndex1]->BuildType(), dim_valid_types, prim_name);
  return abstract::MakeAbstract(GatherDInferShape(primitive, input_args), GatherDInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(GatherD, prim::kPrimGatherD, GatherDInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
