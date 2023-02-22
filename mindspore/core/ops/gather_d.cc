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
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
// gather_d
namespace {
bool GetGatherDimValue(const AbstractBasePtr dim_ptr, int64_t *dim_v) {
  MS_EXCEPTION_IF_NULL(dim_ptr);
  auto dim_value_ptr = dim_ptr->BuildValue();
  MS_EXCEPTION_IF_NULL(dim_value_ptr);
  auto dim_type_ptr = dim_ptr->BuildType();
  MS_EXCEPTION_IF_NULL(dim_type_ptr);
  bool dim_value_type_error = false;
  if (dim_value_ptr->isa<tensor::Tensor>()) {
    auto dim_tensor = dim_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(dim_tensor);
    size_t data_size = dim_tensor->DataSize();
    MS_EXCEPTION_IF_CHECK_FAIL(data_size == 1, "dim value is not equal to one!");
    auto dim_type_id = dim_type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(dim_type_id);
    auto element = dim_type_id->element();
    MS_EXCEPTION_IF_NULL(element);
    if (dim_tensor->data_c() == nullptr) {
      return false;
    }
    if (element->type_id() == kNumberTypeInt64) {
      auto dim_data64 = reinterpret_cast<int64_t *>(dim_tensor->data_c());
      MS_EXCEPTION_IF_NULL(dim_data64);
      *dim_v = static_cast<int64_t>(*dim_data64);
      return true;
    } else if (element->type_id() == kNumberTypeInt32) {
      auto dim_data32 = reinterpret_cast<int *>(dim_tensor->data_c());
      MS_EXCEPTION_IF_NULL(dim_data32);
      *dim_v = static_cast<int64_t>(*dim_data32);
      return true;
    } else {
      dim_value_type_error = true;
    }
  } else {
    if (dim_value_ptr->isa<Int32Imm>() || dim_value_ptr->isa<Int64Imm>()) {
      *dim_v = GetValue<int64_t>(dim_value_ptr);
      return true;
    } else {
      dim_value_type_error = true;
    }
  }

  if (dim_value_type_error) {
    MS_LOG(EXCEPTION) << "For GatherD, 'dim' must be one of these types: [int32/int64].";
  }
  return false;
}

void CheckGatherShapeEqual(const std::string &prim_name, const ShapeVector &x_shape, int64_t dim_v,
                           const ShapeVector &index_shape) {
  if (IsDynamic(x_shape) || IsDynamic(index_shape)) {
    return;
  }
  CheckAndConvertUtils::Check("x_rank", SizeToLong(x_shape.size()), kEqual, SizeToLong(index_shape.size()), prim_name);
  for (size_t i = 0; i < x_shape.size(); ++i) {
    if (SizeToLong(i) == dim_v) {
      continue;
    }
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
  MS_EXCEPTION_IF_CHECK_FAIL(input_args[kInputIndex2]->BuildShape()->isa<abstract::Shape>(), "index's shape wrong.");
  auto index_shape_element = input_args[kInputIndex2]->BuildShape()->cast<abstract::ShapePtr>();
  auto index_shape = index_shape_element->shape();
  bool is_dim_dynamic = input_args[kInputIndex1]->BuildValue()->isa<AnyValue>();
  if (IsDynamicRank(x_shape) || is_dim_dynamic) {
    return std::make_shared<abstract::Shape>(index_shape);
  }
  int64_t x_rank = SizeToLong(x_shape.size());
  int64_t dim_v = 0;
  if (!GetGatherDimValue(input_args[kInputIndex1], &dim_v)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  CheckAndConvertUtils::Check("dim value", dim_v, kGreaterEqual, -x_rank, prim_name);
  CheckAndConvertUtils::Check("dim value", dim_v, kLessThan, x_rank, prim_name);

  if (dim_v < 0) {
    dim_v = dim_v + x_rank;
  }

  // For Ascend, only support x.shape[d] == index.shape[d] when d != dim. So limit it.
  CheckGatherShapeEqual(prim_name, x_shape, dim_v, index_shape);
  return std::make_shared<abstract::Shape>(index_shape);
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

  // check
  auto prim_name = primitive->name();
  const int64_t inputs_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, inputs_num, prim_name);
  std::set<TypePtr> index_valid_types = {kInt32, kInt64};
  std::set<TypePtr> dim_valid_types = {kInt32, kInt64, std::make_shared<TensorType>(kInt32),
                                       std::make_shared<TensorType>(kInt64)};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("index", input_args[kInputIndex2]->BuildType(), index_valid_types,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckSubClass("dim", input_args[kInputIndex1]->BuildType(), dim_valid_types, prim_name);
  auto infer_type = GatherDInferType(primitive, input_args);
  auto infer_shape = GatherDInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGGatherDInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherDInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherDInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GatherDInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GatherD, prim::kPrimGatherD, AGGatherDInfer, false);
}  // namespace ops
}  // namespace mindspore
