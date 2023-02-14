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
#include <set>
#include <map>
#include <string>

#include "ops/count_nonzero.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> CheckAttrIntOrTuple(const ValuePtr &attr) {
  std::vector<int64_t> result{};
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>() || attr->isa<ValueList>()) {
    result = GetValue<std::vector<int64_t>>(attr);
  } else {
    auto attr_val = GetValue<int64_t>(attr);
    (void)result.insert(result.begin(), 1, attr_val);
  }
  return result;
}

abstract::ShapePtr CountNonZeroInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto output_shape = input_shape;
  auto input_rank = SizeToLong(input_shape.size());
  std::vector<int64_t> dims = CheckAttrIntOrTuple(primitive->GetAttr("dims"));

  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  if (dims.size() == 0) {
    output_shape = std::vector<int64_t>{};
    return std::make_shared<abstract::Shape>(output_shape);
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    int64_t origin_dims = dims[i];
    if (dims[i] < 0) {
      dims[i] += input_rank;
    }
    string dims_name = "dims[" + std::to_string(i) + "]";
    int64_t int_input_rank = static_cast<int64_t>(input_rank);
    if (input_rank == 0) {
      if (dims[i] != 0 && dims[i] != -1) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the dims[" << i << "] is out of range[-1, 0].";
      }
    } else if (int_input_rank > 0) {
      CheckAndConvertUtils::CheckInRange(dims_name, origin_dims, kIncludeLeft, {-int_input_rank, int_input_rank},
                                         "CountNonZero");
    }
  }
  if (input_rank == 0) {
    output_shape = std::vector<int64_t>{};
    primitive->EraseAttr("dims");
    primitive->set_attr("dims", MakeValue(std::vector<int64_t>{}));
    return std::make_shared<abstract::Shape>(output_shape);
  }

  for (size_t i = 0; i < dims.size(); ++i) {
    output_shape[dims[i]] = -1;
  }

  for (std::vector<int64_t>::iterator iter = output_shape.begin(); iter != output_shape.end(); ++iter) {
    if (*iter == -1) {
      iter = output_shape.erase(iter);
      iter -= 1;
    }
  }
  std::set<int64_t> dim_set(dims.begin(), dims.end());
  if (dim_set.size() != dims.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the dims contain duplicates.";
  } else {
    std::vector<int64_t> dims_processed(dim_set.begin(), dim_set.end());
    primitive->EraseAttr("dims");
    primitive->set_attr("dims", MakeValue(dims_processed));
  }

  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr CountNonZeroInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  TypePtr input_x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_x_type);
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, valid_types, prim->name());
  auto y_type = std::make_shared<TensorType>(kInt64);
  return y_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(CountNonZero, BaseOperator);
class MIND_API CountNonZeroInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CountNonZeroInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const int64_t kInputsNum = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
    return CountNonZeroInferType(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CountNonZero, prim::kPrimCountNonZero, CountNonZeroInfer, false);
}  // namespace ops
}  // namespace mindspore
