/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <set>

#include "ops/split.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr SplitInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto shape = input_args[0]->BuildShape();
  auto x_shape_ptr = shape->cast<abstract::ShapePtr>();
  auto x_shape = shape_map[kShape];

  int64_t output_num_value = GetValue<int64_t>(primitive->GetAttr("output_num"));
  std::vector<abstract::BaseShapePtr> output_list;
  if (IsDynamicRank(x_shape)) {
    for (int64_t i = 0; i < output_num_value; ++i) {
      abstract::ShapePtr output =
        std::make_shared<abstract::Shape>(std::vector<int64_t>(1, abstract::Shape::kShapeRankAny));
      output_list.push_back(output);
    }
    return std::make_shared<abstract::TupleShape>(output_list);
  }

  auto rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("rank", rank, kGreaterEqual, 1, prim_name);
  auto axis = GetValue<int64_t>(primitive->GetAttr("axis"));
  if (axis < 0) {
    axis += rank;
  }

  CheckAndConvertUtils::CheckInRange("axis", axis, kIncludeLeft, {-rank, rank}, prim_name);
  size_t pos = LongToSize(axis);
  if ((!x_shape_ptr->IsDynamic()) && (x_shape[pos] % output_num_value != 0)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', x_shape[" << pos
                             << "] must be divisible by output_num = " << output_num_value << ", but got "
                             << x_shape[pos];
  }
  std::vector<int64_t> size_splits;
  for (int64_t i = 0; i < output_num_value; ++i) {
    size_splits.push_back(x_shape[pos] / output_num_value);
  }
  (void)primitive->AddAttr("size_splits", MakeValue(size_splits));
  auto output_shape = x_shape;
  if (!x_shape_ptr->IsDynamic() || output_shape[pos] > 0) {
    output_shape[pos] = x_shape[pos] / output_num_value;
  }

  for (int64_t i = 0; i < output_num_value; ++i) {
    abstract::ShapePtr output = std::make_shared<abstract::Shape>(output_shape);
    output_list.push_back(output);
  }
  return std::make_shared<abstract::TupleShape>(output_list);
}

TuplePtr SplitInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto output_num = GetValue<int64_t>(prim->GetAttr("output_num"));
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, valid_types, prim->name());
  std::vector<TypePtr> type_tuple;
  for (int64_t i = 0; i < output_num; i++) {
    type_tuple.push_back(type);
  }
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(Split, BaseOperator);
void Split::Init(const int64_t axis, const int64_t output_num) {
  this->set_axis(axis);
  this->set_output_num(output_num);
}

void Split::set_size_splits(const std::vector<int64_t> &size_splits) {
  (void)this->AddAttr(kSizeSplits, api::MakeValue(size_splits));
}
void Split::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }
void Split::set_output_num(const int64_t output_num) { (void)this->AddAttr(kOutputNum, api::MakeValue(output_num)); }

std::vector<int64_t> Split::get_size_splits() const {
  auto value_ptr = GetAttr(kSizeSplits);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Split::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

int64_t Split::get_output_num() const {
  auto value_ptr = GetAttr(kOutputNum);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr SplitInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infertype = SplitInferType(primitive, input_args);
  auto infershape = SplitInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

// AG means auto generated
class MIND_API AGSplitInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SplitInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Split, prim::kPrimSplit, AGSplitInfer, false);
}  // namespace ops
}  // namespace mindspore
