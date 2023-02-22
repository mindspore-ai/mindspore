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
#include "ops/search_sorted.h"

#include <set>
#include <algorithm>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
constexpr auto kNameRight = "right";
void SearchSorted::set_right(const bool right) { (void)AddAttr(kNameRight, api::MakeValue(right)); }
bool SearchSorted::get_right() const { return GetValue<bool>(GetAttr(kNameRight)); }
namespace {
abstract::ShapePtr SearchSortedInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto sequence_shape_ptr = input_args[kInputIndex0]->BuildShape();
  ShapeVector sequence_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(sequence_shape_ptr)[kShape];
  auto values_shape_ptr = input_args[kInputIndex1]->BuildShape();
  ShapeVector values_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(values_shape_ptr)[kShape];
  ShapeVector sequence_shape_c = sequence_shape;
  ShapeVector values_shape_c = values_shape;
  if (sequence_shape_c.empty() || values_shape_c.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', empty tensor is not permitted in 'sorted_sequence' inputs. "
                             << "The shape of 'sorted_sequence': " << sequence_shape_ptr->ToString()
                             << " and shape of 'values': " << values_shape_ptr->ToString() << ".";
  }
  sequence_shape_c.pop_back();
  values_shape_c.pop_back();
  if (!(IsDynamic(sequence_shape) || IsDynamic(values_shape)) && sequence_shape.size() != 1 &&
      sequence_shape_c != values_shape_c) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the 'sorted_sequence' must be 1 dimensional or "
                                "all dimensions except the last dimension of 'sorted_sequence' "
                             << "must be the same as all dimensions except the last dimension of 'values', "
                             << "but got shape of 'sorted_sequence': " << sequence_shape_ptr->ToString()
                             << " and shape of 'values': " << values_shape_ptr->ToString() << ".";
  }
  bool is_empty_shape =
    std::any_of(sequence_shape.begin(), sequence_shape.end(), [](const auto &element) { return element == 0; });
  if (is_empty_shape) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the shape of input sequence_shape contains 0, is not permitted. "
                             << "The shape is: " << sequence_shape;
  }
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
  MS_EXCEPTION_IF_NULL(values_shape_ptr);
  auto shape_element = values_shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SearchSortedInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto dtype = primitive->GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype);
  auto infer_type = dtype->cast<TypePtr>();
  auto sequence_type = input_args[kInputIndex0]->BuildType();
  auto values_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("sorted_sequence", sequence_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("values", values_type, valid_types, prim_name);
  return infer_type;
}
}  // namespace

// 属性相关
void SearchSorted::Init(const bool right, const bool out_int32) { set_right(right); }

MIND_API_OPERATOR_IMPL(SearchSorted, BaseOperator);
AbstractBasePtr SearchSortedInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kInputNum = 2;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
  auto infer_type = SearchSortedInferType(primitive, input_args);
  auto infer_shape = SearchSortedInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGSearchSortedInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SearchSortedInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SearchSortedInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SearchSortedInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SearchSorted, prim::kPrimSearchSorted, AGSearchSortedInfer, false);
}  // namespace ops
}  // namespace mindspore
