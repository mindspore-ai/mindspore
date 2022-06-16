/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ops/scatter_nd.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
constexpr int64_t kScatterNdInputNum = 2LL;
void ScatterNdCheckShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto indices_shape_ptr = input_args[kInputIndex0]->BuildShape();
  ShapeVector indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  const int64_t kIndicesRank = 2LL;
  (void)CheckAndConvertUtils::CheckInteger("rank(indices)", SizeToLong(indices_shape.size()), kGreaterEqual,
                                           kIndicesRank, prim->name());

  auto updates_shape_ptr = input_args[kInputIndex1]->BuildShape();
  ShapeVector updates_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(updates_shape_ptr)[kShape];
  if (updates_shape.size() + 1 < indices_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', when the rank of 'indices' is 'N', "
                             << "the rank of 'updates' should be at lease 'N - 1', but got rank('indices') = "
                             << indices_shape.size() << " and rank('updates') = " << updates_shape.size() << ".";
  }
  for (size_t i = 0; i + 1 < indices_shape.size(); ++i) {
    if (updates_shape[i] != indices_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', when the rank of 'indices' is 'N', "
                               << "the 'updates.shape[0: N-1]' should be equal to 'indices.shape[0: N-1]', "
                               << "but got the shape of 'indices' is " << indices_shape_ptr->ToString()
                               << ", and the shape of 'updates' is " << updates_shape_ptr->ToString();
    }
  }
}

TypePtr ScatterNdInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto dtype = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("updates", dtype, common_valid_types, prim->name());
  return dtype;
}

abstract::BaseShapePtr ScatterNdInferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  ScatterNdCheckShape(prim, input_args);
  ShapeVector shape;
  if (input_args.size() > static_cast<size_t>(kScatterNdInputNum)) {
    auto shape_value = input_args[kInputIndex2]->BuildValue();
    if (shape_value->isa<tensor::Tensor>()) {
      shape = CheckAndConvertUtils::CheckTensorIntValue("shape", shape_value, prim->name());
    } else if (shape_value->isa<ValueTuple>()) {
      shape = CheckAndConvertUtils::CheckTupleInt("input[shape]", shape_value, prim->name());
    } else {
      return input_args[kInputIndex2]->BuildShape();
    }
  } else {
    auto shape_attr = prim->GetAttr("shape");
    MS_EXCEPTION_IF_NULL(shape_attr);
    shape = GetValue<ShapeVector>(shape_attr);
  }
  if (std::any_of(shape.begin(), shape.end(), [](int64_t item) { return item < 1; })) {
    std::ostringstream buffer;
    buffer << "For primitive[ScatterNd], the attribute[shape] should be a tuple with all positive item. but got (";
    for (auto item : shape) {
      buffer << item << ", ";
    }
    buffer << ").";
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  return std::make_shared<abstract::Shape>(shape);
}

AbstractBasePtr ScatterNdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const std::set<TypePtr> valid_indices_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kScatterNdInputNum, name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", input_args[0]->BuildType(), valid_indices_types, name);
  if (input_args.size() > static_cast<size_t>(kScatterNdInputNum)) {
    auto shape_type = input_args[kInputIndex2]->BuildType();
    if (!shape_type->isa<TensorType>()) {
      (void)CheckAndConvertUtils::CheckTypeValid("shape", shape_type, {kTuple}, name);
    }
  }

  auto infer_type = ScatterNdInferType(primitive, input_args);
  auto infer_shape = ScatterNdInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ScatterNd, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ScatterNd, prim::kPrimScatterNd, ScatterNdInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
