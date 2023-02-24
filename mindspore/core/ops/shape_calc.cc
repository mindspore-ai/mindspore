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
#include "ops/shape_calc.h"
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
ShapeFunc ShapeCalc::get_shape_func() const {
  MS_EXCEPTION_IF_NULL(impl_);
  auto shape_func_attr = api::ToRef<mindspore::Primitive>(impl_).GetAttr(kAttrShapeFunc);
  if (shape_func_attr == nullptr) {
    return nullptr;
  }
  auto shape_func = shape_func_attr->cast<ShapeFunctionPtr>();
  MS_EXCEPTION_IF_NULL(shape_func);
  auto func = shape_func->impl();
  return func;
}

std::vector<int64_t> ShapeCalc::get_value_depend_indices() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kAttrValueDependIndices));
}

MIND_API_OPERATOR_IMPL(ShapeCalc, BaseOperator);
abstract::AbstractBasePtr ShapeCalcInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                         const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto value_depend_indices = GetValue<std::vector<int64_t>>(prim->GetAttr(kAttrValueDependIndices));
  std::unordered_set<int64_t> indices(value_depend_indices.begin(), value_depend_indices.end());
  ShapeArray args(input_args.size());
  std::unordered_set<size_t> invalid_indices;
  for (size_t i = 0; i < input_args.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_args[i]);
    auto value_ptr = input_args[i]->BuildValue();
    MS_EXCEPTION_IF_NULL(value_ptr);
    if (input_args[i]->isa<abstract::AbstractSequence>() || input_args[i]->isa<abstract::AbstractScalar>()) {
      if (IsValueKnown(value_ptr)) {
        args[i] = CheckAndConvertUtils::CheckIntOrTupleInt(std::to_string(i), value_ptr, prim_name);
      } else {
        invalid_indices.insert(i);
      }
    } else if (input_args[i]->isa<abstract::AbstractTensor>()) {
      if (indices.find(static_cast<int64_t>(i)) != indices.end()) {
        // value tensor
        if (value_ptr->isa<tensor::Tensor>()) {
          args[i] = CheckAndConvertUtils::CheckTensorIntValue(std::to_string(i), value_ptr, prim_name);
        } else {
          invalid_indices.insert(i);
        }
        continue;
      }
      // shape tensor
      auto input = input_args[i]->cast<abstract::AbstractTensorPtr>();
      MS_EXCEPTION_IF_NULL(input);
      auto shape_ptr = input->shape();
      MS_EXCEPTION_IF_NULL(shape_ptr);
      auto shape = shape_ptr->shape();
      // input is a tensor that saves the shape, and tensor itself should be 0D or 1D
      MS_EXCEPTION_IF_CHECK_FAIL(shape.size() <= 1, "Input tensor's rank must be <= 1");
      if (!shape.empty()) {
        args[i] = shape[0] < 0 ? ShapeVector{abstract::Shape::kShapeRankAny}
                               : ShapeVector(shape[0], abstract::Shape::kShapeDimAny);
      }
    } else {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input[" << i
                              << "] must be a Tensor, scalar integer, list integer or tuple integer, but got "
                              << input_args[i]->ToString();
    }
  }
  auto infer_func_attr = prim->GetAttr(kAttrInferFunc);
  MS_EXCEPTION_IF_NULL(infer_func_attr);
  auto infer_func = infer_func_attr->cast<InferFunctionPtr>();
  MS_EXCEPTION_IF_NULL(infer_func);
  auto func = infer_func->impl();
  MS_EXCEPTION_IF_NULL(func);
  auto out = func(args, invalid_indices);
  if (out.size() == 1) {
    // single output does not use AbstractTuple to avoid TupleGetItem
    return abstract::MakeAbstract(std::make_shared<abstract::Shape>(ShapeVector{out[0]}), kInt64);
  }
  // multiple outputs
  std::vector<TypePtr> types(out.size(), kInt64);
  std::vector<abstract::BaseShapePtr> shapes;
  (void)std::transform(out.begin(), out.end(), std::back_inserter(shapes),
                       [](int64_t s) { return std::make_shared<abstract::Shape>(ShapeVector{s}); });
  return abstract::MakeAbstract(std::make_shared<abstract::TupleShape>(shapes), std::make_shared<Tuple>(types));
}
REGISTER_PRIMITIVE_EVAL_IMPL(ShapeCalc, prim::kPrimShapeCalc, ShapeCalcInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
