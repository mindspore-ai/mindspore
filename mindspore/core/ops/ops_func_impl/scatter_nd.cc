/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/scatter_nd.h"
#include <set>
#include <utility>
#include <memory>
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr int64_t kScatterNdInputNum = 3LL;
constexpr size_t kScatterNdInputsNum = 3;
constexpr size_t kIndicesMinRank = 2;
namespace {
void ScatterNdCheckShape(const PrimitivePtr &prim, const BaseShapePtr &indices_shape_base,
                         const BaseShapePtr &updates_shape_base, const ShapeVector &out_shape) {
  const auto &indices_shape = indices_shape_base->GetShapeVector();
  const auto &updates_shape = updates_shape_base->GetShapeVector();
  if (IsDynamicRank(indices_shape)) return;
  MS_CHECK_VALUE(indices_shape.size() >= kIndicesMinRank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank(indices)", SizeToLong(indices_shape.size()),
                                                             kGreaterEqual, kIndicesMinRank, prim));

  if (IsDynamicRank(out_shape) || IsDynamicRank(updates_shape) || IsDynamicShape(indices_shape)) return;
  size_t n = LongToSize(indices_shape.back());
  if (n > out_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim->name()
                             << "', if the rank of output tensor is 'P' (length of the 'shape'), "
                                "and the last dimension of 'indices' is "
                             << "'N', the 'N' should be less than or equal to 'P', but got P = " << out_shape.size()
                             << ", N = " << n;
  }
  // the rank of updates is Q-1+P-N
  if (updates_shape.size() != indices_shape.size() - 1 + out_shape.size() - n) {
    MS_EXCEPTION(ValueError) << "For '" << prim->name()
                             << "', if the rank of 'indices' is 'Q', the rank of 'updates' is 'R', "
                             << "the rank of output tensor is 'P' (length of the 'shape'), and the "
                                "last dimension of 'indices' is 'N', "
                             << "then 'R' should be equal to 'Q - 1 + P - N'. but got 'R' = " << updates_shape.size()
                             << ", 'Q' = " << indices_shape.size() << ", 'P' = " << out_shape.size() << ", 'N' = " << n;
  }

  if (IsDynamicShape(out_shape) || IsDynamicShape(updates_shape)) return;
  // updates.shape = indices.shape[:-1] + shape[indices.shape[-1]:]
  bool constrain = true;
  for (size_t i = 0; i + 1 < indices_shape.size(); ++i) {
    if (updates_shape[i] != indices_shape[i]) {
      constrain = false;
      break;
    }
  }
  size_t si = n;
  size_t ui = indices_shape.size() - 1;
  for (; si < out_shape.size(); ++si, ++ui) {
    if (updates_shape[ui] != out_shape[si]) {
      constrain = false;
      break;
    }
  }
  if (!constrain) {
    std::ostringstream buffer;
    buffer << "For '" << prim->name()
           << "', if the last dimension of 'indices' is 'N', the shape of "
              "'updates' should be the concatenation of "
           << "'indices.shape[:-1]' and 'shape[N:]'. but got 'indices.shape' is (" << indices_shape_base->ToString()
           << ", 'updates.shape' is (" << updates_shape_base->ToString() << ", 'shape' is (";
    for (auto item : out_shape) {
      buffer << item << ", ";
    }
    buffer << ").";
    MS_EXCEPTION(ValueError) << buffer.str();
  }
}
}  // namespace

BaseShapePtr ScatterNdFuncImpl::InferShape(const PrimitivePtr &prim,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  ShapeVector output_shape{};
  auto indices_shape_base = input_args[kInputIndex0]->GetShape();
  auto updates_shape_base = input_args[kInputIndex1]->GetShape();
  auto shape_base = input_args[kInputIndex2]->GetShape();
  const auto &indices_shape = indices_shape_base->GetShapeVector();
  const auto &updates_shape = updates_shape_base->GetShapeVector();

  auto shape_value_ptr = input_args[kInputIndex2]->GetValue();
  MS_EXCEPTION_IF_NULL(shape_value_ptr);
  auto shape_tuple_opt = GetArrayValue<int64_t>(shape_value_ptr);
  const int64_t last_dim = indices_shape.back();
  bool last_dim_unknown = last_dim == abstract::Shape::kShapeDimAny || last_dim == abstract::Shape::kShapeRankAny;
  if (shape_base->isa<abstract::DynamicSequenceShape>() || !shape_tuple_opt.has_value()) {
    // correspond to the cases that the shape tuple has dynamic length and the case that the tuple has no value;
    output_shape.push_back(abstract::Shape::kShapeRankAny);
    if (last_dim_unknown || IsDynamicRank(updates_shape)) {
      return std::make_shared<abstract::TensorShape>(std::move(output_shape));
    }
  } else {
    const ArrayValue<int64_t> &shape_tuple = shape_tuple_opt.value();
    for (size_t i = 0; i < shape_tuple.size(); i++) {
      if (shape_tuple.IsValueUnknown(i)) {
        output_shape.emplace_back(abstract::Shape::kShapeDimAny);
      } else {
        if (!(shape_tuple[i] > 0)) {
          MS_EXCEPTION(ValueError) << "For 'ScatterNd', the input [shape] should be a tuple with all positive item. "
                                      "But got: "
                                   << shape_tuple.ToString();
        }
        output_shape.emplace_back(shape_tuple[i]);
      }
    }
  }

  ScatterNdCheckShape(prim, indices_shape_base, updates_shape_base, output_shape);

  const int64_t indices_size = SizeToLong(indices_shape.size());
  const int64_t updates_size = SizeToLong(updates_shape.size());
  // try to infer output shape with constraints: updates.shape = indices.shape[:-1] + shape[indices.shape[-1]:].
  if (IsDynamicRank(output_shape)) {
    output_shape.clear();
    output_shape.resize(last_dim + updates_size - indices_size + 1, abstract::Shape::kShapeDimAny);
  }
  if (!(IsDynamicRank(indices_shape) || IsDynamicRank(updates_shape)) && last_dim != abstract::Shape::kShapeDimAny) {
    for (size_t i = LongToSize(last_dim); i < output_shape.size(); i++) {
      auto updates_dim = updates_shape[indices_shape.size() - 1 + i - last_dim];
      if (output_shape[i] == abstract::Shape::kShapeDimAny && updates_dim != abstract::Shape::kShapeDimAny) {
        output_shape[i] = updates_dim;
      }
    }
  }
  return std::make_shared<abstract::TensorShape>(std::move(output_shape));
}

TypePtr ScatterNdFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype = input_args[kInputIndex1]->GetType();
  return dtype->Clone();
}
}  // namespace ops
}  // namespace mindspore
