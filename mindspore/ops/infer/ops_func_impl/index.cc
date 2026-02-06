/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <functional>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <algorithm>
#include "infer/ops_func_impl/index.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"

namespace mindspore {
namespace ops {
constexpr size_t kIndexEmptyShape = 9;

std::vector<TypeId> IndexFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}

ShapeArray IndexFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &x_tensor = input_infos[kIndex0];
  auto op_name = primitive->name();
  if (x_tensor->IsDynamicRank()) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  auto x_shape = x_tensor->GetShape();
  auto &indices = input_infos[kIndex1];
  ShapeVector output_shape = {};
  if (indices->IsSequence()) {
    if (indices->IsDynamicSequence()) {
      MS_EXCEPTION(ValueError) << "For `" << op_name << "` op, 'indices' shape can not DynamicSequenceShape.";
    } else {
      auto elements = indices->GetSequenceElements();
      ShapeArray shapes;
      std::transform(elements.begin(), elements.end(), std::back_inserter(shapes), [](const InferInfoPtr &info) {
        return (info->GetType() == kNumberTypeBool || info->GetType() == kNumberTypeUInt8)
                 ? ShapeVector({abstract::TensorShape::kShapeRankAny})
                 : info->GetShape();
      });
      output_shape = CheckAndCalOutputShapeInTupleCase(x_shape, shapes);
    }
  } else {
    ShapeArray shapes;
    std::transform(input_infos.begin() + kIndex1, input_infos.end(), std::back_inserter(shapes),
                   [](const InferInfoPtr &info) {
                     return (info->GetType() == kNumberTypeBool || info->GetType() == kNumberTypeUInt8)
                              ? ShapeVector({abstract::TensorShape::kShapeRankAny})
                              : info->GetShape();
                   });
    output_shape = CheckAndCalOutputShapeInTupleCase(x_shape, shapes);
  }
  return {output_shape};
}

ShapeVector IndexFuncImpl::CheckAndCalOutputShapeInTupleCase(const ShapeVector &x_shape,
                                                             const ShapeArray &indices_shapes) const {
  // 1. Get the expanded index shape.
  if (x_shape.size() < indices_shapes.size()) {
    MS_EXCEPTION(ValueError) << "For 'Index', too many indices for tensor of dimension " << x_shape.size() << " (got "
                             << indices_shapes.size() << ")";
  }
  auto expand_index_shapes = ExpandIndexShape(indices_shapes);
  if (expand_index_shapes.size() == 1 && IsDynamicRank(expand_index_shapes[0])) {
    return ShapeVector({abstract::TensorShape::kShapeRankAny});
  }
  while (expand_index_shapes.size() < x_shape.size()) {
    ShapeVector empty_shape = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    expand_index_shapes.emplace_back(empty_shape);
  }

  // 2. If the non-empty tensor in the index is not contiguous, move it to the front to make it contiguous.
  ShapeVector fin_x_shape = x_shape;
  ShapeArray fin_index_shapes = expand_index_shapes;
  if (!IndexContiguous(expand_index_shapes)) {
    fin_x_shape.clear();
    fin_index_shapes.clear();
    std::tie(fin_x_shape, fin_index_shapes) = TransposeToFront(x_shape, expand_index_shapes);
  }

  // 3. Get the position and shape information of non-empty tensors in an index.
  int64_t dims_before = 0;
  int64_t dims_after = 0;
  int64_t dims_indexed = 0;
  ShapeVector replacement_shape;
  ShapeVector indexed_sizes;
  for (size_t dim = 0; dim < fin_index_shapes.size(); dim++) {
    auto idx_shape = fin_index_shapes[dim];
    if (idx_shape.size() == kIndexEmptyShape &&
        std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; })) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = idx_shape;
      indexed_sizes.push_back(fin_x_shape[dim]);
    }
  }

  // 4. If the input tensor has shape 0 but the index tensor does not, report error.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    MS_EXCEPTION(ValueError) << "For 'Index', if the input tensor of dimension with size 0"
                             << ", the index tensor should same"
                             << ", but index is out of bounds for dimension with size 0";
  }

  // 5. Replaces the indexed part in the shape of the input tensor with the shape of the index.
  ShapeVector out_shape(fin_x_shape);
  int64_t end = dims_before + dims_indexed;
  out_shape.erase(out_shape.begin() + dims_before, out_shape.begin() + end);
  out_shape.insert(out_shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
  return out_shape;
}

std::tuple<ShapeVector, ShapeArray> IndexFuncImpl::TransposeToFront(const ShapeVector &x_shape,
                                                                    const ShapeArray &index_shapes) const {
  ShapeVector fin_x_shape;
  ShapeArray fin_index_shapes;
  for (size_t i = 0; i < index_shapes.size(); ++i) {
    auto idx_shape = index_shapes[i];
    auto x_size = x_shape[i];
    if (!(idx_shape.size() == kIndexEmptyShape &&
          std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; }))) {
      fin_index_shapes.push_back(idx_shape);
      fin_x_shape.push_back(x_size);
    }
  }

  for (size_t i = 0; i < index_shapes.size(); ++i) {
    auto idx_shape = index_shapes[i];
    auto x_size = x_shape[i];
    if (idx_shape.size() == kIndexEmptyShape &&
        std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; })) {
      fin_index_shapes.push_back(idx_shape);
      fin_x_shape.push_back(x_size);
    }
  }
  return std::make_tuple(std::move(fin_x_shape), std::move(fin_index_shapes));
}

bool IndexFuncImpl::IndexContiguous(const ShapeArray &index_shape) const {
  auto isEmpty = [](const ShapeVector &idx_shape) {
    return idx_shape.size() == kIndexEmptyShape &&
           std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; });
  };
  auto isNoEmpty = [](const ShapeVector &idx_shape) {
    return !(idx_shape.size() == kIndexEmptyShape &&
             std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; }));
  };
  auto start = std::find_if(index_shape.begin(), index_shape.end(), isNoEmpty);
  auto stop = std::find_if(index_shape.rbegin(), index_shape.rend(), isNoEmpty);
  auto it = std::find_if(start, stop.base(), isEmpty);
  return it == stop.base();
}

ShapeArray IndexFuncImpl::ExpandIndexShape(const ShapeArray &to_expand) const {
  // expands a list of Tensors; ignores empty tensors
  bool first = true;
  ShapeVector tmp_shape;
  ShapeArray expanded_shapes(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); ++i) {
    const auto &elem_shape = to_expand[i];
    if (elem_shape.size() == kIndexEmptyShape &&
        std::all_of(elem_shape.begin(), elem_shape.end(), [](int i) { return i == 0; })) {
      expanded_shapes[i] = elem_shape;
      continue;
    }
    if (first) {
      tmp_shape = elem_shape;
      first = false;
    } else {
      tmp_shape = CalBroadCastShape(tmp_shape, elem_shape, "Index", "indices_x", "indices_y");
    }
    if (IsDynamicRank(tmp_shape)) {
      return {{abstract::TensorShape::kShapeRankAny}};
    }
    expanded_shapes[i] = tmp_shape;
  }
  return expanded_shapes;
}

REGISTER_SIMPLE_INFER(kNameIndex, IndexFuncImpl)
}  // namespace ops
}  // namespace mindspore
