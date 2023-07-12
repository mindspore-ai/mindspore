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

#include "ops/slice_to_indices.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ops/normalize_slice.h"
#include "ops/normalize_dim_index.h"

namespace mindspore {
namespace ops {
static abstract::AbstractTuplePtr VectorToAbsTuple(const std::vector<int64_t> &nums) {
  abstract::AbstractBasePtrList elems;
  std::transform(nums.begin(), nums.end(), std::back_inserter(elems),
                 [](int64_t num) { return std::make_shared<abstract::AbstractScalar>(num); });
  return std::make_shared<abstract::AbstractTuple>(elems);
}
AbstractBasePtr ConstSliceToIndices(const std::vector<int64_t> &init_by_none, const ShapeVector &data_shape,
                                    const AbstractBasePtr &start_abs, const AbstractBasePtr &stop_abs,
                                    const AbstractBasePtr &step_abs, size_t dim_index,
                                    const std::vector<int64_t> &tuple_index_types, size_t expand_dims_mask) {
  auto new_dim_index =
    ops::NormalizeDimIndex::ConstNormalizeDimIndex(data_shape.size(), dim_index, tuple_index_types, expand_dims_mask);
  if (new_dim_index >= data_shape.size()) {
    MS_EXCEPTION(IndexError) << "Index size out of data dims.";
  }
  std::shared_ptr<IndexSlice> slice_ptr = std::make_shared<IndexSlice>(
    GetValue<int64_t>(start_abs->BuildValue()), GetValue<int64_t>(stop_abs->BuildValue()),
    GetValue<int64_t>(step_abs->BuildValue()), data_shape[new_dim_index], init_by_none, true);

  if (slice_ptr->is_empty_slice()) {
    int64_t empty_stub_data = 0;
    auto indices_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector{0}, &empty_stub_data, 0);
    abstract::AbstractBasePtrList elems({6, std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(1))});
    elems[0] = indices_tensor->ToAbstract();
    return std::make_shared<abstract::AbstractTuple>(elems);
  }
  int64_t start = slice_ptr->start();
  int64_t stop = slice_ptr->stop();
  int64_t step = slice_ptr->step();
  std::vector<int64_t> indices;
  if (step > 0) {
    for (int64_t i = start; i < stop; i += step) {
      (void)indices.emplace_back(i);
    }
  } else {
    for (int64_t i = start; i > stop; i += step) {
      (void)indices.emplace_back(i);
    }
  }

  ShapeVector indices_shp({static_cast<int64_t>(indices.size()), 1});
  if (!tuple_index_types.empty()) {
    indices_shp = {static_cast<int64_t>(indices.size())};
  }
  auto shp_buf_size = sizeof(int64_t) * indices.size();
  auto indices_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, indices_shp, indices.data(), shp_buf_size);

  auto value_shape = data_shape;
  value_shape[0] = SizeToLong(indices.size());
  abstract::AbstractBasePtrList elems(
    {indices_tensor->ToAbstract(), VectorToAbsTuple(value_shape), std::make_shared<abstract::AbstractScalar>(start),
     std::make_shared<abstract::AbstractScalar>(stop), std::make_shared<abstract::AbstractScalar>(step),
     std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(0))});
  return std::make_shared<abstract::AbstractTuple>(elems);
}

AbstractBasePtr SliceToIndicesInferInner(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 4;
  CheckArgsSize(op_name, input_args, inputs_size);

  ShapeVector data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (!IsDynamic(data_shape) && std::all_of(input_args.begin() + 1, input_args.end(), [](const AbstractBasePtr &abs) {
        return IsValueKnown(abs->BuildValue());
      })) {
    size_t dim_index = LongToSize(GetValue<int64_t>(primitive->GetAttr(kAttrTupleIndexAxis)));
    auto tuple_index_types = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrTupleIndexTypes));
    size_t expand_dims_mask = LongToSize(GetValue<int64_t>(primitive->GetAttr(kAttrExpandDimsMask)));
    auto init_by_none = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrInitByNone));
    return ConstSliceToIndices(init_by_none, data_shape, input_args[kIndex1], input_args[kIndex2], input_args[kIndex3],
                               dim_index, tuple_index_types, expand_dims_mask);
  }

  auto abs_any = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  auto empty_abs_any = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  auto abs_tensor =
    std::make_shared<abstract::AbstractTensor>(abs_any, std::make_shared<abstract::Shape>(ShapeVector{1}));
  auto empty_abs_tensor =
    std::make_shared<abstract::AbstractTensor>(empty_abs_any, std::make_shared<abstract::Shape>());

  auto indices_tensor_abs = abstract::MakeAbstractTensor(
    std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny, 1}), kInt64);

  if (!data_shape.empty() && data_shape[0] != abstract::Shape::kShapeDimAny) {
    indices_tensor_abs = abstract::MakeAbstractTensor(
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny, 1}, ShapeVector{data_shape[0], 1}),
      kInt64);
  }
  auto value_shape_tensor_abs = abstract::MakeAbstractTensor(
    std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny}, ShapeVector{8}), kInt64);
  auto output = std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{
    indices_tensor_abs, value_shape_tensor_abs, abs_tensor, abs_tensor, abs_tensor, empty_abs_tensor});
  return output;
}

MIND_API_OPERATOR_IMPL(SliceToIndices, BaseOperator);

class SliceToIndicesInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    ShapeVector data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
    auto indices_tensor_shape =
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny, 1}, ShapeVector{data_shape[0], 1});
    const size_t max_dims = 8;
    auto value_shape =
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny}, ShapeVector{max_dims});
    auto slice_tensor_shape = std::make_shared<abstract::Shape>(ShapeVector{1});
    auto empty_tensor_shape = std::make_shared<abstract::Shape>(ShapeVector{});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{indices_tensor_shape, value_shape, slice_tensor_shape, slice_tensor_shape,
                                          slice_tensor_shape, empty_tensor_shape});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceToIndicesInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceToIndicesInferInner(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SliceToIndices, prim::kPrimSliceToIndices, SliceToIndicesInfer, false);
}  // namespace ops
}  // namespace mindspore
