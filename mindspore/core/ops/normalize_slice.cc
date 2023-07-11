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

#include "ops/normalize_slice.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/structure_ops.h"
#include "ops/normalize_dim_index.h"

namespace mindspore {
namespace ops {
static abstract::AbstractTuplePtr VectorToTuple(const std::vector<int64_t> &nums) {
  abstract::AbstractBasePtrList elems;
  std::transform(nums.begin(), nums.end(), std::back_inserter(elems),
                 [](int64_t num) { return std::make_shared<abstract::AbstractScalar>(num); });
  return std::make_shared<abstract::AbstractTuple>(elems);
}

AbstractBasePtr NormalizeSliceInfo(const std::vector<int64_t> &init_by_none, const ShapeVector &data_shape,
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
    GetValue<int64_t>(step_abs->BuildValue()), data_shape[new_dim_index], init_by_none, false);
  if (slice_ptr->is_empty_slice()) {
    auto stub_slice = std::make_shared<abstract::AbstractTuple>(
      abstract::AbstractBasePtrList{std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(1))});
    return std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{stub_slice, stub_slice, stub_slice});
  }

  size_t data_dim = data_shape.size();
  std::vector<int64_t> start_strides(data_dim, 0);
  std::vector<int64_t> stop_strides = data_shape;
  std::vector<int64_t> step_strides(data_dim, 1);
  start_strides[0] = slice_ptr->start();
  stop_strides[0] = slice_ptr->stop();
  step_strides[0] = slice_ptr->step();
  if (!tuple_index_types.empty()) {
    return VectorToTuple({start_strides[0], stop_strides[0], step_strides[0]});
  }
  auto slice_infos = std::vector<AbstractBasePtr>{VectorToTuple(start_strides), VectorToTuple(stop_strides),
                                                  VectorToTuple(step_strides)};
  return std::make_shared<abstract::AbstractTuple>(slice_infos);
}

AbstractBasePtr NormalizeSliceInferInner(const PrimitivePtr &primitive,
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
    return NormalizeSliceInfo(init_by_none, data_shape, input_args[kIndex1], input_args[kIndex2], input_args[kIndex3],
                              dim_index, tuple_index_types, expand_dims_mask);
  }
  auto abs_any = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  auto abs_tensor =
    std::make_shared<abstract::AbstractTensor>(abs_any, std::make_shared<abstract::Shape>(std::vector<int64_t>{1}));
  // Used in x[:], following op is dynamic StridedSlice so output is a tuple of tensors.
  auto output_any_abs =
    std::make_shared<abstract::AbstractTuple>(abstract::AbstractBasePtrList{abs_tensor, abs_tensor, abs_tensor});

  return output_any_abs;
}
MIND_API_OPERATOR_IMPL(NormalizeSlice, BaseOperator);
class NormalizeSliceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto slice_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{1});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{slice_shape, slice_shape, slice_shape});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeSliceInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeSliceInferInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(NormalizeSlice, prim::kPrimNormalizeSlice, NormalizeSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
