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

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include "abstract/abstract_value.h"
#include "abstract/ops/infer_functions.h"
#include "abstract/param_validator.h"
#include "abstract/utils.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
namespace {
const int kInputIndex0 = 0;
const int kInputIndex1 = 1;
const int kInputIndex2 = 2;
#define IsSameType(source_type, cmp_type) (cmp_type->equal(source_type))
#define IsNoneOrAnyValue(value_ptr) ((value_ptr->isa<None>()) || (value_ptr->isa<AnyValue>()))
// Get 3rd argument for UnsortedSegmentOps' inferImpl function
int64_t GetUnsortedSegmentOpScalarArg(const AbstractBasePtrList &args_spec_list, const std::string &op_name) {
  int64_t num_segments_value = 0;
  constexpr size_t scalar_index = 2;
  constexpr size_t min_len = 3;
  if (args_spec_list.size() < min_len) {
    MS_LOG(EXCEPTION) << "Index out of range, the len of args_spec_list is: " << args_spec_list.size()
                      << " and the index is " << scalar_index;
  }
  if (args_spec_list[scalar_index]->isa<AbstractTensor>()) {  // num_segments is Tensor
    auto num_segments = args_spec_list[scalar_index]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments);
    auto num_segments_value_ptr = num_segments->BuildValue();
    MS_EXCEPTION_IF_NULL(num_segments_value_ptr);
    auto num_segments_tensor = num_segments_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments_tensor);
    if (num_segments->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
      num_segments_value = *static_cast<int64_t *>(num_segments_tensor->data_c());
    } else {
      num_segments_value = *static_cast<int32_t *>(num_segments_tensor->data_c());
    }
  } else if (args_spec_list[scalar_index]->isa<AbstractScalar>()) {  // num_segments is Scalar
    auto num_segments = CheckArg<AbstractScalar>(op_name, args_spec_list, scalar_index);
    if (num_segments->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
      num_segments_value = GetValue<int64_t>(num_segments->BuildValue());
    } else {
      num_segments_value = GetValue<int32_t>(num_segments->BuildValue());
    }
  } else {
    MS_LOG(EXCEPTION) << "num_segments incorrect type in " << op_name;
  }
  return num_segments_value;
}

template <typename T>
int64_t RangeCalculateShape(const tensor::TensorPtr start_ptr, const tensor::TensorPtr limit_ptr,
                            const tensor::TensorPtr delta_ptr) {
  T start = *(reinterpret_cast<T *>(start_ptr->data_c()));
  T limit = *(reinterpret_cast<T *>(limit_ptr->data_c()));
  T delta = *(reinterpret_cast<T *>(delta_ptr->data_c()));
  bool valid_value = (delta == T(0) || (delta > 0 && start > limit) || (delta < 0 && start < limit));
  if (valid_value) {
    if (delta == T(0)) {
      MS_EXCEPTION(ValueError) << "For Range, delta cannot be equal to zero.";
    }
    if (delta > 0 && start > limit) {
      MS_EXCEPTION(ValueError) << "For Range, delta cannot be positive when limit < start.";
    }
    if (delta < 0 && start < limit) {
      MS_EXCEPTION(ValueError) << "For Range, delta cannot be negative when limit > start.";
    }
  }
  int64_t shape_size = 0;
  if (std::is_integral<T>::value) {
    shape_size = static_cast<int64_t>((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
  } else {
    shape_size = static_cast<int64_t>(std::ceil(std::abs((limit - start) / delta)));
  }
  return shape_size;
}

abstract::ShapePtr RangeCheckAndInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  int64_t shape_size = abstract::Shape::kShapeDimAny;
  auto start_value = input_args[kInputIndex0]->BuildValue();
  auto limit_value = input_args[kInputIndex1]->BuildValue();
  auto delta_value = input_args[kInputIndex2]->BuildValue();
  MS_EXCEPTION_IF_NULL(start_value);
  MS_EXCEPTION_IF_NULL(limit_value);
  MS_EXCEPTION_IF_NULL(delta_value);

  bool is_compile = (IsNoneOrAnyValue(start_value) || IsNoneOrAnyValue(limit_value) || IsNoneOrAnyValue(delta_value));
  // not in compile, need inferShape
  if (!is_compile) {
    auto op_name = "Range";
    auto dtype = CheckAndConvertUtils::GetTensorInputType(op_name, input_args, kInputIndex0);
    auto start_tensor = start_value->cast<tensor::TensorPtr>();
    auto limit_tensor = limit_value->cast<tensor::TensorPtr>();
    auto delta_tensor = delta_value->cast<tensor::TensorPtr>();
    if (IsSameType(dtype, kInt) || IsSameType(dtype, kInt32)) {
      shape_size = RangeCalculateShape<int32_t>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kInt64)) {
      shape_size = RangeCalculateShape<int64_t>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kFloat) || IsSameType(dtype, kFloat32)) {
      shape_size = RangeCalculateShape<float>(start_tensor, limit_tensor, delta_tensor);
    } else if (IsSameType(dtype, kFloat64)) {
      shape_size = RangeCalculateShape<double>(start_tensor, limit_tensor, delta_tensor);
    } else {
      MS_EXCEPTION(TypeError) << "For Range, the dtype of input must be int32, int64, float32, float64, but got "
                              << dtype->meta_type() << ".";
    }
    if (shape_size < 0) {
      MS_EXCEPTION(ValueError) << "For Range, infer shape error, shape_size [" << shape_size << "] is negative.";
    }
  }

  ShapeVector out_shape = {};
  if (is_compile) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  (void)out_shape.emplace_back(shape_size);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr RangeCheckAndInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::set<TypePtr> support_types = {kInt32, kInt64, kFloat32, kFloat64};
  auto start_type = CheckAndConvertUtils::CheckTensorTypeValid("start", input_args[kInputIndex0]->BuildType(),
                                                               support_types, prim->name());
  auto limit_type = CheckAndConvertUtils::CheckTensorTypeValid("limit", input_args[kInputIndex1]->BuildType(),
                                                               support_types, prim->name());
  auto delta_type = CheckAndConvertUtils::CheckTensorTypeValid("delta", input_args[kInputIndex2]->BuildType(),
                                                               support_types, prim->name());
  MS_EXCEPTION_IF_NULL(start_type);
  MS_EXCEPTION_IF_NULL(limit_type);
  MS_EXCEPTION_IF_NULL(delta_type);
  bool same_type = IsSameType(start_type, limit_type) && IsSameType(limit_type, delta_type);
  if (!same_type) {
    MS_EXCEPTION(TypeError) << "For Range, start, limit delta should have same type, but get start["
                            << start_type->meta_type() << "], limit[" << limit_type->meta_type() << "], delta["
                            << delta_type->meta_type() << "].";
  }
  return start_type;
}
}  // namespace
AbstractBasePtr InferImplScalarToArray(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a scalar.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractScalarPtr arg = CheckArg<AbstractScalar>(op_name, args_spec_list, 0);
  return std::make_shared<AbstractTensor>(arg, std::make_shared<Shape>());
}

AbstractBasePtr InferImplArrayToScalar(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor with 0 shape.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto arg = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto a_shp = arg->shape();
  MS_EXCEPTION_IF_NULL(a_shp);
  if (!a_shp->shape().empty()) {
    MS_LOG(EXCEPTION) << "array_to_scalar requires zero size shape.";
  }
  return arg->element();
}

AbstractBasePtr InferImplBroadCastShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples.
  const std::string op_name = primitive->name();
  constexpr size_t args_size = 2;
  CheckArgsSize(op_name, args_spec_list, args_size);
  auto xs = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  auto ys = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  auto x_value = xs->BuildValue();
  MS_EXCEPTION_IF_NULL(x_value);
  auto value_tuple_x = x_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_x);
  auto shp_tuple_x = value_tuple_x->value();
  ShapeVector shp_x;
  (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(shp_x),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  auto tupe_value_y = ys->BuildValue();
  MS_EXCEPTION_IF_NULL(tupe_value_y);
  auto value_tuple_y = tupe_value_y->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_y);
  auto shp_tuple_y = value_tuple_y->value();
  ShapeVector shp_y;
  (void)std::transform(std::begin(shp_tuple_y), std::end(shp_tuple_y), std::back_inserter(shp_y),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });

  ShapeVector res = BroadcastShape(shp_x, shp_y);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  if (res.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }

  AbstractBasePtrList elems;
  (void)std::transform(res.begin(), res.end(), std::back_inserter(elems), [](int64_t n) -> AbstractBasePtr {
    return std::make_shared<AbstractScalar>(std::make_shared<Int64Imm>(n), kInt64);
  });
  return std::make_shared<AbstractTuple>(elems);
}

AbstractBasePtr InferImplUnique(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  auto shape = input->shape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
  }

  bool is_input_dynamic = IsDynamic(shape->shape());

  ShapeVector ids_shape = {Shape::kShapeDimAny};
  ShapeVector min_shape;
  ShapeVector max_shape;
  if (!is_input_dynamic) {
    max_shape = shape->shape();
    min_shape.push_back(1);
  }

  auto ids =
    std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(ids_shape, min_shape, max_shape));
  // Currently we choose the same data type as input for the idx.
  TypePtr ids_idx_type = kInt32;
  MS_EXCEPTION_IF_NULL(input->element());
  MS_EXCEPTION_IF_NULL(input->element()->GetTypeTrack());
  if (input->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
    ids_idx_type = kInt64;
  }
  ShapeVector idx_shape = shape->shape();
  ShapeVector idx_min_shape = shape->min_shape();
  if (idx_min_shape.empty() && !is_input_dynamic) {
    idx_min_shape = shape->shape();
  }
  ShapeVector idx_max_shape = shape->max_shape();
  if (idx_max_shape.empty() && !is_input_dynamic) {
    idx_max_shape = shape->shape();
  }

  auto ids_idx = std::make_shared<AbstractTensor>(ids_idx_type, idx_shape);
  ids_idx->set_shape(std::make_shared<Shape>(idx_shape, idx_min_shape, idx_max_shape));
  // outputs: ids, ids_idx
  AbstractBasePtrList elements = {ids, ids_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplUniqueWithPad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  constexpr size_t kUniqueWithPadInputNum = 2;
  constexpr size_t kPadIndex = 1;
  CheckArgsSize(op_name, args_spec_list, kUniqueWithPadInputNum);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto shape = input->shape();
  MS_EXCEPTION_IF_NULL(shape);
  size_t batch_rank = 0;
  if (primitive->HasAttr(ops::kBatchRank)) {
    auto value_ptr = primitive->GetAttr(ops::kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  if (batch_rank != 0) {
    (void)CheckAndConvertUtils::CheckInteger("input_shape size", shape->shape().size(), kEqual, batch_rank + 1,
                                             op_name);
    AbstractTensorPtr pad = CheckArg<AbstractTensor>(op_name, args_spec_list, kPadIndex);
    auto pad_shape = pad->shape();
    MS_EXCEPTION_IF_NULL(pad_shape);
    auto pad_num = std::accumulate(pad_shape->shape().begin(), pad_shape->shape().end(), 1, std::multiplies<int64_t>());
    auto input_batch =
      std::accumulate(shape->shape().begin(), shape->shape().begin() + batch_rank, 1, std::multiplies<int64_t>());
    (void)CheckAndConvertUtils::CheckInteger("elements num of input 'pad'", pad_num, kEqual, input_batch, op_name);
  } else {
    if (shape->shape().size() != 1) {
      MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
    }
  }

  // Currently we choose the same data type as input for the idx.
  TypePtr ids_idx_type = kInt32;
  MS_EXCEPTION_IF_NULL(input->element());
  MS_EXCEPTION_IF_NULL(input->element()->GetTypeTrack());
  if (input->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
    ids_idx_type = kInt64;
  }
  ShapeVector idx_shape = shape->shape();
  ShapeVector idx_min_shape = shape->min_shape();
  if (idx_min_shape.empty()) {
    idx_min_shape = shape->shape();
  }
  ShapeVector idx_max_shape = shape->max_shape();
  if (idx_max_shape.empty()) {
    idx_max_shape = shape->shape();
  }
  auto ids_idx = std::make_shared<AbstractTensor>(ids_idx_type, idx_shape);
  ids_idx->set_shape(std::make_shared<Shape>(idx_shape, idx_min_shape, idx_max_shape));

  AbstractBasePtr ids = input->Broaden();
  return std::make_shared<AbstractTuple>(AbstractBasePtrList({ids, ids_idx}));
}

AbstractBasePtr InferImplPadAndShift(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(input);
  auto shape = input->shape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
  }
  ShapeVector ids_shape = {Shape::kShapeDimAny};
  ShapeVector min_shape;
  ShapeVector max_shape;
  ShapeVector input_shape = shape->shape();
  if (!IsDynamic(input_shape)) {
    max_shape = input_shape;
    min_shape.push_back(1);
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(ids_shape, min_shape, max_shape));
}

AbstractBasePtr InferImplUniqueGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  AbstractTuplePtr dout = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  CheckArgsSize(op_name + " dout", dout->elements(), size_expected);
  auto ids = CheckArg<AbstractTensor>(op_name, dout->elements(), 0);
  auto ids_idx = CheckArg<AbstractTensor>(op_name, dout->elements(), 1);
  auto ids_shape = ids->shape();
  auto ids_idx_shape = ids_idx->shape();
  MS_EXCEPTION_IF_NULL(ids_shape);
  MS_EXCEPTION_IF_NULL(ids_idx_shape);
  if (ids->shape()->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Dims of dout[0] of " << op_name << "' input must be 1.";
  }
  if (ids_idx->shape()->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Dims of dout[1] of " << op_name << "' input must be 1.";
  }

  // outputs: dx
  return std::make_shared<AbstractTensor>(ids->element(), ids_idx->shape());
}

AbstractBasePtr InferImplUnsortedSegmentSum(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  constexpr size_t args_size = 3;
  CheckArgsSize(op_name, args_spec_list, args_size);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto segment_ids = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(segment_ids);
  MS_EXCEPTION_IF_NULL(segment_ids->shape());
  auto segment_ids_shape = segment_ids->shape()->shape();
  (void)CheckTensorDType(x, {kFloat16, kFloat32, kFloat64, kInt32}, "Input 0 (x) for UnsortedSegmentSum should be %s");
  (void)CheckTensorDType(segment_ids, {kInt32, kInt64}, "Input 1 (segment_ids) for UnsortedSegmentSum should be %s");
  bool x_is_dyn = (!x->shape()->min_shape().empty() && !x->shape()->max_shape().empty());  // check if dynamic shape
  bool ids_is_dyn = (!segment_ids->shape()->min_shape().empty() && !segment_ids->shape()->max_shape().empty());
  bool op_is_dynamic = x_is_dyn || ids_is_dyn;
  auto x_shape = x->shape()->shape();
  ShapeVector shape;
  int64_t num_segments_value = GetUnsortedSegmentOpScalarArg(args_spec_list, op_name);
  if (num_segments_value <= 0) {
    MS_LOG(EXCEPTION) << "num_segments must be > 0 in UnsortedSegmentSum";
  }
  shape.emplace_back(num_segments_value);
  shape.insert(shape.end(), x_shape.begin() + segment_ids_shape.size(), x_shape.end());
  if (!op_is_dynamic) {  // not dynamic
    for (size_t i = 0; i < segment_ids_shape.size(); i++) {
      if (x_shape[i] != segment_ids_shape[i]) {
        MS_LOG(EXCEPTION) << "Shape values of segments_ids must match with corresponding x shape values";
      }
    }
    return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
  }
  ShapeVector min_shape;
  ShapeVector max_shape;
  min_shape.emplace_back(num_segments_value);
  max_shape.emplace_back(num_segments_value);
  bool x_any_shape = IsDynamic(x_shape);
  bool ids_any_shape = IsDynamic(segment_ids_shape);
  if (!x_any_shape && !ids_any_shape) {  // only validate when shapes fully known
    for (size_t i = 0; i < segment_ids_shape.size(); i++) {
      if (x_shape[i] != segment_ids_shape[i]) {
        MS_LOG(EXCEPTION) << "Shape values of segments_ids must match with corresponding x shape values";
      }
    }
  }
  ShapeVector x_shape_min;
  ShapeVector x_shape_max;
  x_shape_min = (x_is_dyn) ? x->shape()->min_shape() : x->shape()->shape();
  x_shape_max = (x_is_dyn) ? x->shape()->max_shape() : x->shape()->shape();
  min_shape.insert(min_shape.end(), x_shape_min.begin() + segment_ids_shape.size(), x_shape_min.end());
  max_shape.insert(max_shape.end(), x_shape_max.begin() + segment_ids_shape.size(), x_shape_max.end());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplUnsortedSegmentMax(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto segment_ids = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(segment_ids);
  MS_EXCEPTION_IF_NULL(segment_ids->shape());
  auto segment_ids_shape = segment_ids->shape()->shape();
  (void)CheckTensorDType(x, {kFloat16, kFloat32, kInt32}, "Input 0 (x) for UnsortedSegmentMax should be %s");
  (void)CheckTensorDType(segment_ids, {kInt32, kInt64}, "Input 1 (segment_ids) for UnsortedSegmentMax should be %s");
  bool x_is_dyn = (!x->shape()->min_shape().empty() && !x->shape()->max_shape().empty());  // check if dynamic
  bool ids_is_dyn = (!segment_ids->shape()->min_shape().empty() && !segment_ids->shape()->max_shape().empty());
  bool op_is_dynamic = x_is_dyn || ids_is_dyn;
  auto x_shape = x->shape()->shape();
  ShapeVector shape;
  int64_t num_segments_value = GetUnsortedSegmentOpScalarArg(args_spec_list, op_name);
  if (num_segments_value <= 0) {
    MS_LOG(EXCEPTION) << "num_segments must be > 0 in UnsortedSegmentMax";
  }
  shape.emplace_back(num_segments_value);
  shape.insert(shape.end(), x_shape.begin() + segment_ids_shape.size(), x_shape.end());
  if (!op_is_dynamic) {  // not dynamic
    if (x_shape[0] != segment_ids_shape[0]) {
      MS_LOG(EXCEPTION) << "Length of segment_ids must match first value of x shape UnsortedSegmentMax";
    }
    return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
  }
  ShapeVector min_shape;
  ShapeVector max_shape;
  min_shape.emplace_back(num_segments_value);
  max_shape.emplace_back(num_segments_value);
  bool x_any_shape = IsDynamic(x_shape);
  bool ids_any_shape = IsDynamic(segment_ids_shape);
  if (!x_any_shape && !ids_any_shape) {
    if (x_shape[0] != segment_ids_shape[0]) {
      MS_LOG(EXCEPTION) << "Length of segment_ids must match first value of x shape UnsortedSegmentMax";
    }
  }
  ShapeVector x_shape_min;
  ShapeVector x_shape_max;
  x_shape_min = (x_is_dyn) ? x->shape()->min_shape() : x->shape()->shape();
  x_shape_max = (x_is_dyn) ? x->shape()->max_shape() : x->shape()->shape();
  min_shape.insert(min_shape.end(), x_shape_min.begin() + segment_ids_shape.size(), x_shape_min.end());
  max_shape.insert(max_shape.end(), x_shape_max.begin() + segment_ids_shape.size(), x_shape_max.end());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplUnsortedSegmentMin(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto segment_ids = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(segment_ids);
  MS_EXCEPTION_IF_NULL(segment_ids->shape());
  auto segment_ids_shape = segment_ids->shape()->shape();
  (void)CheckTensorDType(x, {kFloat16, kFloat32, kInt32}, "Input 0 (x) for UnsortedSegmentMin should be %s");
  (void)CheckTensorDType(segment_ids, {kInt32}, "Input 1 (segment_ids) for UnsortedSegmentMin should be %s");
  bool x_is_dyn = (!x->shape()->min_shape().empty() && !x->shape()->max_shape().empty());  // check if dynamic shape
  bool ids_is_dyn = (!segment_ids->shape()->min_shape().empty() && !segment_ids->shape()->max_shape().empty());
  bool op_is_dynamic = x_is_dyn || ids_is_dyn;
  auto x_shape = x->shape()->shape();
  ShapeVector shape;
  int64_t num_segments_value = GetUnsortedSegmentOpScalarArg(args_spec_list, op_name);
  if (num_segments_value <= 0) {
    MS_LOG(EXCEPTION) << "num_segments must be > 0 in UnsortedSegmentMin";
  }
  shape.emplace_back(num_segments_value);
  shape.insert(shape.end(), x_shape.begin() + segment_ids_shape.size(), x_shape.end());
  if (!op_is_dynamic) {  // not dynamic
    if (x_shape[0] != segment_ids_shape[0]) {
      MS_LOG(EXCEPTION) << "Length of segment_ids must match first value of x shape UnsortedSegmentMin";
    }
    return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
  }
  ShapeVector min_shape;
  ShapeVector max_shape;
  min_shape.emplace_back(num_segments_value);
  max_shape.emplace_back(num_segments_value);
  bool x_any_shape = IsDynamic(x_shape);
  bool ids_any_shape = IsDynamic(segment_ids_shape);
  if (!x_any_shape && !ids_any_shape) {  // only validate when shapes fully known
    if (x_shape[0] != segment_ids_shape[0]) {
      MS_LOG(EXCEPTION) << "Length of segment_ids must match first value of x shape UnsortedSegmentMin";
    }
  }
  ShapeVector x_shape_min;
  ShapeVector x_shape_max;
  x_shape_min = (x_is_dyn) ? x->shape()->min_shape() : x->shape()->shape();
  x_shape_max = (x_is_dyn) ? x->shape()->max_shape() : x->shape()->shape();
  min_shape.insert(min_shape.end(), x_shape_min.begin() + segment_ids_shape.size(), x_shape_min.end());
  max_shape.insert(max_shape.end(), x_shape_max.begin() + segment_ids_shape.size(), x_shape_max.end());
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplScatterAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  constexpr auto kScatterAddInputNum = 3;
  const std::string op_name = primitive->name();
  CheckRequiredArgsSize(op_name, args_spec_list, kScatterAddInputNum);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  ShapeVector min_shape = x->shape()->min_shape();
  ShapeVector max_shape = x->shape()->max_shape();
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplScatterSub(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  constexpr auto kScatterSubInputNum = 3;
  const std::string op_name = primitive->name();
  CheckRequiredArgsSize(op_name, args_spec_list, kScatterSubInputNum);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  ShapeVector shape = x->shape()->shape();
  ShapeVector min_shape = x->shape()->min_shape();
  ShapeVector max_shape = x->shape()->max_shape();
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplMapCacheIdx(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 5;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto hash_map = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(hash_map->shape());

  auto indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto indices_shp = indices->shape();
  MS_EXCEPTION_IF_NULL(indices_shp);

  ShapeVector shape(indices_shp->shape().size(), -1);
  ShapeVector min_shape = indices_shp->min_shape();
  ShapeVector max_shape = indices_shp->max_shape();

  auto cache_idx = std::make_shared<AbstractTensor>(hash_map->element(), indices->shape());
  auto old_emb_idx =
    std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  auto miss_emb_idx =
    std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  auto swap_emb_idx =
    std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape, min_shape, max_shape));

  AbstractBasePtrList elements = {cache_idx, old_emb_idx, miss_emb_idx, swap_emb_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplCacheSwapTable(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto cache_table = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto cache_table_shp = cache_table->shape();
  MS_EXCEPTION_IF_NULL(cache_table_shp);

  auto swap_cache_idx = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto swap_cache_idx_shp = swap_cache_idx->shape();
  MS_EXCEPTION_IF_NULL(swap_cache_idx_shp);

  auto cache_table_shape = cache_table_shp->shape();
  auto swap_cache_idx_shape = swap_cache_idx_shp->shape();
  ShapeVector shape;
  shape.emplace_back(swap_cache_idx_shape[0]);
  shape.emplace_back(cache_table_shape[1]);
  auto swap_cache_idx_max_shape = swap_cache_idx_shp->max_shape();
  ShapeVector max_shape;
  ShapeVector min_shape;
  if (!swap_cache_idx_max_shape.empty() && cache_table_shape[1] > 0) {
    max_shape.emplace_back(swap_cache_idx_max_shape[0]);
    max_shape.emplace_back(cache_table_shape[1]);
  }

  for (size_t i = 0; i < max_shape.size(); ++i) {
    min_shape.emplace_back(1);
  }

  AbstractTensorPtr ret =
    std::make_shared<AbstractTensor>(cache_table->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  return ret;
}

AbstractBasePtr InferImplUpdateCache(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  ShapeVector shape;
  shape.emplace_back(1);

  AbstractTensorPtr ret = std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape));
  return ret;
}

AbstractBasePtr InferImplSubAndFilter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  auto input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_x_shp = input_x->shape();
  MS_EXCEPTION_IF_NULL(input_x_shp);

  ShapeVector shape(input_x_shp->shape().size(), -1);
  ShapeVector min_shape = input_x_shp->min_shape();
  ShapeVector max_shape = input_x_shp->max_shape();

  auto filter_res =
    std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  auto filter_idx =
    std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  AbstractBasePtrList elements = {filter_res, filter_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector y_shape = y->shape()->shape();
  ShapeVector out_shape = BroadcastShape(x_shape, y_shape);
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplRealDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto y = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(y->shape());
  ShapeVector x_shape = x->shape()->shape();
  ShapeVector y_shape = y->shape()->shape();
  ShapeVector out_shape = BroadcastShape(x_shape, y_shape);
  if (out_shape.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplGatherV2(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  constexpr size_t args_size = 3;
  CheckArgsSize(op_name, args_spec_list, args_size);
  AbstractTensorPtr params = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  AbstractTensorPtr indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  bool ind_dyn = (!indices->shape()->min_shape().empty() && !indices->shape()->max_shape().empty());
  bool param_dyn = (!params->shape()->min_shape().empty() && !params->shape()->max_shape().empty());
  int64_t axis_val = 0;
  // 3rd input is a Tensor when GatherV2 is a dynamic shape operator
  constexpr size_t aixs_index = 2;
  if (args_spec_list[aixs_index]->isa<AbstractTensor>()) {
    auto axis = args_spec_list[aixs_index]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis);
    auto axis_value_ptr = axis->BuildValue();
    MS_EXCEPTION_IF_NULL(axis_value_ptr);
    auto axis_tensor = axis_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_tensor);
    axis_val = *static_cast<int64_t *>(axis_tensor->data_c());
  } else if (args_spec_list[aixs_index]->isa<AbstractScalar>()) {
    auto axis = args_spec_list[aixs_index]->cast<AbstractScalarPtr>();
    axis_val = GetValue<int64_t>(axis->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << args_spec_list[2]->type_name();
  }
  auto params_shp = params->shape()->shape();
  auto indices_shp = indices->shape()->shape();
  auto params_rank = static_cast<int64_t>(params_shp.size());
  // either inputs or both can be dynamic and computation requires min/max shapes for both
  ShapeVector param_shp_min = (param_dyn) ? params->shape()->min_shape() : params->shape()->shape();
  ShapeVector param_shp_max = (param_dyn) ? params->shape()->max_shape() : params->shape()->shape();
  ShapeVector indices_shp_min = (ind_dyn) ? indices->shape()->min_shape() : indices->shape()->shape();
  ShapeVector indices_shp_max = (ind_dyn) ? indices->shape()->max_shape() : indices->shape()->shape();
  // check axis_val within interval: [-params_rank, params_rank)
  if (-params_rank > axis_val || axis_val >= params_rank) {
    MS_LOG(EXCEPTION) << "For Gather - Axis value must be within [ " << -params_rank << ", " << params_rank << " ) "
                      << "Got " << axis_val << ".";
  }
  if (axis_val < 0) {
    axis_val += params_rank;
  }
  auto calc_shape = [axis_val](const ShapeVector &ind_vec, const ShapeVector &params_vec) -> ShapeVector {
    ShapeVector out_vec;
    std::copy(params_vec.begin(), params_vec.begin() + axis_val, std::back_inserter(out_vec));
    copy(ind_vec.begin(), ind_vec.end(), std::back_inserter(out_vec));
    copy(params_vec.begin() + axis_val + 1, params_vec.end(), std::back_inserter(out_vec));
    return out_vec;
  };
  ShapeVector out_shape = calc_shape(indices_shp, params_shp);
  if (ind_dyn || param_dyn) {
    ShapeVector min_shape = calc_shape(indices_shp_min, param_shp_min);
    ShapeVector max_shape = calc_shape(indices_shp_max, param_shp_max);
    return std::make_shared<AbstractTensor>(params->element(),
                                            std::make_shared<Shape>(out_shape, min_shape, max_shape));
  }
  return std::make_shared<AbstractTensor>(params->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplDynamicAssign(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor
  const size_t size_expected = 2;
  CheckArgsSize(primitive->name(), args_spec_list, size_expected);

  MS_LOG(INFO) << "InferImplDynamicAssign " << args_spec_list[0];
  auto type = args_spec_list[0]->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  if (type->type_id() == kObjectTypeRefKey) {
    return args_spec_list[1]->Broaden();
  } else {
    auto x = CheckArg<AbstractTensor>(primitive->name(), args_spec_list, 0);
    auto y = CheckArg<AbstractTensor>(primitive->name(), args_spec_list, 1);
    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(y);
    auto y_shape = y->shape();
    MS_EXCEPTION_IF_NULL(y_shape);
    if (!y_shape->max_shape().empty()) {
      x->set_shape(y->shape());
    }
    return args_spec_list[0];
  }
}

AbstractBasePtr InferImplEmbeddingLookup(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  auto params = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto params_shp = params->shape();
  MS_EXCEPTION_IF_NULL(params_shp);
  auto params_shape = params_shp->shape();
  auto indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  auto indices_shp = indices->shape();
  MS_EXCEPTION_IF_NULL(indices_shp);
  auto indices_shape = indices_shp->shape();
  auto indices_max_shape = indices_shp->max_shape();
  auto indices_min_shape = indices_shp->min_shape();
  ShapeVector shape;
  ShapeVector max_shape;
  ShapeVector min_shape;
  shape.insert(shape.end(), indices_shape.begin(), indices_shape.end());
  shape.insert(shape.end(), params_shape.begin() + 1, params_shape.end());
  if (!indices_max_shape.empty()) {
    max_shape.insert(max_shape.end(), indices_max_shape.begin(), indices_max_shape.end());
    max_shape.insert(max_shape.end(), params_shape.begin() + 1, params_shape.end());
  }
  if (!indices_min_shape.empty()) {
    min_shape.insert(min_shape.end(), indices_min_shape.begin(), indices_min_shape.end());
    min_shape.insert(min_shape.end(), params_shape.begin() + 1, params_shape.end());
  }

  AbstractTensorPtr ret =
    std::make_shared<AbstractTensor>(params->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
  return ret;
}

AbstractBasePtr InferImplTranspose(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto input_shp = input->shape()->shape();
  ValuePtr perm = primitive->GetAttr("perm");
  MS_EXCEPTION_IF_NULL(perm);
  auto perm_val = perm->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(perm_val);
  auto perm_val_data = perm_val->value();
  ShapeVector perm_vec;
  (void)std::transform(std::begin(perm_val_data), std::end(perm_val_data), std::back_inserter(perm_vec),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  ShapeVector result_shp;
  ShapeVector max_shp;
  ShapeVector min_shp;
  ShapeVector x_max_shp = input->shape()->max_shape();
  ShapeVector x_min_shp = input->shape()->min_shape();
  for (size_t i = 0; i < perm_vec.size(); i++) {
    auto idx = static_cast<size_t>(perm_vec[i]);
    result_shp.push_back(input_shp[idx]);
    if (!x_max_shp.empty() && !x_min_shp.empty()) {
      max_shp.push_back(x_max_shp[idx]);
      min_shp.push_back(x_min_shp[idx]);
    }
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(result_shp, min_shp, max_shp));
}

AbstractBasePtr InferImplMapUniform(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // Inputs: one tensor.
  const std::string op_name = primitive->name();
  const size_t size_expected = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplSplit(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input_x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  ShapeVector x_shape = input_x->shape()->shape();
  ShapeVector x_shape_min = input_x->shape()->min_shape();
  ShapeVector x_shape_max = input_x->shape()->max_shape();
  int64_t rank = SizeToLong(x_shape.size());

  ValuePtr axis = primitive->GetAttr("axis");
  int64_t axis_value_pos = CheckAxis(op_name, "axis", axis, -(rank + 1), rank, "input_x");
  int64_t output_num_value = GetValue<int64_t>(primitive->GetAttr("output_num"));
  size_t pos = LongToSize(axis_value_pos);
  if ((x_shape[pos] != Shape::kShapeDimAny) && (x_shape[pos] % output_num_value != 0)) {
    MS_LOG(EXCEPTION) << "x_shape[" << pos << "] = " << x_shape[pos]
                      << " must be divisible by output_num = " << output_num_value;
  }

  ShapeVector output_shape = x_shape;
  if (output_shape[pos] != Shape::kShapeDimAny) {
    output_shape[pos] = static_cast<int>(x_shape[pos] / output_num_value);
  }

  if (!x_shape_min.empty() && !x_shape_max.empty()) {
    x_shape_min[pos] = static_cast<int>(x_shape_min[pos] / output_num_value);
    x_shape_max[pos] = static_cast<int>(x_shape_max[pos] / output_num_value);
  }

  AbstractBasePtrList output_list;
  for (int64_t i = 0; i < output_num_value; ++i) {
    auto output = input_x->Broaden();
    output->set_shape(std::make_shared<Shape>(output_shape, x_shape_min, x_shape_max));
    output_list.push_back(output);
  }
  return std::make_shared<AbstractTuple>(output_list);
}

AbstractBasePtr InferImplSequenceMask(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  const size_t size_expected = 2;
  CheckArgsSize(op_name, args_spec_list, size_expected);

  AbstractTensorPtr lengths = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  (void)CheckTensorDType(lengths, {kInt32, kInt64}, "Input 1 (lengths) for SequenceMask should be one of: %s");

  int64_t maxlen_value = 0;

  if (args_spec_list[1]->isa<AbstractScalar>()) {
    AbstractScalarPtr maxlen = CheckArg<AbstractScalar>(op_name, args_spec_list, 1);
    (void)CheckScalarType(maxlen, {kInt32, kInt64}, "Input 0 (maxlen) for SequenceMask should be one of: %s");

    TypePtr maxlen_type = nullptr;
    maxlen_type = maxlen->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(maxlen_type);

    if (maxlen_type->type_id() == TypeId::kNumberTypeInt32) {
      maxlen_value = static_cast<int64_t>(GetValue<int32_t>(maxlen->BuildValue()));
    } else if (maxlen_type->type_id() == TypeId::kNumberTypeInt64) {
      maxlen_value = GetValue<int64_t>(maxlen->BuildValue());
    }
  } else if (args_spec_list[1]->isa<AbstractTensor>()) {
    auto maxlen_tensor_ptr = args_spec_list[1]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(maxlen_tensor_ptr);
    auto maxlen_value_ptr = maxlen_tensor_ptr->BuildValue();
    MS_EXCEPTION_IF_NULL(maxlen_value_ptr);
    auto maxlen_tensor = maxlen_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(maxlen_tensor);
    maxlen_value = *static_cast<int64_t *>(maxlen_tensor->data_c());
  }

  if (maxlen_value <= 0) {
    MS_LOG(EXCEPTION) << "maxlen must be positive, but got: " << maxlen_value;
  }

  ShapeVector lengths_shape = lengths->shape()->shape();
  ShapeVector lengths_shape_min = lengths->shape()->min_shape();
  ShapeVector lengths_shape_max = lengths->shape()->max_shape();
  if (!lengths_shape_max.empty() && !lengths_shape_min.empty()) {
    lengths_shape_min.push_back(maxlen_value);
    lengths_shape_max.push_back(maxlen_value);
  }

  lengths_shape.push_back(maxlen_value);

  ShapePtr output_shape = std::make_shared<Shape>(lengths_shape, lengths_shape_min, lengths_shape_max);
  return std::make_shared<AbstractTensor>(kBool, output_shape);
}

AbstractBasePtr InferImplConcatOffset(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list is empty.";
  }

  AbstractTuplePtr arg = nullptr;
  AbstractTensorPtr tensor_base = nullptr;
  size_t tuple_len = 0;
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  if (args_spec_list[0]->isa<AbstractTuple>()) {
    CheckArgsSize(op_name, args_spec_list, 1);
    arg = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
    tuple_len = arg->elements().size();
    tensor_base = CheckArg<AbstractTensor>(op_name, arg->elements(), 0);
  } else if (args_spec_list[0]->isa<AbstractTensor>()) {
    tuple_len = args_spec_list.size();
    tensor_base = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  }

  MS_EXCEPTION_IF_NULL(tensor_base);
  ShapeVector shape_base = tensor_base->shape()->shape();
  size_t rank = shape_base.size();
  ShapeVector out_shape{SizeToLong(tuple_len), SizeToLong(rank)};
  TypePtr out_type = kInt64;
  return std::make_shared<AbstractTensor>(out_type, std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list is empty.";
  }

  AbstractTuplePtr arg = nullptr;
  AbstractTensorPtr tensor_base = nullptr;
  size_t tuple_len = 0;
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  if (args_spec_list[0]->isa<AbstractTuple>()) {
    CheckArgsSize(op_name, args_spec_list, 1);
    arg = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
    tuple_len = arg->elements().size();
    tensor_base = CheckArg<AbstractTensor>(op_name, arg->elements(), 0);
  } else if (args_spec_list[0]->isa<AbstractTensor>()) {
    tuple_len = args_spec_list.size();
    tensor_base = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  }

  MS_EXCEPTION_IF_NULL(tensor_base);
  ShapeVector shape_base = tensor_base->shape()->shape();
  int64_t rank_base = SizeToLong(shape_base.size());
  primitive->set_attr("T", tensor_base->element()->BuildType());
  primitive->set_attr("inputNums", MakeValue(SizeToLong(tuple_len)));

  ValuePtr axis = primitive->GetAttr("axis");
  // Axis value should be in [-(rank_base + 1), rank_base).
  int64_t axis_value = CheckAxis(op_name, "axis", axis, -(rank_base + 1), rank_base, "input_x");

  int64_t all_shp = shape_base[axis_value];
  for (size_t i = 1; i < tuple_len; ++i) {
    AbstractTensorPtr tensor = nullptr;
    if (args_spec_list[0]->isa<AbstractTuple>()) {
      tensor = CheckArg<AbstractTensor>(op_name, arg->elements(), i);
    } else if (args_spec_list[0]->isa<AbstractTensor>()) {
      tensor = CheckArg<AbstractTensor>(op_name, args_spec_list, i);
    }
    ShapeVector shape_tensor = tensor->shape()->shape();
    int64_t rank_tensor = SizeToLong(shape_tensor.size());
    (void)CheckDtypeSame(op_name, tensor_base, tensor);
    if (rank_tensor != rank_base) {
      MS_LOG(EXCEPTION) << op_name << " can not concat element " << i << " with the first element: Wrong Rank";
    }
    for (int j = 0; j < rank_base; ++j) {
      if (j != axis_value && shape_tensor[j] != shape_base[j]) {
        MS_LOG(EXCEPTION) << op_name << " can not concat element " << i << " with the first element: Wrong Size";
      }
    }
    if (all_shp == -1 || shape_base[axis_value] == -1) {
      all_shp = -1;
    } else {
      all_shp += shape_tensor[axis_value];
    }
  }

  AbstractTensorPtr ret = dyn_cast<AbstractTensor>(tensor_base->Broaden());
  MS_EXCEPTION_IF_NULL(ret);
  auto shape = ret->shape()->shape();
  shape[axis_value] = all_shp;
  ret->set_shape(std::make_shared<Shape>(shape));
  return ret;
}

// Helper struct for FlattenConcat infer.
struct ChunkInfo {
  size_t bytes{0};  // number of bytes.
  size_t size{0};   // number of elements.
};

using ChunkMap = std::map<TypeId, std::vector<ChunkInfo>>;

// Group inputs by data type and fusion size.
static ChunkMap GroupingAbstractTensors(const AbstractBasePtrList &elements, size_t fusion_size,
                                        const std::string &prim_name) {
  ChunkMap chunk_map;
  for (auto &element : elements) {
    auto abs_tensor = dyn_cast<abstract::AbstractTensor>(element);
    if (abs_tensor == nullptr) {
      MS_LOG(EXCEPTION) << "The input element for '" << prim_name << "' should be Tensor, but got "
                        << element->type_name() << ".";
    }
    // Calculate data size (number of elements) by shape.
    auto base_shape = abs_tensor->BuildShape();
    MS_EXCEPTION_IF_NULL(base_shape);
    auto shape = base_shape->cast<ShapePtr>();
    if (shape == nullptr) {
      MS_LOG(EXCEPTION) << "The input tensors for '" << prim_name << "' should have shape, but got "
                        << base_shape->ToString() << ".";
    }
    auto data_size = SizeOf(shape->shape());
    if (data_size == 0) {
      MS_LOG(EXCEPTION) << "The input tensors for '" << prim_name << "'should have static shape, but got "
                        << shape->ToString() << ".";
    }
    // Find data type from the AbstractTensor.
    const auto &element_abs = abs_tensor->element();
    MS_EXCEPTION_IF_NULL(element_abs);
    auto dtype = element_abs->BuildType();
    MS_EXCEPTION_IF_NULL(dtype);
    const auto type_id = dtype->type_id();
    const auto data_bytes = data_size * abstract::TypeIdSize(type_id);
    if (fusion_size != 0 && fusion_size < data_bytes) {
      MS_LOG(EXCEPTION) << "Fusion size " << fusion_size << " is too small for a tensor size " << data_bytes << ".";
    }
    // Group them by data type and fusion size.
    auto &chunks = chunk_map[type_id];
    if (chunks.empty()) {
      (void)chunks.emplace_back();
    }
    if (fusion_size != 0 && chunks.back().bytes + data_bytes > fusion_size) {
      (void)chunks.emplace_back();
    }
    auto &chunk = chunks.back();
    chunk.bytes += data_bytes;
    chunk.size += data_size;
  }
  return chunk_map;
}

AbstractBasePtr InferImplFlattenConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  auto seq = dyn_cast<abstract::AbstractSequence>(args_spec_list[0]);
  if (seq == nullptr) {
    MS_LOG(EXCEPTION) << "The input for '" << primitive->name() << "' should be tuple or list, but got "
                      << args_spec_list[0]->type_name();
  }
  // Get fusion size from primitive attribute.
  const auto fusion_size_attr = primitive->GetAttr("fusion_size");
  const size_t fusion_size = static_cast<size_t>(fusion_size_attr != nullptr ? GetValue<int64_t>(fusion_size_attr) : 0);
  // Group inputs by data type and fusion size.
  auto chunk_map = GroupingAbstractTensors(seq->elements(), fusion_size, primitive->name());
  // Make result AbstractTuple according to the grouping result.
  AbstractBasePtrList tuple_element;
  for (auto &entry : chunk_map) {
    auto dtype = TypeIdToType(entry.first);
    for (auto &chunk : entry.second) {
      ShapeVector shape_vec{static_cast<int64_t>(chunk.size)};
      auto abs = std::make_shared<abstract::AbstractTensor>(dtype, shape_vec);
      (void)tuple_element.emplace_back(abs);
    }
  }
  return std::make_shared<abstract::AbstractTuple>(std::move(tuple_element));
}

AbstractBasePtr InferImplRange(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  const int kInputIndex0 = 0;
  const int kInputIndex1 = 1;
  const int kInputIndex2 = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(args_spec_list, kEqual, input_num, op_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, args_spec_list, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, args_spec_list, kInputIndex1);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, args_spec_list, kInputIndex2);
  // infer type must in before
  auto infer_type = RangeCheckAndInferType(primitive, args_spec_list);
  auto infer_shape = RangeCheckAndInferShape(primitive, args_spec_list);
  return std::make_shared<AbstractTensor>(infer_type, infer_shape);
}

AbstractBasePtr InferImplDynamicStitch(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  bool output_shape_unknow = false;
  auto prim_name = primitive->name();
  constexpr int64_t args_size = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(args_spec_list.size()), kEqual, args_size,
                                           prim_name);
  for (const auto &item : args_spec_list) {
    MS_EXCEPTION_IF_NULL(item);
  }

  // input0: indices
  auto input_tuple = args_spec_list[0]->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple);
  auto indices = input_tuple->elements();
  auto input_indice_size = input_tuple->size();
  int64_t first_dim_size = 0;
  for (size_t i = 0; i < input_indice_size; i++) {
    auto indicei = indices[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(indicei);
    auto valuei = indicei->BuildValue();
    MS_EXCEPTION_IF_NULL(valuei);
    if (!valuei->isa<tensor::Tensor>()) {
      output_shape_unknow = true;
      continue;
    }
    auto indicei_value = CheckAndConvertUtils::CheckTensorIntValue("indices", valuei, prim_name);
    auto indicei_max = std::max_element(indicei_value.begin(), indicei_value.end());
    first_dim_size = *indicei_max > first_dim_size ? *indicei_max : first_dim_size;
  }

  auto indices0 = indices[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(indices0);
  auto indices0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices0->BuildShape())[kShape];

  // input1: data
  auto input_tuple_1 = args_spec_list[1]->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple_1);
  auto data = input_tuple_1->elements();
  auto data0 = data[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(data0);
  auto data0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data0->BuildShape())[kShape];
  if (indices.size() != data.size()) {
    MS_LOG(EXCEPTION) << "The number of input[0] must be the same as input[0]!";
  }

  int64_t indices_total_size = 0;
  std::map<std::string, TypePtr> types;
  (void)types.emplace("data0", data0->BuildType());
  for (size_t i = 1; i < data.size(); ++i) {
    auto indicesi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices[i]->BuildShape())[kShape];
    auto datai_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data[i]->BuildShape())[kShape];
    if (indicesi_shape.size() > datai_shape.size()) {
      MS_LOG(EXCEPTION) << "The rank of indices[i] must be <= rank of data[i]!";
    }
    indices_total_size += SizeToLong(indicesi_shape.size());
  }
  std::set<TypePtr> valid_types = ops::common_valid_types;
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);

  ShapeVector out_shape;
  if (output_shape_unknow) {
    out_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    out_shape.push_back(first_dim_size + 1);
  }
  for (size_t i = indices0_shape.size(); i < data0_shape.size(); ++i) {
    out_shape.push_back(data0_shape[i]);
  }
  ShapeVector min_shape = out_shape;
  ShapeVector max_shape = out_shape;
  if (output_shape_unknow) {
    // delete after dynamic alloc is support
    const int64_t EXPAND_MAX = 10;
    min_shape = out_shape;
    max_shape = out_shape;
    min_shape[0] = 1;
    max_shape[0] = indices_total_size * EXPAND_MAX;
  }
  return std::make_shared<AbstractTensor>(infer_type,
                                          std::make_shared<abstract::Shape>(out_shape, min_shape, max_shape));
}

AbstractBasePtr InferImplTensorCopySlices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list) {
  auto &op_name = primitive->name();
  constexpr auto kTensorCopySlicesInputNum = 5;
  CheckArgsSize(op_name, args_spec_list, kTensorCopySlicesInputNum);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  return std::make_shared<AbstractTensor>(input->element(), input->shape());
}

AbstractBasePtr InferImplOCRRecognitionPreHandle(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  constexpr size_t size_expected = 5;
  constexpr int64_t universe_min_batch = 16;
  constexpr int64_t universe_max_batch = 256;
  constexpr int64_t image_h = 64;
  constexpr int64_t image_w = 512;
  constexpr int64_t images_min_batch = 4;
  constexpr int64_t images_max_batch = 256;
  constexpr int64_t images_channels = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  ValuePtr format_value = primitive->GetAttr("format");
  std::string format = GetValue<std::string>(format_value);

  ShapeVector universe_shp;
  ShapeVector universe_min_shp;
  ShapeVector universe_max_shp;

  (void)universe_shp.emplace_back(Shape::kShapeDimAny);
  (void)universe_min_shp.emplace_back(universe_min_batch);
  (void)universe_max_shp.emplace_back(universe_max_batch);

  auto universe_abstract =
    std::make_shared<AbstractTensor>(kInt32, std::make_shared<Shape>(universe_shp, universe_min_shp, universe_max_shp));

  ShapeVector r_shp = {Shape::kShapeDimAny, image_h, image_w};
  ShapeVector r_max_shp = {images_max_batch, image_h, image_w};
  ShapeVector r_min_shp = {images_min_batch, image_h, image_w};

  if (format == "NHWC") {
    (void)r_shp.emplace(r_shp.end(), images_channels);
    (void)r_max_shp.emplace(r_max_shp.end(), images_channels);
    (void)r_min_shp.emplace(r_min_shp.end(), images_channels);
  } else {
    (void)r_shp.emplace(r_shp.begin() + 1, images_channels);
    (void)r_max_shp.emplace(r_max_shp.begin() + 1, images_channels);
    (void)r_min_shp.emplace(r_min_shp.begin() + 1, images_channels);
  }

  auto r_batched_abstract =
    std::make_shared<AbstractTensor>(kUInt8, std::make_shared<Shape>(r_shp, r_min_shp, r_max_shp));

  AbstractBasePtrList elements = {r_batched_abstract, universe_abstract, universe_abstract, universe_abstract};
  return std::make_shared<AbstractTuple>(elements);
}
}  // namespace abstract
}  // namespace mindspore
