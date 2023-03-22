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
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/infer_functions.h"
#include "abstract/param_validator.h"
#include "abstract/utils.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace abstract {
namespace {
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
}  // namespace
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

AbstractBasePtr InferImplBroadcastShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
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
  ShapeVector max_shape;
  if (!is_input_dynamic) {
    max_shape = shape->shape();
  }

  auto ids = std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(ids_shape, max_shape));
  // Currently we choose the same data type as input for the idx.
  TypePtr ids_idx_type = kInt32;
  MS_EXCEPTION_IF_NULL(input->element());
  MS_EXCEPTION_IF_NULL(input->element()->GetTypeTrack());
  if (input->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
    ids_idx_type = kInt64;
  }
  ShapeVector idx_shape = shape->shape();
  auto ids_idx = std::make_shared<AbstractTensor>(ids_idx_type, idx_shape);
  // outputs: ids, ids_idx
  AbstractBasePtrList elements = {ids, ids_idx};
  return std::make_shared<AbstractTuple>(elements);
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
  ShapeVector max_shape;
  ShapeVector input_shape = shape->shape();
  if (!IsDynamic(input_shape)) {
    max_shape = input_shape;
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(ids_shape, max_shape));
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

  auto x_shape = x->shape()->shape();
  bool x_any_shape = IsDynamic(x_shape);
  bool ids_any_shape = IsDynamic(segment_ids_shape);
  bool op_is_dynamic = x_any_shape || ids_any_shape;
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
  }
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
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
  auto x_shape = x->shape()->shape();
  bool x_any_shape = IsDynamic(x_shape);
  bool ids_any_shape = IsDynamic(segment_ids_shape);
  bool op_is_dynamic = x_any_shape || ids_any_shape;
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
  }
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
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
  auto x_shape = x->shape()->shape();
  bool x_any_shape = IsDynamic(x_shape);
  bool ids_any_shape = IsDynamic(segment_ids_shape);
  bool op_is_dynamic = x_any_shape || ids_any_shape;
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
  }
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
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
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
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
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
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

  auto cache_idx = std::make_shared<AbstractTensor>(hash_map->element(), indices->shape());
  auto old_emb_idx = std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape));
  auto miss_emb_idx = std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape));
  auto swap_emb_idx = std::make_shared<AbstractTensor>(hash_map->element(), std::make_shared<Shape>(shape));

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

  AbstractTensorPtr ret = std::make_shared<AbstractTensor>(cache_table->element(), std::make_shared<Shape>(shape));
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

  auto filter_res = std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape));
  auto filter_idx = std::make_shared<AbstractTensor>(input_x->element(), std::make_shared<Shape>(shape));
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

AbstractBasePtr InferImplRealInnerDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
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
  ShapeVector shape;
  shape.insert(shape.end(), indices_shape.begin(), indices_shape.end());
  shape.insert(shape.end(), params_shape.begin() + 1, params_shape.end());

  AbstractTensorPtr ret = std::make_shared<AbstractTensor>(params->element(), std::make_shared<Shape>(shape));
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
  for (size_t i = 0; i < perm_vec.size(); i++) {
    auto idx = static_cast<size_t>(perm_vec[i]);
    result_shp.push_back(input_shp[idx]);
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(result_shp));
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

  AbstractBasePtrList output_list;
  for (int64_t i = 0; i < output_num_value; ++i) {
    auto output = input_x->Broaden();
    output->set_shape(std::make_shared<Shape>(output_shape));
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
  lengths_shape.push_back(maxlen_value);
  ShapePtr output_shape = std::make_shared<Shape>(lengths_shape);
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
  constexpr int64_t universe_max_batch = 256;
  constexpr int64_t image_h = 64;
  constexpr int64_t image_w = 512;
  constexpr int64_t images_max_batch = 256;
  constexpr int64_t images_channels = 3;
  CheckArgsSize(op_name, args_spec_list, size_expected);
  ValuePtr format_value = primitive->GetAttr("format");
  std::string format = GetValue<std::string>(format_value);

  ShapeVector universe_shp;
  ShapeVector universe_max_shp;

  (void)universe_shp.emplace_back(Shape::kShapeDimAny);
  (void)universe_max_shp.emplace_back(universe_max_batch);

  auto universe_abstract =
    std::make_shared<AbstractTensor>(kInt32, std::make_shared<Shape>(universe_shp, universe_max_shp));

  ShapeVector r_shp = {Shape::kShapeDimAny, image_h, image_w};
  ShapeVector r_max_shp = {images_max_batch, image_h, image_w};

  if (format == "NHWC") {
    (void)r_shp.emplace(r_shp.end(), images_channels);
    (void)r_max_shp.emplace(r_max_shp.end(), images_channels);
  } else {
    (void)r_shp.emplace(r_shp.begin() + 1, images_channels);
    (void)r_max_shp.emplace(r_max_shp.begin() + 1, images_channels);
  }

  auto r_batched_abstract = std::make_shared<AbstractTensor>(kUInt8, std::make_shared<Shape>(r_shp, r_max_shp));

  AbstractBasePtrList elements = {r_batched_abstract, universe_abstract, universe_abstract, universe_abstract};
  return std::make_shared<AbstractTuple>(elements);
}
}  // namespace abstract
}  // namespace mindspore
