/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <iterator>
#include "abstract/infer_functions.h"
#include "abstract/utils.h"
#include "abstract/param_validator.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
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
  if (!a_shp->shape().empty()) {
    MS_LOG(EXCEPTION) << "array_to_scalar requires zero size shape.";
  }
  return arg->element();
}

AbstractBasePtr InferImplBroadCastShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list) {
  // Inputs: two tuples.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  auto xs = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  auto ys = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);

  auto value_tuple_x = xs->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_x);
  auto shp_tuple_x = value_tuple_x->value();
  ShapeVector shp_x;
  (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(shp_x),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });

  auto value_tuple_y = ys->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_y);
  auto shp_tuple_y = value_tuple_y->value();
  ShapeVector shp_y;
  (void)std::transform(std::begin(shp_tuple_y), std::end(shp_tuple_y), std::back_inserter(shp_y),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });

  ShapeVector res = BroadcastShape(shp_x, shp_y);
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

AbstractBasePtr InferImplTile(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor and a tuple.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  auto arg = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto multiples = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);

  ShapePtr input_shape = arg->shape();
  (void)CheckTensorDType(arg, {kInt16, kFloat16, kInt32, kFloat32}, "Input 0 of Tile should be %s");

  auto mul_shp_value = multiples->BuildValue();
  if (mul_shp_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "shape's data field can't be anything: " << args_spec_list[1]->ToString();
  }

  ShapeVector mul_shp;
  auto value_tuple_mul = mul_shp_value->cast<ValueTuplePtr>();
  auto mul_shp_data = value_tuple_mul->value();
  (void)std::transform(std::begin(mul_shp_data), std::end(mul_shp_data), std::back_inserter(mul_shp),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  if (input_shape->shape().size() != mul_shp_data.size()) {
    MS_LOG(EXCEPTION) << "Tile requires input and multiples size equal, while the input size is "
                      << input_shape->shape().size() << ", value size is: " << mul_shp_data.size() << ".";
  }

  ShapeVector result_shp;
  for (size_t i = 0; i < mul_shp_data.size(); ++i) {
    result_shp.push_back(input_shape->shape()[i] * mul_shp[i]);
  }
  return std::make_shared<AbstractTensor>(arg->element(), std::make_shared<Shape>(result_shp));
}

AbstractBasePtr InferImplPack(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple of tensor.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  auto arg = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  if (arg->elements().empty()) {
    MS_LOG(EXCEPTION) << "Arg elements is empty.";
  }

  size_t tuple_len = arg->elements().size();
  AbstractTensorPtr tensor_base = CheckArg<AbstractTensor>(op_name, arg->elements(), 0);
  int64_t rank_base = SizeToLong(tensor_base->shape()->shape().size());

  ValuePtr axis = primitive->GetAttr("axis");
  // Axis value should be in [-(rank_base + 1), rank_base).
  int64_t axis_value = CheckAxis(op_name, axis, -(rank_base + 1), rank_base);
  // If axis is negative, add offset(rank_base + 1) to turn it to positive.
  axis_value = GetPositiveAxis(axis_value, LongToSize(rank_base + 1));

  for (size_t i = 1; i < tuple_len; ++i) {
    AbstractTensorPtr tensor = CheckArg<AbstractTensor>(op_name, arg->elements(), i);
    (void)CheckDtypeSame(op_name, tensor_base, tensor);
    (void)CheckShapeSame(op_name, tensor_base, tensor);
  }

  primitive->set_attr("N", MakeValue(SizeToLong(tuple_len)));
  primitive->set_attr("T", tensor_base->element()->BuildType());

  AbstractTensorPtr ret = dyn_cast<AbstractTensor>(tensor_base->Broaden());
  MS_EXCEPTION_IF_NULL(ret);
  auto shape = ret->shape()->shape();
  (void)shape.insert(shape.begin() + axis_value, tuple_len);
  ret->set_shape(std::make_shared<Shape>(shape));
  return ret;
}

AbstractBasePtr InferImplUnique(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);

  auto shape = input->shape();
  if (shape->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
  }
  ShapeVector ids_shape = {Shape::SHP_ANY};
  ShapeVector min_shape = {1};
  ShapeVector max_shape = shape->shape();
  auto ids =
    std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(ids_shape, min_shape, max_shape));
  // Currently we choose the same data type as input for the idx.
  TypePtr ids_idx_type = kInt32;
  MS_EXCEPTION_IF_NULL(input->element());
  MS_EXCEPTION_IF_NULL(input->element()->GetTypeTrack());
  if (input->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
    ids_idx_type = kInt64;
  }
  auto ids_idx = std::make_shared<AbstractTensor>(ids_idx_type, shape->shape());
  // outputs: ids, ids_idx
  AbstractBasePtrList elements = {ids, ids_idx};
  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplUniqueGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  // inputs: a 1-d Tensor
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTuplePtr dout = CheckArg<AbstractTuple>(op_name, args_spec_list, 0);
  CheckArgsSize(op_name + " dout", dout->elements(), 2);
  auto ids = CheckArg<AbstractTensor>(op_name, dout->elements(), 0);
  auto ids_idx = CheckArg<AbstractTensor>(op_name, dout->elements(), 1);
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
  CheckArgsSize(op_name, args_spec_list, 3);
  // input x
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto x_shape = x->shape()->shape();
  // segment_ids
  auto segment_ids = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(segment_ids);
  MS_EXCEPTION_IF_NULL(segment_ids->shape());
  auto segment_ids_shape = segment_ids->shape()->shape();
  // checks on Tensors 0 and 1 types
  (void)CheckTensorDType(x, {kFloat32, kInt32}, "Input 0 (x) for SequenceMask should be %s");
  (void)CheckTensorDType(segment_ids, {kInt32, kInt64}, "Input 1 (segment_ids) for SequenceMask should be %s");
  ShapeVector shape;
  ShapeVector max_shape;
  ShapeVector min_shape;
  int64_t num_segments_value;
  if (args_spec_list[2]->isa<AbstractTensor>()) {  // Num segments is Tensor
    auto num_segments = args_spec_list[2]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments);
    auto num_segments_value_ptr = num_segments->BuildValue();
    MS_EXCEPTION_IF_NULL(num_segments_value_ptr);
    auto num_segments_tensor = num_segments_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments_tensor);
    num_segments_value = *static_cast<int64_t *>(num_segments_tensor->data_c());
    shape.emplace_back(num_segments_value);
  } else if (args_spec_list[2]->isa<AbstractScalar>()) {  // Num segments is Scalar
    auto num_segments = CheckArg<AbstractScalar>(op_name, args_spec_list, 2);
    num_segments_value = GetValue<int64_t>(num_segments->BuildValue());
    shape.emplace_back(num_segments_value);
  } else {
    MS_LOG(EXCEPTION) << "num_segments incorrect type in UnsortedSegmentSum";
  }
  shape.insert(shape.end(), x_shape.begin() + segment_ids_shape.size(), x_shape.end());
  // calc max shape
  if (!x->shape()->max_shape().empty()) {  // copy max shape from x if present
    std::copy(x->shape()->max_shape().begin(), x->shape()->max_shape().end(), std::back_inserter(max_shape));
  } else {  // copy x shape directly if not present
    std::copy(x->shape()->shape().begin(), x->shape()->shape().end(), std::back_inserter(max_shape));
  }
  // calc min shape
  min_shape.push_back(segment_ids_shape.size());
  std::copy(x->shape()->shape().begin() + segment_ids_shape.size(), x->shape()->shape().end(),
            back_inserter(min_shape));
  // return shape, min shape, max shape
  return std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape, min_shape, max_shape));
}

AbstractBasePtr InferImplScatterAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  return std::make_shared<AbstractTensor>(x->element(), x->shape());
}

AbstractBasePtr InferImplScatterUpdate(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 3);
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  return std::make_shared<AbstractTensor>(x->element(), x->shape());
}

AbstractBasePtr InferImplDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list) {
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
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
  CheckArgsSize(op_name, args_spec_list, 2);
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
  CheckArgsSize(op_name, args_spec_list, 3);
  AbstractTensorPtr params = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  AbstractTensorPtr indices = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);

  int64_t axis_val = 0;
  // 3rd input is a Tensor when GatherV2 is a dynamic shape operator
  if (args_spec_list[2]->isa<AbstractTensor>()) {
    auto axis = args_spec_list[2]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis);
    auto axis_value_ptr = axis->BuildValue();
    MS_EXCEPTION_IF_NULL(axis_value_ptr);
    auto axis_tensor = axis_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_tensor);
    axis_val = *static_cast<int64_t *>(axis_tensor->data_c());
  } else if (args_spec_list[2]->isa<AbstractScalar>()) {
    auto axis = args_spec_list[2]->cast<AbstractScalarPtr>();
    axis_val = GetValue<int64_t>(axis->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << args_spec_list[2]->type_name();
  }

  auto params_shp = params->shape()->shape();
  auto indices_shp = indices->shape()->shape();

  auto params_rank = static_cast<int64_t>(params_shp.size());
  if (axis_val < 0) {
    axis_val += params_rank;
  }

  auto calc_shape = [axis_val, &params_shp](const ShapeVector &inp_vec) -> ShapeVector {
    ShapeVector out_vec;
    std::copy(params_shp.begin(), params_shp.begin() + axis_val, std::back_inserter(out_vec));
    copy(inp_vec.begin(), inp_vec.end(), std::back_inserter(out_vec));
    copy(params_shp.begin() + axis_val + 1, params_shp.end(), std::back_inserter(out_vec));
    return out_vec;
  };

  ShapeVector out_shape = calc_shape(indices_shp);
  if (!indices->shape()->min_shape().empty() && !indices->shape()->max_shape().empty()) {
    ShapeVector min_shape = calc_shape(indices->shape()->min_shape());
    ShapeVector max_shape = calc_shape(indices->shape()->max_shape());
    return std::make_shared<AbstractTensor>(params->element(),
                                            std::make_shared<Shape>(out_shape, min_shape, max_shape));
  }

  return std::make_shared<AbstractTensor>(params->element(), std::make_shared<Shape>(out_shape));
}

AbstractBasePtr InferImplShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto shape = input->shape()->shape();

  AbstractBasePtrList elements;
  for (const auto &dim : shape) {
    if (dim == Shape::SHP_ANY) {
      elements.push_back(std::make_shared<AbstractScalar>(std::make_shared<AnyValue>(), std::make_shared<Int>(64)));
    } else {
      elements.push_back(std::make_shared<AbstractScalar>(dim));
    }
  }

  return std::make_shared<AbstractTuple>(elements);
}

AbstractBasePtr InferImplDynamicShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto shape = input->shape()->shape();

  bool has_dyn_shape = std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim == Shape::SHP_ANY; });
  ShapeVector tensor_shp({static_cast<int64_t>(shape.size())});
  if (has_dyn_shape) {
    auto elem = std::make_shared<AbstractScalar>(std::make_shared<AnyValue>(), std::make_shared<Int>(64));
    return std::make_shared<AbstractTensor>(elem, std::make_shared<Shape>(tensor_shp));
  }
  auto shp_buf_size = sizeof(int64_t) * shape.size();
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, tensor_shp, shape.data(), shp_buf_size);

  return tensor->ToAbstract();
}

AbstractBasePtr InferImplZerosLike(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  CheckArgsSize(primitive->name(), args_spec_list, 1);
  return args_spec_list[0]->Broaden();
}

AbstractBasePtr InferImplTranspose(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto perm = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  auto input_shp = input->shape()->shape();
  auto perm_val = perm->BuildValue();
  if (perm_val->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "Perm can't be anything: " << args_spec_list[1]->ToString();
  }
  auto perm_val_data = perm_val->cast<ValueTuplePtr>()->value();
  ShapeVector perm_vec;
  (void)std::transform(std::begin(perm_val_data), std::end(perm_val_data), std::back_inserter(perm_vec),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  ShapeVector result_shp;
  std::set<size_t> indices;
  for (size_t i = 0; i < perm_vec.size(); i++) {
    size_t idx = static_cast<size_t>(perm_vec[i]);
    if (indices.find(idx) != indices.end()) {
      MS_LOG(EXCEPTION) << "Perm values must be unique";
    }
    if (idx >= perm_vec.size()) {
      MS_LOG(EXCEPTION) << "One value in perm is " << idx << ", not in range [0, " << perm_vec.size() << ")";
    }
    result_shp.push_back(input_shp[idx]);
    indices.insert(idx);
  }
  ShapeVector max_shp;
  ShapeVector min_shp;
  if (input->shape()->max_shape().size() == input_shp.size() &&
      input->shape()->min_shape().size() == input_shp.size()) {
    for (size_t i = 0; i < perm_vec.size(); i++) {
      size_t idx = static_cast<size_t>(perm_vec[i]);
      max_shp.push_back(input->shape()->max_shape()[idx]);
      min_shp.push_back(input->shape()->min_shape()[idx]);
    }
    return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(result_shp, min_shp, max_shp));
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(result_shp));
}

AbstractBasePtr InferImplReshape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list) {
  const std::string &op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 2);
  AbstractTensorPtr input = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  auto reshape = CheckArg<AbstractTuple>(op_name, args_spec_list, 1);
  auto input_shp = input->shape()->shape();
  auto reshape_val = reshape->BuildValue();
  if (reshape_val->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "Input_shape can't be anything: " << args_spec_list[1]->ToString();
  }
  auto reshape_val_data = reshape_val->cast<ValueTuplePtr>()->value();
  ShapeVector reshape_vec;
  (void)std::transform(std::begin(reshape_val_data), std::end(reshape_val_data), std::back_inserter(reshape_vec),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  ShapeVector result_shp;
  auto input_prod = input_shp[0];
  int64_t dim_prod = 1;
  size_t neg_idx = 0;
  for (size_t i = 1; i < input_shp.size(); i++) {
    input_prod *= input_shp[i];
  }
  auto num_neg_one = count(std::begin(reshape_vec), std::end(reshape_vec), -1);
  if (num_neg_one > 1) {
    MS_LOG(EXCEPTION) << "The shape can only has one -1 at most, but " << num_neg_one;
  }
  for (size_t i = 0; i < reshape_vec.size(); i++) {
    if (reshape_vec[i] == -1) {
      neg_idx = i;
      result_shp.push_back(-1);
    } else {
      dim_prod *= reshape_vec[i];
      result_shp.push_back(reshape_vec[i]);
    }
  }
  if (dim_prod < 0 || input_prod % dim_prod != 0) {
    MS_LOG(EXCEPTION) << "The input_x shape product is " << input_prod << ", input_shape shape product is " << dim_prod
                      << ", and this value should be > 0 and should divide product of input_x.";
  }
  if (num_neg_one == 1) {
    int64_t val = static_cast<int64_t>(input_prod) / dim_prod;
    dim_prod *= val;
    result_shp[neg_idx] = val;
  }
  if (dim_prod != input_prod) {
    MS_LOG(EXCEPTION)
      << "The product of input_x shape should be equal to product of input_shape shape, but input_x shape is "
      << input_prod << ", product of input_shape shape is " << dim_prod;
  }
  return std::make_shared<AbstractTensor>(input->element(), std::make_shared<Shape>(result_shp));
}
}  // namespace abstract
}  // namespace mindspore
