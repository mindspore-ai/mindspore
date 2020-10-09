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
                       [](const ValuePtr &e) -> int { return GetValue<int>(e); });

  auto value_tuple_y = ys->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_y);
  auto shp_tuple_y = value_tuple_y->value();
  ShapeVector shp_y;
  (void)std::transform(std::begin(shp_tuple_y), std::end(shp_tuple_y), std::back_inserter(shp_y),
                       [](const ValuePtr &e) -> int { return GetValue<int>(e); });

  ShapeVector res = BroadcastShape(shp_x, shp_y);
  if (res.empty()) {
    MS_LOG(EXCEPTION) << "BroadcastShape fail: " << args_spec_list[0]->ToString() << ","
                      << args_spec_list[1]->ToString();
  }

  AbstractBasePtrList elems;
  (void)std::transform(res.begin(), res.end(), std::back_inserter(elems), [](int n) -> AbstractBasePtr {
    return std::make_shared<AbstractScalar>(std::make_shared<Int32Imm>(n), kInt32);
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
                       [](const ValuePtr &e) -> int { return GetValue<int>(e); });
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
  int rank_base = SizeToInt(tensor_base->shape()->shape().size());

  ValuePtr axis = primitive->GetAttr("axis");
  // Axis value should be in [-(rank_base + 1), rank_base).
  int axis_value = CheckAxis(op_name, axis, -(rank_base + 1), rank_base);
  // If axis is negative, add offset(rank_base + 1) to turn it to positive.
  axis_value = GetPositiveAxis(axis_value, IntToSize(rank_base + 1));

  for (size_t i = 1; i < tuple_len; ++i) {
    AbstractTensorPtr tensor = CheckArg<AbstractTensor>(op_name, arg->elements(), i);
    (void)CheckDtypeSame(op_name, tensor_base, tensor);
    (void)CheckShapeSame(op_name, tensor_base, tensor);
  }

  primitive->set_attr("N", MakeValue(SizeToInt(tuple_len)));
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
  auto ids_idx = std::make_shared<AbstractTensor>(std::make_shared<Int>(32), shape->shape());
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
  auto x = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(x->shape());
  auto x_shape = x->shape()->shape();

  auto segment_ids = CheckArg<AbstractTensor>(op_name, args_spec_list, 1);
  MS_EXCEPTION_IF_NULL(segment_ids);
  MS_EXCEPTION_IF_NULL(segment_ids->shape());
  auto segment_ids_shape = segment_ids->shape()->shape();
  auto num_segments = CheckArg<AbstractTensor>(op_name, args_spec_list, 2);

  std::vector<int> shape;
  auto num_segments_value = num_segments->BuildValue();
  MS_EXCEPTION_IF_NULL(num_segments_value);
  if (!num_segments_value->isa<tensor::Tensor>()) {
    MS_LOG(WARNING) << num_segments_value << "evaluator num_segments_value should be tensor, but got "
                    << num_segments_value->type_name();
    shape.emplace_back(-1);
  } else {
    auto num_segments_tensor = num_segments_value->cast<tensor::TensorPtr>();
    int value = *(static_cast<int *>(num_segments_tensor->data_c()));
    MS_LOG(INFO) << "Infer UnsortedSegmentSum output shape:" << value;
    shape.emplace_back(value);
  }

  shape.insert(shape.end(), x_shape.begin() + segment_ids_shape.size(), x_shape.end());

  AbstractTensorPtr ret = std::make_shared<AbstractTensor>(x->element(), std::make_shared<Shape>(shape));
  return ret;
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
  std::vector<int> x_shape = x->shape()->shape();
  std::vector<int> y_shape = y->shape()->shape();
  std::vector<int> out_shape = BroadcastShape(x_shape, y_shape);
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
  std::vector<int> x_shape = x->shape()->shape();
  std::vector<int> y_shape = y->shape()->shape();
  std::vector<int> out_shape = BroadcastShape(x_shape, y_shape);
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

  int axis_val = 0;
  // 3rd input is a Tensor when GatherV2 is a dynamic shape operator
  if (args_spec_list[2]->isa<AbstractTensor>()) {
    auto axis = args_spec_list[2]->cast<AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis);
    auto axis_value_ptr = axis->BuildValue();
    MS_EXCEPTION_IF_NULL(axis_value_ptr);
    auto axis_tensor = axis_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_tensor);
    axis_val = *static_cast<int *>(axis_tensor->data_c());
  } else if (args_spec_list[2]->isa<AbstractScalar>()) {
    auto axis = args_spec_list[2]->cast<AbstractScalarPtr>();
    axis_val = GetValue<int>(axis->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << args_spec_list[2]->type_name();
  }

  auto params_shp = params->shape()->shape();
  auto indices_shp = indices->shape()->shape();

  auto params_rank = static_cast<int>(params_shp.size());
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
      elements.push_back(std::make_shared<AbstractScalar>(std::make_shared<AnyValue>(), std::make_shared<Int>(32)));
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

  bool has_dyn_shape = std::any_of(shape.begin(), shape.end(), [](int dim) { return dim == Shape::SHP_ANY; });
  std::vector<int> tensor_shp({static_cast<int>(shape.size())});
  if (has_dyn_shape) {
    auto elem = std::make_shared<AbstractScalar>(std::make_shared<AnyValue>(), std::make_shared<Int>(32));
    return std::make_shared<AbstractTensor>(elem, std::make_shared<Shape>(tensor_shp));
  }
  auto shp_buf_size = sizeof(int) * shape.size();
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, tensor_shp, shape.data(), shp_buf_size);

  return tensor->ToAbstract();
}
}  // namespace abstract
}  // namespace mindspore
