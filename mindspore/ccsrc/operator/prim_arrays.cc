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

#include "pipeline/static_analysis/prim.h"
#include "operator/ops.h"
#include "pipeline/static_analysis/utils.h"
#include "operator/cc_implementations.h"
#include "pipeline/static_analysis/param_validator.h"

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
  std::vector<int> shp_x;
  (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(shp_x),
                       [](const ValuePtr &e) -> int { return GetValue<int>(e); });

  auto value_tuple_y = ys->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple_y);
  auto shp_tuple_y = value_tuple_y->value();
  std::vector<int> shp_y;
  (void)std::transform(std::begin(shp_tuple_y), std::end(shp_tuple_y), std::back_inserter(shp_y),
                       [](const ValuePtr &e) -> int { return GetValue<int>(e); });

  std::vector<int> res = prim::BroadcastShape_(shp_x, shp_y);
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

AbstractBasePtr InferImplShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tensor.
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, args_spec_list, 1);
  AbstractTensorPtr arg = CheckArg<AbstractTensor>(op_name, args_spec_list, 0);
  MS_LOG(DEBUG) << "InferImplShape:" << arg->ToString();

  AbstractBasePtrList values;
  auto shp = arg->shape();
  for (int entry : shp->shape()) {
    auto entry_v = MakeValue(entry);
    values.push_back(std::make_shared<AbstractScalar>(entry_v, entry_v->type()));
  }
  return std::make_shared<AbstractTuple>(values);
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

  std::vector<int> mul_shp;
  auto value_tuple_mul = mul_shp_value->cast<ValueTuplePtr>();
  auto mul_shp_data = value_tuple_mul->value();
  (void)std::transform(std::begin(mul_shp_data), std::end(mul_shp_data), std::back_inserter(mul_shp),
                       [](const ValuePtr &e) -> int { return GetValue<int>(e); });
  if (input_shape->shape().size() != mul_shp_data.size()) {
    MS_LOG(EXCEPTION) << "Tile requires input and multiples size equal, while the input size is "
                      << input_shape->shape().size() << ", value size is: " << mul_shp_data.size() << ".";
  }

  std::vector<int> result_shp;
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
}  // namespace abstract
}  // namespace mindspore
