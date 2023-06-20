/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/neighborexchange.h"
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/other_ops.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kNeighborExchangeRecvShapes = "recv_shapes";
constexpr auto kNeighborExchangeRecvType = "recv_type";

inline std::string GetShapeStr(const std::vector<int64_t> &shape) {
  std::string shape_str = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == 0) {
      shape_str += std::to_string(shape[i]);
    } else {
      shape_str += "," + std::to_string(shape[i]);
    }
  }
  return shape_str + "]";
}

void CheckAttr(const PrimitivePtr &primitive, const std::string &shape_attr_name,
               const std::string &rank_ids_attr_name) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // size of send/recv_rank_ids equal to size of send/recv_shapes
  ValuePtrList attr_shapes;
  try {
    auto attr = primitive->GetAttr(shape_attr_name);
    if (attr->cast<ValueTuplePtr>() == nullptr) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', attr '" << shape_attr_name
                              << "' is necessary, but missing it.";
    }
    attr_shapes = GetValue<ValuePtrList>(attr);
  } catch (const std::exception &) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', attr '" << shape_attr_name
                            << "' must be a tuple(list, list, ...).";
  }
  if (!attr_shapes.empty()) {
    auto ele = attr_shapes[0]->cast<ValueSequencePtr>();
    if (ele == nullptr) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "',  first element of attr " << shape_attr_name
                              << " is necessary, but missing it.";
    }
  }
  std::vector<int64_t> attr_rank_ids;
  try {
    auto attr = primitive->GetAttr(rank_ids_attr_name);
    if (attr->cast<ValueTuplePtr>() != nullptr) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', attr '" << shape_attr_name
                              << "' is necessary, but missing it";
    }
    attr_rank_ids = GetValue<std::vector<int64_t>>(attr);
  } catch (const std::exception &) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', attr '" << rank_ids_attr_name
                            << "' must be a list[int, int, ...].";
  }
  if (attr_shapes.size() != attr_rank_ids.size()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', attr '" << shape_attr_name
                             << "' size must be equal to attr  '" << rank_ids_attr_name << "' size, but got attr '"
                             << shape_attr_name << "' size: " << attr_shapes.size() << ", attr '" << rank_ids_attr_name
                             << "' size: " << attr_rank_ids.size();
  }
}

void NeighborExchangeCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr auto kSendShapes = "send_shapes";
  constexpr auto kRecvRankIds = "recv_rank_ids";
  constexpr auto kSendRankIds = "send_rank_ids";
  constexpr auto kGroup = "group";
  CheckAttr(primitive, kNeighborExchangeRecvShapes, kRecvRankIds);
  CheckAttr(primitive, kSendShapes, kSendRankIds);
  // check recv type
  auto recv_type_attr = primitive->GetAttr(kNeighborExchangeRecvType);
  MS_EXCEPTION_IF_NULL(recv_type_attr);
  if (!recv_type_attr->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', attr '" << kNeighborExchangeRecvType
                            << "' must be a mindspore data type, but got: " << recv_type_attr << ".";
  }
  // check group
  auto group_attr = primitive->GetAttr(kGroup);
  try {
    MS_EXCEPTION_IF_NULL(group_attr);
    (void)GetValue<std::string>(group_attr);
  } catch (const std::exception &) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', attr '" << kGroup
                            << "' must be a str, but got: " << group_attr << ".";
  }
  // check empty input
  auto send_rank_ids = GetValue<std::vector<int64_t>>(primitive->GetAttr(kSendRankIds));
  const int64_t input_num = 0;
  if (send_rank_ids.empty()) {
    (void)CheckAndConvertUtils::CheckInteger("input_numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    return;
  }
  // check input shape & attr send shape
  const int64_t input_num_ = 1;
  (void)CheckAndConvertUtils::CheckInteger("input_numbers", SizeToLong(input_args.size()), kEqual, input_num_,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTuple>(prim_name, input_args, input_num);
  auto abstract_tuple = input_args[0]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  auto abstract_element = abstract_tuple->elements();
  auto send_shapes = GetValue<ValuePtrList>(primitive->GetAttr(kSendShapes));
  if (abstract_element.size() != send_shapes.size()) {
    MS_EXCEPTION(ArgumentError) << "For '" << prim_name << "', input tuple size must be equal to attr '" << kSendShapes
                                << "' size, but got input tuple size: " << abstract_element.size() << ", attr '"
                                << kSendShapes << "' size: " << send_shapes.size() << ".";
  }
  for (size_t i = 0; i < abstract_element.size(); ++i) {
    // get attr shape
    MS_EXCEPTION_IF_NULL(send_shapes[i]);
    auto send_shape_value = send_shapes[i]->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(send_shape_value);
    std::vector<int64_t> send_shape = GetValue<std::vector<int64_t>>(send_shape_value);
    // get input tensor shape
    MS_EXCEPTION_IF_NULL(abstract_element[i]);
    auto arg_base_shape = abstract_element[i]->BuildShape();
    MS_EXCEPTION_IF_NULL(arg_base_shape);
    auto shape = arg_base_shape->cast<abstract::ShapePtr>();
    if (shape == nullptr) {
      MS_EXCEPTION(ArgumentError) << "For '" << prim_name << "', input[" << i << "] must be a tensor.";
    }
    // comp two shape
    auto shape_vec = shape->shape();
    if (shape_vec != send_shape) {
      MS_EXCEPTION(ArgumentError) << "For '" << prim_name << "', input[" << i
                                  << "] shape must be equal to attr shape, but got input[" << i
                                  << "] shape: " << GetShapeStr(shape_vec)
                                  << ", attr shape : " << GetShapeStr(send_shape) << ".";
    }
  }
}

abstract::BaseShapePtr NeighborExchangeInferShape(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto recv_shapes = primitive->GetAttr(kNeighborExchangeRecvShapes);
  MS_EXCEPTION_IF_NULL(recv_shapes);
  auto shapes_seq = recv_shapes->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(shapes_seq);
  auto shapes_value = shapes_seq->value();
  abstract::BaseShapePtrList base_shape_list;
  for (auto &value : shapes_value) {
    auto each_shape_value = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(each_shape_value);
    std::vector<int64_t> each_shape = GetValue<std::vector<int64_t>>(each_shape_value);
    BaseShapePtr base_shape = std::make_shared<abstract::Shape>(each_shape);
    MS_EXCEPTION_IF_NULL(base_shape);
    base_shape_list.push_back(base_shape);
  }
  if (base_shape_list.empty()) {
    return std::make_shared<abstract::Shape>();
  }
  return std::make_shared<abstract::TupleShape>(base_shape_list);
}

TypePtr NeighborExchangeInferType(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto recv_shapes = primitive->GetAttr(kNeighborExchangeRecvShapes);
  MS_EXCEPTION_IF_NULL(recv_shapes);
  auto shapes_seq = recv_shapes->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(shapes_seq);
  auto shapes_value = shapes_seq->value();
  auto out_num = shapes_value.size();
  auto recv_type = primitive->GetAttr(kNeighborExchangeRecvType)->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(recv_type);
  std::vector<TypePtr> type_vec(out_num, recv_type);
  if (type_vec.empty()) {
    return std::make_shared<TypeNone>();
  }
  return std::make_shared<Tuple>(type_vec);
}
}  // namespace

MIND_API_OPERATOR_IMPL(NeighborExchange, BaseOperator);
AbstractBasePtr NeighborExchangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  NeighborExchangeCheck(primitive, input_args);
  auto type = NeighborExchangeInferType(primitive);
  auto shape = NeighborExchangeInferShape(primitive);
  return abstract::MakeAbstract(shape, type);
}

// AG means auto generated
class MIND_API AGNeighborExchangeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) const override {
    return NeighborExchangeInferShape(primitive);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) const override {
    return NeighborExchangeInferType(primitive);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NeighborExchangeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NeighborExchange, prim::kPrimNeighborExchange, AGNeighborExchangeInfer, false);
}  // namespace ops
}  // namespace mindspore
