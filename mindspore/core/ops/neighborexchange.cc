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
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
abstract::TupleShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input_numbers", input_args.size(), kEqual, 1, prim_name);
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTuple>(prim_name, input_args, 0);
  auto recv_shapes = primitive->GetAttr(RecvShapes);
  MS_EXCEPTION_IF_NULL(recv_shapes);
  auto shapes_seq = recv_shapes->cast<ValueSequeuePtr>();
  MS_EXCEPTION_IF_NULL(shapes_seq);
  auto shapes_value = shapes_seq->value();
  abstract::BaseShapePtrList base_shape_list;
  for (auto &value : shapes_value) {
    auto each_shape_value = value->cast<ValueSequeuePtr>();
    MS_EXCEPTION_IF_NULL(each_shape_value);
    std::vector<int64_t> each_shape = GetValue<std::vector<int64_t>>(each_shape_value);
    BaseShapePtr base_shape = std::make_shared<abstract::Shape>(each_shape);
    MS_EXCEPTION_IF_NULL(base_shape);
    base_shape_list.push_back(base_shape);
  }
  return std::make_shared<abstract::TupleShape>(base_shape_list);
}

TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("NeighborExchange infer", SizeToLong(input_args.size()), kEqual, 1,
                                           prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto recv_shapes = primitive->GetAttr(RecvShapes);
  MS_EXCEPTION_IF_NULL(recv_shapes);
  auto shapes_seq = recv_shapes->cast<ValueSequeuePtr>();
  MS_EXCEPTION_IF_NULL(shapes_seq);
  auto shapes_value = shapes_seq->value();
  auto out_num = shapes_value.size();
  auto recv_type = primitive->GetAttr(RecvType)->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(recv_type);
  std::vector<TypePtr> type_vec(out_num, recv_type);
  return std::make_shared<Tuple>(type_vec);
}

AbstractBasePtr NeighborExchangeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  auto type = InferType(primitive, input_args);
  auto shape = InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(NeighborExchange, prim::kPrimNeighborExchange, NeighborExchangeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
