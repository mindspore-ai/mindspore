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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "ops/dynamic_resize_nearest_neighbor.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DynamicResizeNearestNeighborInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto x_shape = x_shape_ptr->shape();
  const int64_t shape_size = 4;
  const int64_t size_size = 2;
  (void)CheckAndConvertUtils::CheckInteger("the dimension of input_x", SizeToLong(x_shape.size()), kEqual, shape_size,
                                           prim_name);
  auto size = input_args[1];
  MS_EXCEPTION_IF_NULL(size);
  auto size_v = size->BuildValue();
  MS_EXCEPTION_IF_NULL(size_v);
  std::vector<int64_t> size_value;
  std::vector<int64_t> output_shape;
  if (size->isa<abstract::AbstractTensor>()) {
    if (size_v->isa<tensor::Tensor>()) {
      size_value = CheckAndConvertUtils::CheckTensorIntValue("size", size_v, prim_name);
    } else {
      size_value.push_back(-1);
      size_value.push_back(-1);
    }
  } else if (size->isa<abstract::AbstractTuple>()) {
    size_value = CheckAndConvertUtils::CheckIntOrTupleInt("size", size_v, prim_name);
  } else if (size->isa<abstract::AbstractList>()) {
    size_value = CheckAndConvertUtils::CheckListInt("size", size_v, prim_name);
  }
  (void)CheckAndConvertUtils::CheckInteger("the dimension of size", SizeToLong(size_value.size()), kEqual, size_size,
                                           prim_name);
  output_shape.push_back(x_shape[0]);
  output_shape.push_back(x_shape[1]);
  output_shape.push_back(size_value[0]);
  output_shape.push_back(size_value[1]);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr DynamicResizeNearestNeighborInferType(const PrimitivePtr &prim,
                                              const std::vector<AbstractBasePtr> &input_args) {
  auto valid_types = common_valid_types;
  (void)valid_types.insert(kComplex128);
  (void)valid_types.insert(kComplex64);
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(DynamicResizeNearestNeighbor, BaseOperator);
AbstractBasePtr DynamicResizeNearestNeighborInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)),
                                           kEqual, input_num, prim_name);
  auto res = abstract::MakeAbstract(DynamicResizeNearestNeighborInferShape(primitive, input_args),
                                    DynamicResizeNearestNeighborInferType(primitive, input_args));
  return res;
}
REGISTER_PRIMITIVE_EVAL_IMPL(DynamicResizeNearestNeighbor, prim::kPrimDynamicResizeNearestNeighbor,
                             DynamicResizeNearestNeighborInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
