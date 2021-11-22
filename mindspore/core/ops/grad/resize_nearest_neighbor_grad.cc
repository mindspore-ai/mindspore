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
#include "ops/grad/resize_nearest_neighbor_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto grad_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto grad_shape = grad_shape_ptr->shape();
  auto size_ptr = input_args[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(size_ptr);
  std::vector<int64_t> size_v = GetValue<std::vector<int64_t>>(size_ptr);
  std::vector<int64_t> ret_shape;
  ret_shape.push_back(grad_shape[0]);
  ret_shape.push_back(grad_shape[1]);
  ret_shape.insert(ret_shape.end(), size_v.begin(), size_v.end());
  if (grad_shape_ptr->IsDynamic()) {
    auto grad_min_shape = grad_shape_ptr->min_shape();
    std::vector<int64_t> ret_min_shape;
    ret_min_shape.push_back(grad_min_shape[0]);
    ret_min_shape.push_back(grad_min_shape[1]);
    ret_min_shape.insert(ret_min_shape.end(), size_v.begin(), size_v.end());
    auto grad_max_shape = grad_shape_ptr->max_shape();
    std::vector<int64_t> ret_max_shape;
    ret_max_shape.push_back(grad_max_shape[0]);
    ret_max_shape.push_back(grad_max_shape[1]);
    ret_max_shape.insert(ret_max_shape.end(), size_v.begin(), size_v.end());
    return std::make_shared<abstract::Shape>(ret_shape, ret_min_shape, ret_max_shape);
  }
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  return input_args[0]->BuildType();
}
}  // namespace
AbstractBasePtr ResizeNearestNeighborGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)),
                                           kEqual, input_num, prim_name);
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(ResizeNearestNeighborGrad, prim::kPrimResizeNearestNeighborGrad,
                             ResizeNearestNeighborGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
