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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "ops/grad/resize_nearest_neighbor_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kResizeNearestNeighborGradInputNum = 2;
constexpr auto kResizeIdx = 1;

abstract::ShapePtr ResizeNearestNeighborGradInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  if (input_args.size() != kResizeNearestNeighborGradInputNum) {
    MS_LOG(EXCEPTION) << "ResizeNearsetNeighborGrad's input num should be " << kResizeNearestNeighborGradInputNum
                      << ", but got " << input_args.size();
  }
  auto grad_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto grad_shape = grad_shape_ptr->shape();
  auto size_ptr = input_args[kResizeIdx]->BuildValue();
  MS_EXCEPTION_IF_NULL(size_ptr);

  std::vector<int64_t> size_v;
  if (size_ptr->isa<tensor::Tensor>()) {
    auto size_tensor = size_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(size_tensor);
    size_t data_size = size_tensor->DataSize();
    auto tensor_data = reinterpret_cast<int64_t *>(size_tensor->data_c());
    MS_EXCEPTION_IF_NULL(tensor_data);
    for (size_t i = 0; i < data_size; ++i) {
      size_v.push_back(static_cast<int64_t>(*tensor_data));
      ++tensor_data;
    }
  } else if (size_ptr->isa<ValueTuple>()) {
    std::vector<ValuePtr> size_vec = size_ptr->cast<ValueTuplePtr>()->value();
    (void)std::transform(size_vec.begin(), size_vec.end(), std::back_inserter(size_v),
                         [](const ValuePtr e) { return GetValue<int64_t>(e); });
  } else if (size_ptr->isa<AnyValue>()) {
    size_v.push_back(-1);
    size_v.push_back(-1);
  } else {
    size_v = GetValue<std::vector<int64_t>>(size_ptr);
  }

  std::vector<int64_t> ret_shape;
  ret_shape.push_back(grad_shape[0]);
  ret_shape.push_back(grad_shape[1]);
  ret_shape.insert(ret_shape.end(), size_v.begin(), size_v.end());
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr ResizeNearestNeighborGradInferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  return input_args[0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(ResizeNearestNeighborGrad, BaseOperator);
AbstractBasePtr ResizeNearestNeighborGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)),
                                           kEqual, input_num, prim_name);
  return abstract::MakeAbstract(ResizeNearestNeighborGradInferShape(primitive, input_args),
                                ResizeNearestNeighborGradInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(ResizeNearestNeighborGrad, prim::kPrimResizeNearestNeighborGrad,
                             ResizeNearestNeighborGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
