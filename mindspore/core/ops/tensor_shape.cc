/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/tensor_shape.h"
#include <vector>
#include <memory>
#include "ops/dynamic_shape.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
namespace ops {
abstract::AbstractBasePtr TensorShapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, 1, primitive->name());
  auto input = input_args[0]->cast<abstract::AbstractTensorPtr>();
  if (input == nullptr) {
    MS_EXCEPTION(TypeError) << "For the input['input_x'] of primitive[" << primitive->name() << "] should be a Tensor"
                            << ", but got " << input_args[0]->BuildType()->ToString() << ".";
  }
  MS_EXCEPTION_IF_NULL(input->shape());
  auto shape = input->shape()->shape();
  bool has_dyn_shape =
    std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim == abstract::Shape::SHP_ANY; });
  ShapeVector tensor_shp({static_cast<int64_t>(shape.size())});
  if (has_dyn_shape) {
    auto elem = std::make_shared<abstract::AbstractScalar>(std::make_shared<AnyValue>(), std::make_shared<Int>(64));
    auto min_value = MakeValue(input->shape()->min_shape());
    auto max_value = MakeValue(input->shape()->max_shape());
    auto abs_tensor = std::make_shared<abstract::AbstractTensor>(elem, std::make_shared<abstract::Shape>(tensor_shp));
    abs_tensor->set_value_range(min_value, max_value);
    return abs_tensor;
  }
  auto shp_buf_size = sizeof(int64_t) * shape.size();
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, tensor_shp, shape.data(), shp_buf_size);

  return tensor->ToAbstract();
}
REGISTER_PRIMITIVE_EVAL_IMPL(TensorShape, prim::kPrimTensorShape, TensorShapeInfer, nullptr, true);
REGISTER_PRIMITIVE_EVAL_IMPL(DynamicShape, prim::kPrimDynamicShape, TensorShapeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
