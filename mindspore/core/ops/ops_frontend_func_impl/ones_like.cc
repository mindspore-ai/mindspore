/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
class OnesLikeFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    constexpr size_t input_index = 0;
    auto type = input_args[input_index]->GetType();
    auto shape = input_args[input_index]->GetShape()->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    auto shape_vec = shape->shape();
    if (IsDynamic(shape_vec)) {
      return nullptr;
    }
    auto tensor_ptr = TensorConstructUtils::CreateOnesTensor(type, shape_vec, true);
    if (tensor_ptr == nullptr) {
      return nullptr;
    }
    return tensor_ptr;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("OnesLike", OnesLikeFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
