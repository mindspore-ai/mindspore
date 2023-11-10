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
#include "ops/ops_func_impl/addcdiv.h"
#include <vector>
#include <memory>
#include <string>
#include "ops/op_name.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "ir/primitive.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr AddcdivFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto output_shape = input_shape_ptr->GetShapeVector();
  std::vector<std::string> input_names = {"input", "tensor1", "tensor2", "value"};
  if (MS_UNLIKELY(input_args.size() != input_names.size())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the number of inputs should be "
                             << input_names.size() << ", but got " << input_args.size();
  }
  for (size_t i = 1; i < input_args.size(); ++i) {
    auto input_shape = input_args[i]->GetShape()->GetShapeVector();
    output_shape = CalBroadCastShape(output_shape, input_shape, primitive->name(), input_names[i - 1], input_names[i]);
  }
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr AddcdivFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
