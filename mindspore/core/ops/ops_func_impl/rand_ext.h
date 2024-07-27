/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_RAND_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_RAND_H_

#include <set>
#include <vector>
#include <memory>
#include "ops/ops_func_impl/ones.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops/base_operator.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
class MIND_API RandExtFuncImpl : public OnesFuncImpl {
 public:
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto prim_name = primitive->name();
    // check
    auto dtype_type = input_args[kIndex3]->GetType();
    if (dtype_type->isa<TypeNone>()) {
      return kFloat32;
    }
    auto dtype_ptr = input_args[kIndex3]->GetValue();
    if (!dtype_ptr->isa<Int64Imm>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name
                              << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype_ptr->ToString()
                              << ".";
    }
    auto val = GetValue<int64_t>(dtype_ptr);
    auto infer_type = TypeIdToType(static_cast<TypeId>(val));
    CheckAndConvertUtils::CheckTypeValid("dtype", infer_type, {kFloat16, kFloat32, kFloat64, kBFloat16},
                                         primitive->name());
    return std::make_shared<TensorType>(infer_type);
  }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_RAND_H_
