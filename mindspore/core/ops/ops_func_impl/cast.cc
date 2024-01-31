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
#include "ops/ops_func_impl/cast.h"
#include <utility>
#include <memory>
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr CastFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr CastFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTypeValid("x", input_args[kIndex0]->GetType(),
                                             common_valid_types_with_complex_and_bool, prim_name);
  constexpr int64_t kCastInputNumWithDtype = 2;
  if (input_args.size() == kCastInputNumWithDtype) {
    auto dtype_ptr = GetScalarValue<int64_t>(input_args[kIndex1]->GetValue());
    MS_CHECK_VALUE(dtype_ptr.has_value(), primitive->name() + " error: dtype input should has valid value.");
    auto type = TypeIdToType(static_cast<TypeId>(dtype_ptr.value()));
    return std::make_shared<TensorType>(type);
  } else {
    auto dst_type = primitive->GetAttr(kDstType);
    MS_EXCEPTION_IF_NULL(dst_type);
    return dst_type->cast<TypePtr>();
  }
}
}  // namespace ops
}  // namespace mindspore
