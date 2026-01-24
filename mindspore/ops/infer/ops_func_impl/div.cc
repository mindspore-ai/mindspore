/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/div.h"
#include <set>
#include <map>
#include <limits>
#include <string>
#include <vector>
#include "utils/check_convert_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace ops {
namespace div_internal {
constexpr int kTypeLevelBool = 0;
constexpr int kTypeLevelInt = 1;
constexpr int kTypeLevelFloat = 2;
constexpr int kTypeLevelComplex = 3;

static inline bool IsBoolType(TypeId t) { return t == kNumberTypeBool; }
static inline bool IsComplexType(TypeId t) { return t == kNumberTypeComplex64 || t == kNumberTypeComplex128; }

static inline bool IsIntegralType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}

static inline bool IsFloatingType(TypeId t) {
  return t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeFloat64 || t == kNumberTypeBFloat16;
}

static inline int TypeToLevel(TypeId t) {
  if (IsBoolType(t)) {
    return kTypeLevelBool;
  } else if (IsIntegralType(t)) {
    return kTypeLevelInt;
  } else if (IsFloatingType(t)) {
    return kTypeLevelFloat;
  } else {
    // Do not silently treat unsupported types as complex.
    if (IsComplexType(t)) {
      return kTypeLevelComplex;
    }
    MS_EXCEPTION(TypeError) << "For Div, unsupported dtype: " << TypeIdToString(t);
  }
}
}  // namespace div_internal

ShapeArray DivFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &x_shape = input_infos[kIndex0]->GetShape();
  const auto &y_shape = input_infos[kIndex1]->GetShape();
  auto output_shape = CalBroadCastShape(x_shape, y_shape, primitive->name());
  return {output_shape};
}

std::vector<TypeId> DivFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &x_type = input_infos[kIndex0]->GetType();
  const auto &y_type = input_infos[kIndex1]->GetType();

  auto x_shape = input_infos[kIndex0]->GetShape();
  bool is_x_scalar = x_shape.empty();
  auto y_shape = input_infos[kIndex1]->GetShape();
  bool is_y_scalar = y_shape.empty();

  TypeId promote_type = x_type;
  if (is_x_scalar && !is_y_scalar) {
    promote_type = (div_internal::TypeToLevel(x_type) > div_internal::TypeToLevel(y_type)) ? x_type : y_type;
  } else if (!is_x_scalar && is_y_scalar) {
    promote_type = (div_internal::TypeToLevel(x_type) < div_internal::TypeToLevel(y_type)) ? y_type : x_type;
  } else if (x_type != y_type) {
    promote_type = PromoteType(x_type, y_type, primitive->name());
  }
  if (div_internal::IsIntegralType(promote_type) || div_internal::IsBoolType(promote_type)) {
    return {kNumberTypeFloat32};
  }
  return {promote_type};
}
}  // namespace ops
}  // namespace mindspore
