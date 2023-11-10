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

#include <memory>
#include <limits>
#include "abstract/dshape.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "util/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/randperm_v2.h"

namespace mindspore::ops {
bool CheckForOverflow(TypeId tpyeId, int64_t n) {
  constexpr int64_t max_float16 = 65504;
  int64_t max = 0;

  switch (tpyeId) {
    case kNumberTypeUInt8:
      max = static_cast<int64_t>(std::numeric_limits<uint8_t>::max()) + 1;
      break;

    case kNumberTypeInt8:
      max = static_cast<int64_t>(std::numeric_limits<int8_t>::max()) + 1;
      break;

    case kNumberTypeInt16:
      max = static_cast<int64_t>(std::numeric_limits<int16_t>::max()) + 1;
      break;

    case kNumberTypeInt32:
      max = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
      break;

    case kNumberTypeInt64:
      max = std::numeric_limits<int64_t>::max();
      break;

    case kNumberTypeFloat16:
      max = max_float16 + 1;
      break;

    // float32 and float64 do not overflow
    default:
      max = std::numeric_limits<int64_t>::max();
      break;
  }
  return n > max;
}

BaseShapePtr RandpermV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto n_opt = GetScalarValue<int64_t>(input_args[kInputIndex0]->GetValue());
  auto seed_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  auto offset_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  auto type_opt = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());

  if (!n_opt.has_value() || !seed_opt.has_value() || !offset_opt.has_value() || !type_opt.has_value()) {
    ShapeVector output_shape = {abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(output_shape);
  }

  int64_t n = n_opt.value();
  if (n <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the input 'n' must be greater than 0, but got data: " << n
                             << ".";
  }
  if (CheckForOverflow(static_cast<TypeId>(type_opt.value()), n)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the value of 'n' is " << n
                             << ", it's too lagrge for input 'dtype'. ";
  }

  if (seed_opt.value() < 0 && seed_opt.value() != -1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the input 'seed' must be greater than 0 or equal to 0 or -1, but got data: "
                             << seed_opt.value() << ".";
  }

  if (offset_opt.value() < 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the input 'offset' must be greater than 0 or equal to 0, but got data: "
                             << offset_opt.value() << ".";
  }

  ShapeVector out_shape = {n};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr RandpermV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype = GetValue<int64_t>(input_args[kInputIndex3]->GetValue());
  return mindspore::TypeIdToType(static_cast<TypeId>(dtype));
}
}  // namespace mindspore::ops
