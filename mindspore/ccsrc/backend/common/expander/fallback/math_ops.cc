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

#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace expander {
namespace {
const std::set<TypeId> kIntergralSet = {kNumberTypeBool, kNumberTypeUInt8, kNumberTypeInt8, kNumberTypeInt16,
                                        kNumberTypeInt32};
}  // namespace
REG_FALLBACK_BUILDER("AddExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  auto alpha_tensor = ib->Cast(ib->ScalarToTensor(alpha, x->dtype()), y->dtype());
  return {x + y * alpha_tensor};
});

REG_FALLBACK_BUILDER("SubExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto alpha = ib->GetInput(kIndex2);
  auto alpha_tensor = ib->Cast(ib->ScalarToTensor(alpha, x->dtype()), y->dtype());
  return {x - y * alpha_tensor};
});

REG_FALLBACK_BUILDER("MeanExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto dtype = ib->GetInput(kIndex3);

  auto dtype_type = dtype->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(dtype_type);
  // cppcheck-suppress *
  if (!dtype_type->isa<TypeNone>()) {
    auto dtype_opt = ops::GetScalarValue<int64_t>(dtype->BuildValue());
    MS_CHECK_VALUE(dtype_opt.has_value(), "For 'MeanExt', dtype must have valid value.");
    input = ib->Cast(input, TypeIdToType(static_cast<TypeId>(dtype_opt.value())));
  }

  auto axis_type = axis->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(axis_type);
  if (axis_type->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  }

  auto out = ib->Emit("ReduceMean", {input, axis, keep_dims});
  return {out};
});

REG_FALLBACK_BUILDER("SumExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto dtype = ib->GetInput(kIndex3);

  auto dtype_type = dtype->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(dtype_type);
  if (!dtype_type->isa<TypeNone>()) {
    auto dtype_opt = ops::GetScalarValue<int64_t>(dtype->BuildValue());
    MS_CHECK_VALUE(dtype_opt.has_value(), "For 'SumExt', dtype must have valid value.");
    input = ib->Cast(input, TypeIdToType(static_cast<TypeId>(dtype_opt.value())));
  } else {
    auto input_type = input->dtype()->type_id();
    if (kIntergralSet.find(input_type) != kIntergralSet.end()) {
      input = ib->Cast(input, kInt64);
    }
  }

  auto axis_type = axis->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(axis_type);
  if (axis_type->isa<TypeNone>()) {
    axis = ib->Value<std::vector<int64_t>>({});
  }

  auto out = ib->Emit("ReduceSum", {input, axis, keep_dims, ib->Value<bool>(false)});
  return {out};
});
}  // namespace expander
}  // namespace mindspore
