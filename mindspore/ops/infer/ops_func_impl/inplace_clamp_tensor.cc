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
#include "infer/ops_func_impl/inplace_clamp_tensor.h"
#include <utility>
#include <memory>
#include "infer/ops_func_impl/reduce_arithmetic.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

ShapeArray InplaceClampTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  if (input_infos[kInputIndex1]->IsNone() && input_infos[kInputIndex2]->IsNone()) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }

  const auto &input0 = input_infos[kIndex0];
  MS_EXCEPTION_IF_NULL(input0);
  auto x_shape = input0->GetShape();
  const auto &input1 = input_infos[kIndex1];
  MS_EXCEPTION_IF_NULL(input1);
  if (!input1->IsNone()) {
    auto min_shape = input1->GetShape();
    if (!IsBroadcastable(x_shape, min_shape)) {
      MS_EXCEPTION(ValueError) << "For Clamp, the shape of 'input' " << x_shape << " and the shape of 'min' "
                               << min_shape << " cannot broadcast.";
    }
  }

  const auto &input2 = input_infos[kIndex2];
  MS_EXCEPTION_IF_NULL(input2);
  if (!input2->IsNone()) {
    auto max_shape = input2->GetShape();
    if (!IsBroadcastable(x_shape, max_shape)) {
      MS_EXCEPTION(ValueError) << "For Clamp, the shape of 'input' " << x_shape << " and the shape of 'max' "
                               << max_shape << " cannot broadcast.";
    }
  }
  return {x_shape};
}

std::vector<TypeId> InplaceClampTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                          const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(input_infos[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_infos[kInputIndex1]);
  MS_EXCEPTION_IF_NULL(input_infos[kInputIndex2]);

  if (input_infos[kInputIndex1]->IsNone() && input_infos[kInputIndex2]->IsNone()) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }

  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
