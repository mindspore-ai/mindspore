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

#include "ops/ops_func_impl/replication_pad_3d.h"
#include <map>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReplicationPad3DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  constexpr size_t kReplicationPad3DDim = 3;
  return PadInferShapeBase(primitive, input_args, kReplicationPad3DDim);
}

TypePtr ReplicationPad3DFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto padding_type = input_args[kInputIndex1]->GetType();
  if (!padding_type->isa<Tuple>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', type of 'padding' should be tuple of int, but got"
                            << padding_type;
  }
  return CheckAndConvertUtils::CheckSubClass("input_x", input_args[kInputIndex0]->GetType(), {kTensorType}, prim_name);
}
}  // namespace ops
}  // namespace mindspore
