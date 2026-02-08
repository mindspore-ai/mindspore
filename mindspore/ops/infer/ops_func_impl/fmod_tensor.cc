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

#include "infer/ops_func_impl/fmod_tensor.h"
#include <set>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace ops {
ShapeArray FmodTensorFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto input_shape = input_infos[kIndex0]->GetShape();
  auto other_shape = input_infos[kIndex1]->GetShape();
  return {CalBroadCastShape(input_shape, other_shape, primitive->name())};
}

std::vector<TypeId> FmodTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kIndex0]->GetType();
  auto other_type = input_infos[kIndex1]->GetType();

  return {PromoteType(input_type, other_type, primitive->name())};
}
}  // namespace ops
}  // namespace mindspore
