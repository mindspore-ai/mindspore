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

#include "infer/ops_func_impl/ones_like_ext.h"
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_o.h"

namespace mindspore {
namespace ops {
std::vector<TypeId> OnesLikeExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto &dtype = input_infos[kInputIndex1];
  TypeId output_type;
  if (!dtype->IsNone()) {
    output_type = static_cast<TypeId>(dtype->GetScalarValueWithCheck<int64_t>());
  } else {
    output_type = input_infos[kInputIndex0]->GetType();
  }
  return {output_type};
}
}  // namespace ops
}  // namespace mindspore
