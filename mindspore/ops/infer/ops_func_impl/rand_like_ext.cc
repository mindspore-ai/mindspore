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

#include "infer/ops_func_impl/rand_like_ext.h"
#include <memory>
#include <string>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {
void CheckRandRange(const InferInfoPtr &from, const InferInfoPtr &to, const std::string &name, bool output_bool);

ShapeArray RandLikeExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

TypeIdList RandLikeExtFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto prim_name = primitive->name();
  auto &dtype = input_infos[dtype_idx_];
  TypeId output_type;
  if (!dtype->IsNone()) {
    output_type = static_cast<TypeId>(dtype->GetScalarValueWithCheck<int64_t>());
  } else {
    output_type = input_infos[kIndex0]->GetType();
  }
  if (prim_name == kNameRandIntLike) {
    CheckRandRange(input_infos[kInputIndex1], input_infos[kInputIndex2], prim_name, (output_type == kNumberTypeBool));
  }
  CheckAndConvertUtils::CheckTypeIdValid("dtype", output_type, common_mint_valid_type_ids_with_bool, primitive->name());
  return {output_type};
}
}  // namespace ops
}  // namespace mindspore
