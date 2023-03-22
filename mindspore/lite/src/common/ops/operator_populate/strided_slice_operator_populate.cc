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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/strided_slice_parameter.h"
#include "ops/strided_slice.h"
using mindspore::ops::kNameStridedSlice;
using mindspore::schema::PrimitiveType_StridedSlice;
namespace mindspore {
namespace lite {
OpParameter *PopulateStridedSliceOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<StridedSliceParameter *>(PopulateOpParameter<StridedSliceParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Make OpParameter ptr failed";
    return nullptr;
  }

  auto op = dynamic_cast<ops::StridedSlice *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to StridedSlice failed";
    free(param);
    return nullptr;
  }

  param->begins_mask_ = static_cast<int>(op->get_begin_mask());
  param->ends_mask_ = static_cast<int>(op->get_end_mask());
  param->ellipsisMask_ = static_cast<int>(op->get_ellipsis_mask());
  param->newAxisMask_ = static_cast<int>(op->get_new_axis_mask());
  param->shrinkAxisMask_ = static_cast<int>(op->get_shrink_axis_mask());

  if (param->begins_mask_ < C0NUM || param->ends_mask_ < C0NUM || param->ellipsisMask_ < C0NUM ||
      param->newAxisMask_ < C0NUM || param->shrinkAxisMask_ < C0NUM) {
    MS_LOG(ERROR) << "invalid StridedSliceParameter value";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameStridedSlice, PrimitiveType_StridedSlice, PopulateStridedSliceOpParameter)
}  // namespace lite
}  // namespace mindspore
