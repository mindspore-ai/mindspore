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
#include "ops/grad/strided_slice_grad.h"
using mindspore::ops::kNameStridedSliceGrad;
using mindspore::schema::PrimitiveType_StridedSliceGrad;

namespace mindspore {
namespace lite {
OpParameter *PopulateStridedSliceGradOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<StridedSliceParameter *>(PopulateOpParameter<StridedSliceParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new StridedSliceParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::StridedSliceGrad *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not StridedSliceGrad.";
    return nullptr;
  }

  param->begins_mask_ = op->get_begin_mask();
  param->ends_mask_ = op->get_end_mask();
  param->ellipsisMask_ = op->get_ellipsis_mask();
  param->newAxisMask_ = op->get_new_axis_mask();
  param->shrinkAxisMask_ = op->get_shrink_axis_mask();
  if (param->begins_mask_ < C0NUM || param->ends_mask_ < C0NUM || param->ellipsisMask_ < C0NUM ||
      param->newAxisMask_ < C0NUM || param->shrinkAxisMask_ < C0NUM) {
    MS_LOG(ERROR) << "invalid StridedSliceGradParameter value";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameStridedSliceGrad, PrimitiveType_StridedSliceGrad, PopulateStridedSliceGradOpParameter);
}  // namespace lite
}  // namespace mindspore
