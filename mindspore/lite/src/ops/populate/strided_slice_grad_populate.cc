/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/strided_slice_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateStridedSliceGradParameter(const void *prim) {
  auto *strided_slice_param = reinterpret_cast<StridedSliceParameter *>(malloc(sizeof(StridedSliceParameter)));
  if (strided_slice_param == nullptr) {
    MS_LOG(ERROR) << "malloc StridedSliceParameter failed.";
    return nullptr;
  }
  memset(strided_slice_param, 0, sizeof(StridedSliceParameter));

  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_StridedSliceGrad();
  strided_slice_param->op_parameter_.type_ = primitive->value_type();

  strided_slice_param->begins_mask_ = value->begin_mask();
  strided_slice_param->ends_mask_ = value->end_mask();
  strided_slice_param->ellipsisMask_ = value->ellipsis_mask();
  strided_slice_param->newAxisMask_ = value->new_axis_mask();
  strided_slice_param->shrinkAxisMask_ = value->shrink_axis_mask();
  return reinterpret_cast<OpParameter *>(strided_slice_param);
}

Registry StridedSliceGradParameterRegistry(schema::PrimitiveType_StridedSliceGrad, PopulateStridedSliceGradParameter,
                                           SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
