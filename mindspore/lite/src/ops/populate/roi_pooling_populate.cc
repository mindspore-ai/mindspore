/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/roi_pooling.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/roi_pooling_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateROIPoolingParameter(const mindspore::lite::PrimitiveC *primitive) {
  const auto param =
    reinterpret_cast<mindspore::lite::ROIPooling *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  ROIPoolingParameter *roi_pooling_param = reinterpret_cast<ROIPoolingParameter *>(malloc(sizeof(ROIPoolingParameter)));
  if (roi_pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc ROIPoolingParameter failed.";
    return nullptr;
  }
  memset(roi_pooling_param, 0, sizeof(ROIPoolingParameter));
  roi_pooling_param->op_parameter_.type_ = primitive->Type();
  roi_pooling_param->pooledH_ = param->GetPooledH();
  roi_pooling_param->pooledW_ = param->GetPooledW();
  roi_pooling_param->scale_ = param->GetScale();
  return reinterpret_cast<OpParameter *>(roi_pooling_param);
}

Registry ROIPoolingParameterRegistry(schema::PrimitiveType_ROIPooling, PopulateROIPoolingParameter);

}  // namespace lite
}  // namespace mindspore
