/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/roi_pooling_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateROIPoolingParameter(const void *prim) {
  ROIPoolingParameter *roi_param = reinterpret_cast<ROIPoolingParameter *>(malloc(sizeof(ROIPoolingParameter)));
  if (roi_param == nullptr) {
    MS_LOG(ERROR) << "malloc ROIPoolingParameter failed.";
    return nullptr;
  }

  memset(roi_param, 0, sizeof(ROIPoolingParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  roi_param->op_parameter_.type_ = primitive->value_type();
  auto roi_prim = primitive->value_as_ROIPooling();
  roi_param->pooledH_ = roi_prim->pooled_h();
  roi_param->pooledW_ = roi_prim->pooled_w();
  roi_param->scale_ = roi_prim->scale();
  return reinterpret_cast<OpParameter *>(roi_param);
}
}  // namespace

Registry g_ROIPoolingParameterRegistry(schema::PrimitiveType_ROIPooling, PopulateROIPoolingParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
