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
#include "nnacl/crop_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateCropParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto crop_prim = primitive->value_as_Crop();
  auto param_offset = crop_prim->offsets();
  if (param_offset->size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "crop_param offset size(" << param_offset->size() << ") should <= " << COMM_SHAPE_SIZE;
    return nullptr;
  }
  CropParameter *crop_param = reinterpret_cast<CropParameter *>(malloc(sizeof(CropParameter)));
  if (crop_param == nullptr) {
    MS_LOG(ERROR) << "malloc CropParameter failed.";
    return nullptr;
  }
  memset(crop_param, 0, sizeof(CropParameter));
  crop_param->op_parameter_.type_ = primitive->value_type();
  crop_param->axis_ = crop_prim->axis();
  crop_param->offset_size_ = param_offset->size();
  for (size_t i = 0; i < param_offset->size(); ++i) {
    crop_param->offset_[i] = *(param_offset->begin() + i);
  }
  return reinterpret_cast<OpParameter *>(crop_param);
}
}  // namespace

Registry g_cropParameterRegistry(schema::PrimitiveType_Crop, PopulateCropParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
