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

#include "src/ops/pad.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/pad_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulatePadParameter(const mindspore::lite::PrimitiveC *primitive) {
  PadParameter *pad_param = reinterpret_cast<PadParameter *>(malloc(sizeof(PadParameter)));
  if (pad_param == nullptr) {
    MS_LOG(ERROR) << "malloc PadParameter failed.";
    return nullptr;
  }
  memset(pad_param, 0, sizeof(PadParameter));
  pad_param->op_parameter_.type_ = primitive->Type();
  auto pad_node = reinterpret_cast<mindspore::lite::Pad *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  pad_param->pad_mode_ = pad_node->GetPaddingMode();
  pad_param->constant_value_ = pad_node->GetConstantValue();
  auto size = pad_node->GetPaddings().size();
  if (size > MAX_PAD_SIZE) {
    MS_LOG(ERROR) << "Invalid padding size: " << size;
    free(pad_param);
    return nullptr;
  }

  for (size_t i = 0; i < MAX_PAD_SIZE - size; ++i) {
    pad_param->paddings_[i] = 0;
  }
  for (size_t i = 0; i < size; i++) {
    pad_param->paddings_[MAX_PAD_SIZE - size + i] = pad_node->GetPaddings()[i];
  }
  pad_param->padding_length = MAX_PAD_SIZE;

  return reinterpret_cast<OpParameter *>(pad_param);
}
Registry PadParameterRegistry(schema::PrimitiveType_Pad, PopulatePadParameter);

}  // namespace lite
}  // namespace mindspore
