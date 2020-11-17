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

#include "src/ops/unsqueeze.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/unsqueeze_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateUnsqueezeParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto unsqueeze_attr =
    reinterpret_cast<mindspore::lite::Unsqueeze *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  UnsqueezeParameter *unsqueeze_param = reinterpret_cast<UnsqueezeParameter *>(malloc(sizeof(UnsqueezeParameter)));
  if (unsqueeze_param == nullptr) {
    MS_LOG(ERROR) << "malloc UnsqueezeParameter failed.";
    return nullptr;
  }
  memset(unsqueeze_param, 0, sizeof(UnsqueezeParameter));
  unsqueeze_param->op_parameter_.type_ = primitive->Type();
  auto flatAxis = unsqueeze_attr->GetAxis();
  unsqueeze_param->num_dim_ = flatAxis.size();
  int i = 0;
  for (auto iter = flatAxis.begin(); iter != flatAxis.end(); iter++) {
    unsqueeze_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(unsqueeze_param);
}
Registry UnsqueezeParameterRegistry(schema::PrimitiveType_Unsqueeze, PopulateUnsqueezeParameter);

}  // namespace lite
}  // namespace mindspore
