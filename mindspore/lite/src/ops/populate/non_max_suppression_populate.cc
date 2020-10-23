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

#include "src/ops/non_max_suppression.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/non_max_suppression_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateNonMaxSuppressionParameter(const mindspore::lite::PrimitiveC *primitive) {
  NMSParameter *param = reinterpret_cast<NMSParameter *>(malloc(sizeof(NMSParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc param failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(NMSParameter));
  param->op_parameter_.type_ = primitive->Type();
  auto prim =
    reinterpret_cast<mindspore::lite::NonMaxSuppression *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  param->center_point_box_ = prim->GetCenterPointBox();
  return reinterpret_cast<OpParameter *>(param);
}
Registry NonMaxSuppressionParameterRegistry(schema::PrimitiveType_NonMaxSuppression,
                                            PopulateNonMaxSuppressionParameter);

}  // namespace lite
}  // namespace mindspore
