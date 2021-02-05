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

#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "nnacl/reshape_parameter.h"
#include "src/ops/reshape.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateReshapeParameter(const mindspore::lite::PrimitiveC *primitive) {
  ReshapeParameter *reshape_param = reinterpret_cast<ReshapeParameter *>(malloc(sizeof(ReshapeParameter)));
  if (reshape_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReshapeParameter failed.";
    return nullptr;
  }
  memset(reshape_param, 0, sizeof(ReshapeParameter));
  reshape_param->op_parameter_.type_ = primitive->Type();
  auto reshape_lite_primitive = (lite::Reshape *)primitive;
  auto shape = reshape_lite_primitive->GetShape();
  reshape_param->shape_dim_ = shape.size();
  int i = 0;
  for (auto iter = shape.begin(); iter != shape.end(); iter++) {
    reshape_param->shape_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(reshape_param);
}

Registry ReshapeParameterRegistry(schema::PrimitiveType_Reshape, PopulateReshapeParameter);

}  // namespace lite
}  // namespace mindspore
