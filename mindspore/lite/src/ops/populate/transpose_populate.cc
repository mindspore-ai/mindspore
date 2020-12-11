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

#include "src/ops/transpose.h"
#include <memory>
#include "src/common/log_adapter.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/transpose.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateTransposeParameter(const mindspore::lite::PrimitiveC *primitive) {
  TransposeParameter *transpose_param = reinterpret_cast<TransposeParameter *>(malloc(sizeof(TransposeParameter)));
  if (transpose_param == nullptr) {
    MS_LOG(ERROR) << "malloc TransposeParameter failed.";
    return nullptr;
  }
  memset(transpose_param, 0, sizeof(TransposeParameter));
  auto param = reinterpret_cast<mindspore::lite::Transpose *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  transpose_param->op_parameter_.type_ = primitive->Type();
  auto perm_vector_ = param->GetPerm();
  int i = 0;
  for (auto iter = perm_vector_.begin(); iter != perm_vector_.end(); iter++) {
    transpose_param->perm_[i++] = *iter;
  }
  transpose_param->num_axes_ = i;
  return reinterpret_cast<OpParameter *>(transpose_param);
}

Registry TransposeParameterRegistry(schema::PrimitiveType_Transpose, PopulateTransposeParameter);

}  // namespace lite
}  // namespace mindspore
