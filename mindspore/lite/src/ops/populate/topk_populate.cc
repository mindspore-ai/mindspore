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

#include "src/ops/topk.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/topk_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateTopKParameter(const mindspore::lite::PrimitiveC *primitive) {
  TopkParameter *topk_param = reinterpret_cast<TopkParameter *>(malloc(sizeof(TopkParameter)));
  if (topk_param == nullptr) {
    MS_LOG(ERROR) << "malloc TopkParameter failed.";
    return nullptr;
  }
  memset(topk_param, 0, sizeof(TopkParameter));
  topk_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::TopK *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  topk_param->k_ = param->GetK();
  topk_param->sorted_ = param->GetSorted();
  return reinterpret_cast<OpParameter *>(topk_param);
}
Registry TopKParameterRegistry(schema::PrimitiveType_TopK, PopulateTopKParameter);

}  // namespace lite
}  // namespace mindspore
