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
#include "nnacl/arg_min_max_parameter.h"
using mindspore::schema::PrimitiveType_ArgMinFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateArgMinParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_ArgMinFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<ArgMinMaxParameter *>(malloc(sizeof(ArgMinMaxParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ArgMinMaxParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ArgMinMaxParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->axis_ = value->axis();
  param->topk_ = value->top_k();
  param->out_value_ = value->out_max_value();
  param->keep_dims_ = value->keep_dims();
  param->get_max_ = false;
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_ArgMinFusion, PopulateArgMinParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
