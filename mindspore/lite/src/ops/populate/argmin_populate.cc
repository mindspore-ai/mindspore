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

#include "src/ops/argmin.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/arg_min_max_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateArgMinParameter(const mindspore::lite::PrimitiveC *primitive) {
  ArgMinMaxParameter *arg_param = reinterpret_cast<ArgMinMaxParameter *>(malloc(sizeof(ArgMinMaxParameter)));
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArgMinMaxParameter failed.";
    return nullptr;
  }
  memset(arg_param, 0, sizeof(ArgMinMaxParameter));
  arg_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::ArgMin *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  arg_param->axis_ = param->GetAxis();
  arg_param->topk_ = param->GetTopK();
  arg_param->axis_type_ = param->GetAxisType();
  arg_param->out_value_ = param->GetOutMaxValue();
  arg_param->keep_dims_ = param->GetKeepDims();
  arg_param->get_max_ = false;
  return reinterpret_cast<OpParameter *>(arg_param);
}

Registry ArgMinParameterRegistry(schema::PrimitiveType_ArgMin, PopulateArgMinParameter);
}  // namespace lite
}  // namespace mindspore
