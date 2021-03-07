/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/arg_min_max_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateArgMinParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto argmin_prim = primitive->value_as_ArgMin();
  ArgMinMaxParameter *arg_param = reinterpret_cast<ArgMinMaxParameter *>(malloc(sizeof(ArgMinMaxParameter)));
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "malloc ArgMinMaxParameter failed.";
    return nullptr;
  }
  memset(arg_param, 0, sizeof(ArgMinMaxParameter));
  arg_param->op_parameter_.type_ = schema::PrimitiveType_ArgMinFusion;

  arg_param->axis_ = argmin_prim->axis();
  arg_param->topk_ = argmin_prim->topK();
  arg_param->axis_type_ = argmin_prim->axisType();
  arg_param->out_value_ = argmin_prim->outMaxValue();
  arg_param->keep_dims_ = argmin_prim->keepDims();
  arg_param->get_max_ = false;
  return reinterpret_cast<OpParameter *>(arg_param);
}
}  // namespace

Registry g_argMinV0ParameterRegistry(schema::v0::PrimitiveType_ArgMin, PopulateArgMinParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
