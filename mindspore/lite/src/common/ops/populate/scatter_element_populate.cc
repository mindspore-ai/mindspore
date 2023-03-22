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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/scatter_elements_parameter.h"
using mindspore::schema::PrimitiveType_ScatterElements;

namespace mindspore {
namespace lite {
OpParameter *PopulateScatterElementsParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  auto value = primitive->value_as_ScatterElements();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<ScatterElementsParameter *>(malloc(sizeof(ScatterElementsParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ScatterElements Parameter failed.";
    return nullptr;
  }
  (void)memset(param, 0, sizeof(ScatterElementsParameter));
  param->op_parameter_.type_ = primitive->value_type();
  param->axis_ = static_cast<int>(value->axis());
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_ScatterElements, PopulateScatterElementsParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
