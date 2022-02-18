/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "src/common/log_adapter.h"
#include "src/tensor.h"
using mindspore::schema::PrimitiveType_Custom;

namespace mindspore {
namespace lite {
OpParameter *PopulateCustomParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Custom();
  if (value == nullptr) {
    MS_LOG(ERROR) << "the value is nullptr.";
    return nullptr;
  }
  MS_CHECK_TRUE_RET(value->type() != nullptr, nullptr);
  std::string type = value->type()->c_str();
  if (type == "ShapeFusion") {
    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc ShapeParameter failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(OpParameter));
    param->type_ = PrimType_Inner_ShapeFusion;
    return reinterpret_cast<OpParameter *>(param);
  } else {
    MS_LOG(ERROR) << "Unsupported custom type: " << type;
  }
  return nullptr;
}

REG_POPULATE(PrimType_Custom, PopulateCustomParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
