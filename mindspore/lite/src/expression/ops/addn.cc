/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/expression/ops/addn.h"

namespace mindspore {
namespace lite {
AddN::AddN(int dummy) {
  auto op_param = calloc(1, sizeof(OpParameter));
  if (op_param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate parameter ";
    return;
  }
  set_name(UniqueName("addN"));
  SetOpParam(op_param);
  set_primitive(schema::PrimitiveType_AddN);
}

int AddN::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::AddNT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
