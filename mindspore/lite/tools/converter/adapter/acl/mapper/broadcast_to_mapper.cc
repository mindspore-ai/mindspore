/**
 * Copyright 2022~2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/broadcast_to_mapper.h"
#include <string>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "ops/expand.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameInputNums = 3;
}

STATUS BroadcastToMapper::Mapper(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
  int input_num = cnode->inputs().size();
  if (input_num == kNameInputNums) {
    ops::Expand expand;
    auto dst_prim = expand.GetPrim();
    if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
      MS_LOG(ERROR) << "Expand mapper failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameBroadcastTo, BroadcastToMapper)
}  // namespace lite
}  // namespace mindspore
