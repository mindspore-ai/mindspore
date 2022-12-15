/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/split_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
STATUS SplitMapper::Mapper(const CNodePtr &cnode) {
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(prim);

  auto split_num_val = prim->GetAttr(ops::kOutputNum);
  CHECK_NULL_RETURN(split_num_val);
  prim->AddAttr("num_split", split_num_val);
  int status = AddIntAttrToInput(func_graph, cnode, prim, ops::kAxis, false);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Add axis constant value to input failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameSplit, SplitMapper)
}  // namespace lite
}  // namespace mindspore
