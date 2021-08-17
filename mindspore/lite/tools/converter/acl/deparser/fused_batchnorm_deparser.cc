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

#include "tools/converter/acl/deparser/fused_batchnorm_deparser.h"
#include "tools/converter/acl/deparser/primitive_deparser_register.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
STATUS FusedBatchNormDeparser::Deparser(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  src_prim->AddAttr(ops::kIsTraining, MakeValue(false));
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_DEPARSER(kNameFusedBatchNorm, FusedBatchNormDeparser)
}  // namespace lite
}  // namespace mindspore
