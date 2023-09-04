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

#include <vector>
#include "tools/converter/adapter/acl/mapper/im2col_mapper.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "ops/im2col.h"

namespace mindspore {
namespace lite {
constexpr size_t kDimension2D = 2;
STATUS Im2ColMapper::Mapper(const CNodePtr &cnode) {
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(prim);

  auto pads = prim->GetAttr("pads");
  CHECK_NULL_RETURN(pads);
  auto pads_vec = GetValue<std::vector<int64_t>>(pads);
  ops::Im2Col im2col;
  PrimitivePtr dst_prim = im2col.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "Im2Col mapper failed.";
    return RET_ERROR;
  }
  // acl: pads is 4D, meaning (top, bottom, left, right)
  if (pads_vec.size() == kDimension2D) {
    std::vector<int64_t> pads_mapped = {pads_vec[0], pads_vec[0], pads_vec[1], pads_vec[1]};
    dst_prim->AddAttr("pads", MakeValue(pads_mapped));
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameIm2Col, Im2ColMapper)
}  // namespace lite
}  // namespace mindspore
