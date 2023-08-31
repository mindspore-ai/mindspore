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

#include "tools/converter/adapter/acl/mapper/space_to_depth_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "ops/op_utils.h"
#include "src/common/log_util.h"
#include "mindspore/core/ops/op_name.h"
namespace mindspore {
namespace lite {
STATUS SpaceToDepthMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "value node is nullptr.";
    return RET_ERROR;
  }
  if (src_prim == nullptr) {
    MS_LOG(ERROR) << "src prim is nullptr.";
    return RET_ERROR;
  }

  auto fmk_attr_val = src_prim->GetAttr(ops::kFmkType);
  if (fmk_attr_val == nullptr) {
    MS_LOG(ERROR) << "fmk attr val is nullptr.";
    return RET_ERROR;
  }
  int64_t fmk_type = GetValue<int64_t>(fmk_attr_val);
  if (static_cast<converter::FmkType>(fmk_type) == converter::kFmkTypeOnnx) {
    // SpaceToDepth op : cann need attr of data_format
    src_prim->AddAttr("data_format", MakeValue("NCHW"));
  }
  return lite::RET_OK;
}
REGISTER_PRIMITIVE_MAPPER(kNameSpaceToDepth, SpaceToDepthMapper)
}  // namespace lite
}  // namespace mindspore
