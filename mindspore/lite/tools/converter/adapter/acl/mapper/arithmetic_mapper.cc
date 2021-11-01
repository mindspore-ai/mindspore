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

#include "tools/converter/adapter/acl/mapper/arithmetic_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"

namespace mindspore {
namespace lite {
STATUS AddFusionMapper::Mapper(const CNodePtr &cnode) {
  auto dst_prim = std::make_shared<ops::Add>();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "AddFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS DivFusionMapper::Mapper(const CNodePtr &cnode) {
  auto dst_prim = std::make_shared<ops::Div>();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "DivFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MulFusionMapper::Mapper(const CNodePtr &cnode) {
  auto dst_prim = std::make_shared<ops::Mul>();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "MulFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS PowFusionMapper::Mapper(const CNodePtr &cnode) {
  auto dst_prim = std::make_shared<ops::Pow>();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "PowFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS SubFusionMapper::Mapper(const CNodePtr &cnode) {
  auto dst_prim = std::make_shared<ops::Sub>();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "SubFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameAddFusion, AddFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameDivFusion, DivFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameMulFusion, MulFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNamePowFusion, PowFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameSubFusion, SubFusionMapper)
}  // namespace lite
}  // namespace mindspore
