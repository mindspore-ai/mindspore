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

#define USE_DEPRECATED_API
#include "tools/converter/adapter/acl/mapper/arithmetic_mapper.h"
#include <memory>
#include <map>
#include <string>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "src/common/log_util.h"
#include "ops/real_div.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
STATUS AddFusionMapper::Mapper(const CNodePtr &cnode) {
  ops::Add op_add;
  auto dst_prim = op_add.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "AddFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS DivFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  std::string original_name = "Div";
  auto name_ptr = src_prim->GetAttr(ops::kOriginalOpName);
  if (name_ptr != nullptr) {
    original_name = GetValue<std::string>(name_ptr);
    original_name = original_name.empty() ? "Div" : original_name;
  }
  std::map<std::string, BaseOperatorPtr> kDivTypeMap = {{"Div", std::make_shared<ops::Div>()},
                                                        {"RealDiv", std::make_shared<ops::RealDiv>()}};
  PrimitivePtr dst_prim = nullptr;
  if (kDivTypeMap.find(original_name) != kDivTypeMap.end()) {
    auto dst_op = kDivTypeMap.at(original_name);
    MS_CHECK_TRUE_MSG(dst_op != nullptr, lite::RET_ERROR, "Div op is nullptr.");
    dst_prim = dst_op->GetPrim();
  }
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return RET_OK;
}

STATUS MulFusionMapper::Mapper(const CNodePtr &cnode) {
  ops::Mul mul_op;
  auto dst_prim = mul_op.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "MulFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS PowFusionMapper::Mapper(const CNodePtr &cnode) {
  ops::Pow pow_op;
  auto dst_prim = pow_op.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "PowFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS SubFusionMapper::Mapper(const CNodePtr &cnode) {
  ops::Sub sub_op;
  auto dst_prim = sub_op.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "SubFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ExpFusionMapper::Mapper(const CNodePtr &cnode) {
  ops::Exp op_add;
  auto dst_prim = op_add.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "ExpFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameAddFusion, AddFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameDivFusion, DivFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameMulFusion, MulFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNamePowFusion, PowFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameSubFusion, SubFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameExpFusion, ExpFusionMapper)
}  // namespace lite
}  // namespace mindspore
