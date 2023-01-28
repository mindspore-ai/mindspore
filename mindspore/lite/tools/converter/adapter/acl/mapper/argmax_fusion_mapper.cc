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

#include "tools/converter/adapter/acl/mapper/argmax_fusion_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNameInputNum = 2;
constexpr size_t kNumFlagThree = 3;
}  // namespace

STATUS ArgMaxFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  if (cnode->size() != kNameInputNum) {
    MS_LOG(ERROR) << "Input size of argmax must be " << kNameInputNum << " real size: " << cnode->size();
    return lite::RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::ArgMaxV2>();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->AddAttr("output_type", TypeIdToType(kNumberTypeInt32));
  dst_prim->SetAttrs(src_prim->attrs());
  // convert attr to parameter node
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  int status = AddIntAttrToInput(func_graph, cnode, dst_prim, ops::kAxis, true);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Add axis constant value to input failed.";
    return lite::RET_ERROR;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameArgMaxFusion, ArgMaxFusionMapper)
}  // namespace lite
}  // namespace mindspore
