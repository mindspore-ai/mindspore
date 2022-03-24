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

#include "tools/converter/adapter/acl/mapper/avgpool_fusion_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "include/registry/converter_context.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
STATUS AvgPoolFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  auto attr_val = src_prim->GetAttr(ops::kFmkType);
  int fmk_type = attr_val != nullptr ? GetValue<int>(attr_val) : converter::kFmkTypeTf;
  PrimitivePtr dst_prim = nullptr;
  CreateTargetPrim(src_prim, &dst_prim, fmk_type);
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  if (AdjustPoolAttr(fmk_type, kNameAvgPoolFusion, dst_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust pool attr failed.";
    return lite::RET_ERROR;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

void AvgPoolFusionMapper::CreateTargetPrim(const PrimitivePtr &src_prim, PrimitivePtr *dst_prim, int fmk_type) {
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "Target prim is nullptr.";
    return;
  }

  if (fmk_type == converter::kFmkTypeCaffe) {
    *dst_prim = std::make_shared<acl::Pooling>();
  } else if (fmk_type == converter::kFmkTypeOnnx) {
    ValuePtr val_ptr = src_prim->GetAttr(ops::kKernelSize);
    if (val_ptr == nullptr) {
      *dst_prim = std::make_shared<acl::GlobalAveragePool>();
    } else {
      *dst_prim = std::make_shared<acl::AvgPoolV2>();
    }
  } else {
    ops::AvgPool dst_node;
    *dst_prim = dst_node.GetPrim();
  }
}

REGISTER_PRIMITIVE_MAPPER(kNameAvgPoolFusion, AvgPoolFusionMapper)
}  // namespace lite
}  // namespace mindspore
