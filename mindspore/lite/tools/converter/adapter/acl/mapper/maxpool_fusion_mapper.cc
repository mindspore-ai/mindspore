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

#include "tools/converter/adapter/acl/mapper/maxpool_fusion_mapper.h"
#include <memory>
#include <string>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "include/registry/converter_context.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kCommonAttrValueNum = 2;
constexpr auto kMaxKernelSize = 20;
constexpr auto kMaxKernelSize_H_Mul_W = 255;
}  // namespace
STATUS MaxPoolFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  auto attr_val = src_prim->GetAttr(ops::kFmkType);
  int fmk_type = attr_val != nullptr ? GetValue<int>(attr_val) : converter::kFmkTypeTf;
  PrimitivePtr dst_prim = nullptr;
  std::string max_pool_name;
  if (fmk_type == converter::kFmkTypeCaffe) {
    dst_prim = std::make_shared<acl::Pooling>();
    max_pool_name = acl::kNamePooling;
  } else if (fmk_type == converter::kFmkTypeOnnx && IsKernelSizeValid(src_prim)) {
    dst_prim = std::make_shared<acl::MaxPoolV3>();
    max_pool_name = acl::kNameMaxPoolV3;
  } else {
    ops::MaxPool max_pool_op;
    dst_prim = max_pool_op.GetPrim();
    max_pool_name = ops::kNameMaxPool;
  }
  MS_CHECK_TRUE_MSG(dst_prim != nullptr, lite::RET_ERROR, "Get primitive by fmk type failed.");
  dst_prim->SetAttrs(src_prim->attrs());
  if (AdjustPoolAttr(fmk_type, max_pool_name, dst_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust pool attr failed.";
    return lite::RET_ERROR;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

bool MaxPoolFusionMapper::IsKernelSizeValid(const PrimitivePtr &prim) const {
  auto attr_val = prim->GetAttr(ops::kKernelSize);
  if (attr_val == nullptr) {
    return true;
  }
  auto kernel_value = opt::CastToInt(attr_val);
  if (kernel_value.size() != kCommonAttrValueNum) {
    MS_LOG(ERROR) << " Kernel size value num must be two, but size is " << kernel_value.size();
    return false;
  }
  int64_t kernel_h = kernel_value[0];
  int64_t kernel_w = kernel_value[1];
  if ((kernel_h <= kMaxKernelSize && kernel_w <= kMaxKernelSize) || (kernel_h * kernel_w <= kMaxKernelSize_H_Mul_W)) {
    return true;
  }
  return false;
}

REGISTER_PRIMITIVE_MAPPER(kNameMaxPoolFusion, MaxPoolFusionMapper)
}  // namespace lite
}  // namespace mindspore
