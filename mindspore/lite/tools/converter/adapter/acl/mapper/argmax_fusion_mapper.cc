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
#include "ops/auto_generate/gen_lite_ops.h"

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
  // ArgMaxV2 doesn't have keep_dims attr, replace by ArgMaxWithValue
  auto keep_dims_ptr = src_prim->GetAttr(ops::kKeepDims);
  if (keep_dims_ptr != nullptr && GetValue<bool>(keep_dims_ptr)) {
    // adjust axis and keep_dims to input to adapt mindir.
    auto axis_ptr = src_prim->GetAttr(ops::kAxis);
    CHECK_NULL_RETURN(axis_ptr);
    auto axis_value_node = NewValueNode<int64_t>(GetValue<int64_t>(axis_ptr));
    MS_CHECK_TRUE_MSG(axis_value_node != nullptr, lite::RET_ERROR, "New value node for axis failed.");
    std::vector<int64_t> shape_vec = {};
    auto axis_abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec);
    CHECK_NULL_RETURN(axis_abstract);
    axis_value_node->set_abstract(axis_abstract);
    cnode->add_input(axis_value_node);

    auto keep_dims_value_node = NewValueNode<bool>(GetValue<bool>(keep_dims_ptr));
    MS_CHECK_TRUE_MSG(keep_dims_value_node != nullptr, lite::RET_ERROR, "New value node for keep_dims failed.");
    auto keep_dims_abstract = std::make_shared<abstract::AbstractTensor>(kBool, shape_vec);
    CHECK_NULL_RETURN(keep_dims_abstract);
    keep_dims_value_node->set_abstract(keep_dims_abstract);
    cnode->add_input(keep_dims_value_node);

    auto argmax = std::make_shared<ops::ArgMaxWithValue>();
    CHECK_NULL_RETURN(argmax);
    auto dst_prim = argmax->GetPrim();
    CHECK_NULL_RETURN(dst_prim);
    dst_prim->SetAttrs(src_prim->attrs());
    value_node->set_value(dst_prim);
    return lite::RET_OK;
  }

  auto dst_prim = std::make_shared<acl::ArgMaxV2>();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->AddAttr("output_type", TypeIdToType(kNumberTypeInt32));
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameArgMaxFusion, ArgMaxFusionMapper)
}  // namespace lite
}  // namespace mindspore
