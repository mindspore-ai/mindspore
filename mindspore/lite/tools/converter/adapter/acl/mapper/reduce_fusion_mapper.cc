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

#include "tools/converter/adapter/acl/mapper/reduce_fusion_mapper.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"
#include "ops/reduce_sum.h"
#include "ops/reduce_mean.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameReduceInputNum = 3;

}  // namespace

STATUS ReduceFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto attr_val = src_prim->GetAttr(ops::kMode);
  CHECK_NULL_RETURN(attr_val);
  int64_t mode = GetValue<int64_t>(attr_val);
  PrimitivePtr dst_prim = nullptr;
  if (mode == static_cast<int64_t>(ReduceMode::Reduce_Sum)) {
    dst_prim = std::make_shared<ops::ReduceSum>();
  } else if (mode == static_cast<int64_t>(ReduceMode::Reduce_Mean)) {
    dst_prim = std::make_shared<ops::ReduceMean>();
  }
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);

  if (AdjustInput(cnode) != RET_OK) {
    MS_LOG(ERROR) << "Adjust reduce input failed.";
    return lite::RET_ERROR;
  }
  return RET_OK;
}

STATUS ReduceFusionMapper::AdjustInput(const CNodePtr &cnode) {
  if (cnode->size() != kNameReduceInputNum) {
    MS_LOG(ERROR) << "Input size of reduce must be " << kNameReduceInputNum << ", real size: " << cnode->size();
    return lite::RET_ERROR;
  }
  auto axes_input = cnode->input(kNameReduceInputNum - 1);
  CHECK_NULL_RETURN(axes_input);
  if (!utils::isa<ParameterPtr>(axes_input)) {
    MS_LOG(ERROR) << "The reduce node is not parameter.";
    return lite::RET_ERROR;
  }
  ParameterPtr axes_param = axes_input->cast<ParameterPtr>();
  CHECK_NULL_RETURN(axes_param);
  auto data = acl::GetIntParameterData(axes_param);
  std::vector<int64_t> axes;
  std::transform(data.begin(), data.end(), std::back_inserter(axes),
                 [](int32_t n) -> int64_t { return static_cast<int64_t>(n); });
  ValueNodePtr value_node = NewValueNode<std::vector<int64_t>>(axes);
  CHECK_NULL_RETURN(value_node);
  cnode->set_input(kNameReduceInputNum - 1, value_node);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameReduceFusion, ReduceFusionMapper)
}  // namespace lite
}  // namespace mindspore
