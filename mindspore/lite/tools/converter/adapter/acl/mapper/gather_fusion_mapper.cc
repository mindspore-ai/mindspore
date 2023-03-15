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

#define USE_DEPRECATED_API
#include "tools/converter/adapter/acl/mapper/gather_fusion_mapper.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNameGatherInputNum = 4;
}  // namespace

STATUS GatherMapper::Mapper(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "Cnode is nullptr.");
  if (cnode->size() != kNameGatherInputNum) {
    MS_LOG(ERROR) << "Input size of gather must be " << kNameGatherInputNum << ", real size: " << cnode->size();
    return lite::RET_ERROR;
  }
  // convert last parameter to const value node
  auto axis_input = cnode->input(kNameGatherInputNum - 1);
  MS_CHECK_TRUE_MSG(axis_input != nullptr, lite::RET_ERROR, "axis_input is nullptr.");
  if (axis_input->isa<ValueNode>()) {
    MS_LOG(INFO) << axis_input->fullname_with_scope() << " is value node";
    return lite::RET_OK;
  }
  if (!utils::isa<ParameterPtr>(axis_input)) {
    MS_LOG(ERROR) << "The axis node is not parameter.";
    return lite::RET_ERROR;
  }
  ParameterPtr axis_param = axis_input->cast<ParameterPtr>();
  MS_CHECK_TRUE_MSG(axis_param != nullptr, lite::RET_ERROR, "axis_param is nullptr.");
  auto data = acl::GetIntParameterData(axis_param);
  int64_t axis = data.empty() ? 0 : static_cast<int64_t>(data.front());
  ValueNodePtr value_node = NewValueNode<int64_t>(axis);
  std::vector<int64_t> shape_vec_shape = {};
  auto abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec_shape);
  value_node->set_abstract(abstract);
  MS_CHECK_TRUE_MSG(value_node != nullptr, lite::RET_ERROR, "New value node failed.");
  cnode->set_input(kNameGatherInputNum - 1, value_node);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameGather, GatherMapper)
}  // namespace lite
}  // namespace mindspore
