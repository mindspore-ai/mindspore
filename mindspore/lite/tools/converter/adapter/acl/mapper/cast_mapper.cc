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

#include "tools/converter/adapter/acl/mapper/cast_mapper.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNameCastInputNum = 3;
}  // namespace

STATUS CastMapper::Mapper(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "Cnode is nullptr.");
  if (cnode->size() != kNameCastInputNum) {
    MS_LOG(ERROR) << "Input size of cast must be " << kNameCastInputNum << ", real size: " << cnode->size();
    return lite::RET_ERROR;
  }
  // convert last parameter to const value node
  auto to_input = cnode->input(kNameCastInputNum - 1);
  MS_CHECK_TRUE_MSG(to_input != nullptr, lite::RET_ERROR, "to_input is nullptr.");
  if (to_input->isa<ValueNode>()) {
    return lite::RET_OK;
  }
  if (!utils::isa<ParameterPtr>(to_input)) {
    MS_LOG(ERROR) << "The to node is not parameter.";
    return lite::RET_ERROR;
  }
  ParameterPtr to_param = to_input->cast<ParameterPtr>();
  MS_CHECK_TRUE_MSG(to_param != nullptr, lite::RET_ERROR, "to_param is nullptr.");
  auto data = acl::GetIntParameterData(to_param);
  int dst_type = data.empty() ? kNumberTypeInt32 : data.front();
  TypePtr type_ptr = TypeIdToType(TypeId(dst_type));
  MS_CHECK_TRUE_MSG(type_ptr != nullptr, lite::RET_ERROR, "New type ptr failed.");
  ValueNodePtr value_node = NewValueNode(type_ptr);
  std::vector<int64_t> shape_vec_shape = {};
  auto abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec_shape);
  value_node->set_abstract(abstract);
  MS_CHECK_TRUE_MSG(value_node != nullptr, lite::RET_ERROR, "New value node failed.");
  cnode->set_input(kNameCastInputNum - 1, value_node);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameCast, CastMapper)
}  // namespace lite
}  // namespace mindspore
