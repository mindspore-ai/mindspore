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

#include "tools/converter/acl/mapper/cast_mapper.h"
#include "tools/converter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/acl/common/utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNameCastInputNum = 3;
}  // namespace

STATUS CastMapper::Mapper(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Cnode is nullptr.";
    return lite::RET_ERROR;
  }
  if (cnode->size() != kNameCastInputNum) {
    MS_LOG(ERROR) << "Input size of gather must be three.";
    return lite::RET_ERROR;
  }
  // convert last parameter to const value node
  auto to_input = cnode->input(kNameCastInputNum - 1);
  if (!utils::isa<ParameterPtr>(to_input)) {
    MS_LOG(ERROR) << "The to node is not parameter.";
    return lite::RET_ERROR;
  }
  ParameterPtr to_param = to_input->cast<ParameterPtr>();
  auto data = acl::GetIntParameterData(to_param);
  int dst_type = data.empty() ? kNumberTypeInt32 : data.front();
  TypePtr type_ptr = TypeIdToType(TypeId(dst_type));
  if (type_ptr == nullptr) {
    MS_LOG(ERROR) << "New type ptr failed.";
    return lite::RET_ERROR;
  }
  ValueNodePtr value_node = NewValueNode(type_ptr);
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "New value node failed.";
    return lite::RET_ERROR;
  }
  cnode->set_input(kNameCastInputNum - 1, value_node);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameCast, CastMapper)
}  // namespace lite
}  // namespace mindspore
