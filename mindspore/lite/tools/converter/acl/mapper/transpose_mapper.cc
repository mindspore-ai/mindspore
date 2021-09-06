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

#include "tools/converter/acl/mapper/transpose_mapper.h"
#include <algorithm>
#include <vector>
#include "tools/converter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/acl/common/utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kCommonInputNum = 3;
}

STATUS TransposeMapper::Mapper(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Cnode is nullptr.";
    return lite::RET_ERROR;
  }
  if (cnode->size() != kCommonInputNum) {
    MS_LOG(ERROR) << "Input size of gather must be two.";
    return lite::RET_ERROR;
  }
  // convert last parameter to const value node
  auto perm_input = cnode->input(kCommonInputNum - 1);
  if (!utils::isa<ParameterPtr>(perm_input)) {
    MS_LOG(ERROR) << "The perm node is not parameter.";
    return lite::RET_ERROR;
  }
  ParameterPtr perm_param = perm_input->cast<ParameterPtr>();
  auto data = acl::GetIntParameterData(perm_param);
  std::vector<int64_t> perm;
  std::transform(data.begin(), data.end(), std::back_inserter(perm),
                 [](int32_t n) -> int64_t { return static_cast<int64_t>(n); });
  ValueNodePtr value_node = NewValueNode<std::vector<int64_t>>(perm);
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "New value node failed.";
    return lite::RET_ERROR;
  }
  cnode->set_input(kCommonInputNum - 1, value_node);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameTranspose, TransposeMapper)
}  // namespace lite
}  // namespace mindspore
