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
#include "tools/converter/adapter/acl/mapper/transpose_mapper.h"
#include <algorithm>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kCommonInputNum = 3;
}  // namespace

STATUS TransposeMapper::Mapper(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr.");
  if (cnode->size() != kCommonInputNum) {
    MS_LOG(ERROR) << "Input size of transpose must be " << kCommonInputNum << ", real size: " << cnode->size();
    return lite::RET_ERROR;
  }
  // convert last parameter to const value node
  auto perm_input = cnode->input(kCommonInputNum - 1);
  MS_CHECK_TRUE_MSG(perm_input != nullptr, lite::RET_ERROR, "perm_input is nullptr.");
  if (!utils::isa<ParameterPtr>(perm_input)) {
    MS_LOG(ERROR) << "The perm node is not parameter.";
    return lite::RET_ERROR;
  }
  ParameterPtr perm_param = perm_input->cast<ParameterPtr>();
  MS_CHECK_TRUE_MSG(perm_param != nullptr, lite::RET_ERROR, "ParameterPtr casts failed.");
  auto data = acl::GetIntParameterData(perm_param);
  std::vector<int64_t> perm;
  std::transform(data.begin(), data.end(), std::back_inserter(perm),
                 [](int32_t n) -> int64_t { return static_cast<int64_t>(n); });
  ValueNodePtr value_node = NewValueNode<std::vector<int64_t>>(perm);
  std::vector<int64_t> shape_vec_shape = {};
  auto abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec_shape);
  value_node->set_abstract(abstract);
  MS_CHECK_TRUE_MSG(value_node != nullptr, lite::RET_ERROR, "New value node failed.");
  cnode->set_input(kCommonInputNum - 1, value_node);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameTranspose, TransposeMapper)
}  // namespace lite
}  // namespace mindspore
