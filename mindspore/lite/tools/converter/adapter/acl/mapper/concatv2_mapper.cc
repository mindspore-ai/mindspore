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

#include "tools/converter/adapter/acl/mapper/concatv2_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore {
namespace lite {
const auto kNameConcatV2 = "ConcatV2";

namespace {
constexpr auto kNameInputNums = "N";
constexpr int64_t kInputMinNum = 2;
}  // namespace

STATUS ConcatV2Mapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::ConcatV2>();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  int64_t num = static_cast<int64_t>(cnode->inputs().size());
  if (num < kInputMinNum) {
    MS_LOG(ERROR) << "Input size " << num << " is less than " << kInputMinNum;
    return RET_ERROR;
  }
  dst_prim->AddAttr(kNameInputNums, MakeValue(num - 1));
  value_node->set_value(dst_prim);

  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  auto status = AddIntAttrToInput(func_graph, cnode, dst_prim, ops::kAxis, false);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Add constant value to input failed.";
    return lite::RET_ERROR;
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConcatV2, ConcatV2Mapper)
}  // namespace lite
}  // namespace mindspore
