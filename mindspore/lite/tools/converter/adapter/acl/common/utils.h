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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_UTILS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_UTILS_H_

#include <vector>
#include <memory>
#include <string>
#include "include/errorcode.h"
#include "ir/anf.h"
#include "ir/dtype/type_id.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace lite {
using BaseOperatorPtr = std::shared_ptr<mindspore::ops::BaseOperator>;

namespace acl {
STATUS GetShapeVectorFromCNode(const mindspore::CNodePtr &cnode, std::vector<int64_t> *shape_vector);

TypeId GetTypeFromNode(const AnfNodePtr &node, const size_t tuple_idx = 0);

std::vector<int> GetIntParameterData(const ParameterPtr &param_ptr);

std::vector<int64_t> GetInt64ParameterData(const ParameterPtr &param_ptr);

std::vector<float> GetFloatParameterData(const ParameterPtr &param_ptr);

std::string GetCNodeTargetFuncName(const CNodePtr &cnode);

STATUS DelRedundantParameter(const FuncGraphPtr &func_graph);
}  // namespace acl
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_UTILS_H_
