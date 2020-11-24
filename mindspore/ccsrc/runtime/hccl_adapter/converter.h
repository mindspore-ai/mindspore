/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_RUNTIME_HCCL_ADAPTER_CONVERTER_H
#define MINDSPORE_RUNTIME_HCCL_ADAPTER_CONVERTER_H

#include <string>
#include <vector>
#include <memory>
#include <tuple>
#define google ascend_private
#include "graph/node.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "proto/task.pb.h"
#undef google
#include "runtime/hccl_adapter/hccl_adapter.h"
#include "mindspore/core/ir/anf.h"

namespace mindspore::hccl {
// return graph ptr to keep reference count
std::tuple<ge::NodePtr, ge::ComputeGraphPtr> GenerateStubGeNode(const AnfNodePtr &anf_node, HcclDataType datatype);
HcclTaskInfo ParseDomiTask(const ge::OpDescPtr &op, const domi::TaskDef &task_def);
std::string GetGeNodeName(const CNodePtr &cnode);
}  // namespace mindspore::hccl
#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_CONVERTER_H
