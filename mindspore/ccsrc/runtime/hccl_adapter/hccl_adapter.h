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

#ifndef MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H
#define MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H

#include <string>
#include <vector>
#include <memory>
#include "mindspore/core/ir/anf.h"
#include "hccl/hccl_types.h"

namespace mindspore::hccl {
struct HcclTaskInfo {
  std::string private_def;
  int64_t workspace_size;
  int64_t stream_num;
};

bool InitHccl(uint32_t device_id, std::string_view rank_id, std::string_view rank_file);
bool FinalizeHccl();
bool GenTask(const AnfNodePtr &node, HcclDataType datatype, std::vector<HcclTaskInfo> *task_info_lists);
int64_t CalcWorkspaceSize(const AnfNodePtr &node, HcclDataType datatype);
void *GetHcclOpsKernelInfoStore();
std::string GetHcclType(const AnfNodePtr &node);
}  // namespace mindspore::hccl
#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H
