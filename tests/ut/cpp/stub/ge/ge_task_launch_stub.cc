/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <vector>
#include "runtime/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace hccl {
bool InitHccl(uint32_t, std::string_view, std::string_view) { return true; }
bool FinalizeHccl() { return true; }
bool GenTask(const AnfNodePtr &, HcclDataType, std::vector<HcclTaskInfo> *) { return true; }
int64_t CalcWorkspaceSize(const AnfNodePtr &, HcclDataType) { return 0; }
void *GetHcclOpsKernelInfoStore() { return nullptr; }
std::string GetHcclType(const AnfNodePtr &) { return ""; }
}  // namespace hccl
}  // namespace mindspore
