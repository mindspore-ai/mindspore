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

#define MS_API __attribute__((visibility("default")))

namespace mindspore::hccl {
struct MS_API HcclTaskInfo {
  std::string private_def;
  int64_t workspace_size;
  int64_t stream_num;
};

MS_API bool InitHccl(uint32_t device_id, std::string_view rank_id, std::string_view rank_file);
MS_API bool FinalizeHccl();
MS_API bool GenTask(const AnfNodePtr &node, HcclDataType datatype, std::vector<HcclTaskInfo> *task_info_lists);
MS_API bool CalcOpRunningParam(const AnfNodePtr &node);
MS_API void *GetHcclOpsKernelInfoStore();
MS_API std::string GetHcclType(const AnfNodePtr &node);
}  // namespace mindspore::hccl
#undef MS_API
#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H
