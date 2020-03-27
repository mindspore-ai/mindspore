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

#include "kernel/hccl/hcom_all_reduce.h"

#include <algorithm>
#include <memory>
#include <string>

#include "utils/context/ms_context.h"

namespace mindspore {
namespace kernel {
bool HcomAllReduceKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->enable_task_sink()) {
    return true;
  }
  const char *tag = "Hccl-AllReduce";
  auto stream = reinterpret_cast<rtStream_t>(stream_ptr);
  hcclResult_t ret = hcom_all_reduce(tag, inputs[0]->addr, outputs[0]->addr, hccl_count_, hccl_data_type_list_[0],
                                     op_type_, nullptr, stream);
  if (ret != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcomAllReduceKernelOp : hcom_all_reduce fail, return: " << static_cast<int>(ret);
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
