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

#include "backend/kernel_compiler/hccl/hcom_all_broadcast.h"
#include <memory>
#include "utils/ms_context.h"
#include "backend/kernel_compiler/hccl/hccl_context.h"
#include "external/hccl/hccl.h"

namespace mindspore {
namespace kernel {
bool HcomAllBroadCastKernel::Launch(const std::vector<AddressPtr> &inputs,
                                    const std::vector<AddressPtr> & /*workspace*/,
                                    const std::vector<AddressPtr> & /*outputs*/, void *stream_ptr) {
  if (inputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "BroadCast param is empty";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto hccl_result = HcclBroadcast(inputs[0]->addr, hccl_count_, hccl_data_type_list_[0], root_id_,
                                   HcclContext::GetInstance().hccl_comm(), stream_ptr);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcomBroadcastOp : hcom_broadcast fail, return: " << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
