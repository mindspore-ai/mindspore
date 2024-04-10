/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend_native/hccl_wrapper.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
namespace mindspore::ascend_native {

int HcclAllReduceSumFP32(void *send_buf, void *recv_buf, uint64_t count, void *stream) {
  auto hccl_datatype = HcclDataType::HCCL_DATA_TYPE_FP32;
  auto hccl_op = HcclReduceOp::HCCL_REDUCE_SUM;
  auto comm = mindspore::hccl::HcclAdapter::GetInstance().get_hccl_comm();
  return mindspore::hccl::HcclAdapter::GetInstance().HcclAllReduce(send_buf, recv_buf, count, hccl_datatype, hccl_op,
                                                                   stream, comm);
}
}  // namespace mindspore::ascend_native
