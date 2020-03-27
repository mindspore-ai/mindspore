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

#include "device/ascend/tasksink/runtime_utils.h"

#include <string>

#include "hccl/hcom.h"
#include "utils/log_adapter.h"
#include "utils/utils.h"

constexpr auto kHcomBroadcast = "hcom_broadcast_";
constexpr auto kHcomAllGather = "hcom_all_gather_";
constexpr auto kHcomAllReduce = "hcom_all_reduce_";
constexpr auto kHcomReduceScatter = "hcom_reduce_scatter_";
constexpr auto kUnderline = "_";
namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
bool RuntimeUtils::HcomBindModel(rtModel_t model, rtStream_t stream) {
  hcclResult_t ret = hcom_bind_model(model, stream);
  if (ret != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "Call hcom_bind_model failed, ret: 0x" << static_cast<int>(ret);
    return false;
  }
  return true;
}

bool RuntimeUtils::HcomUnbindModel(rtModel_t model) {
  hcclResult_t ret = hcom_unbind_model(model);
  if (ret != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "Call hcom_unbind_model failed, ret: 0x" << static_cast<int>(ret);
    return false;
  }
  return true;
}

bool RuntimeUtils::HcomDistribute(const std::shared_ptr<HcclTaskInfo> &task_info, rtStream_t stream) {
  MS_LOG(INFO) << "hccl distribute start";
  MS_EXCEPTION_IF_NULL(task_info);
  hcclResult_t ret;
  static uint32_t task_counter = 0;

  if (task_info->hccl_type() == kBroadcastOpName) {
    // call hcom broadcast interface to run op
    const string tag_broadcast = kHcomBroadcast + std::to_string(task_counter++) + kUnderline + std::to_string(0);
    ret = hcom_broadcast(tag_broadcast.c_str(), reinterpret_cast<void *>(task_info->input_data_addr()),
                         static_cast<u64>(task_info->count()), static_cast<hcclDataType_t>(task_info->data_type()),
                         static_cast<u32>(task_info->root_id()), nullptr, stream);
    if (ret != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "hcom_broadcast fail, return ret: " << static_cast<int>(ret);
      return false;
    }
  } else if (task_info->hccl_type() == kAllGatherOpName) {
    // call hcom allgather interface to run op
    const string tag_all_gather = kHcomAllGather + std::to_string(task_counter++) + kUnderline + std::to_string(0);
    ret = hcom_all_gather(tag_all_gather.c_str(), reinterpret_cast<void *>(task_info->input_data_addr()),
                          reinterpret_cast<void *>(task_info->output_data_addr()), static_cast<u64>(task_info->count()),
                          static_cast<hcclDataType_t>(task_info->data_type()), nullptr, stream);
    if (ret != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "hcom_all_gather fail, return ret: " << ret;
      return false;
    }
  } else if (task_info->hccl_type() == kAllReduceOpName) {
    // call hcom allreduce interface to run op
    const string tag_all_reduce = kHcomAllReduce + std::to_string(task_counter++) + kUnderline + std::to_string(0);
    ret = hcom_all_reduce(tag_all_reduce.c_str(), reinterpret_cast<void *>(task_info->input_data_addr()),
                          reinterpret_cast<void *>(task_info->output_data_addr()), static_cast<u64>(task_info->count()),
                          static_cast<hcclDataType_t>(task_info->data_type()),
                          static_cast<hcclRedOp_t>(task_info->op_type()), nullptr, stream);
    if (ret != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "hcom_all_reduce fail, return ret: " << ret;
      return false;
    }
  } else if (task_info->hccl_type() == kReduceScatterOpName) {
    // call hcom reducescatter interface to run op
    const string tag_reduce_scatter =
      kHcomReduceScatter + std::to_string(task_counter++) + kUnderline + std::to_string(0);
    ret = hcom_reduce_scatter(tag_reduce_scatter.c_str(), reinterpret_cast<void *>(task_info->input_data_addr()),
                              reinterpret_cast<void *>(task_info->output_data_addr()),
                              static_cast<u64>(task_info->count()), static_cast<hcclDataType_t>(task_info->data_type()),
                              static_cast<hcclRedOp_t>(task_info->op_type()), nullptr, stream);
    if (ret != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "hcom_reduce_scatter fail, return ret: " << ret;
      return false;
    }
  }
  return true;
}
}  // namespace tasksink
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
