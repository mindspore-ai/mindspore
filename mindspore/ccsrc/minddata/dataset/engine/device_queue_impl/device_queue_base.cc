/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/device_queue_impl/device_queue_base.h"
#include "utils/ms_utils.h"
#ifndef ENABLE_SECURITY
#include "minddata/dataset/engine/perf/profiling.h"
#endif
#include "minddata/dataset/util/log_adapter.h"
#if ENABLE_D
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#endif

namespace mindspore {
namespace dataset {
constexpr auto kUnknownErrorString = "Unknown error occurred";
extern const bool kIsHeterogeneous = []() noexcept -> bool {
  int32_t is_heterogeneous = 0;
  (void)rtGetIsHeterogenous(&is_heterogeneous);
  return is_heterogeneous == 1;
}();

DeviceQueueBase::DeviceQueueBase(const std::string &channel_name, int32_t device_id) {
  // init ErrorManager, 0 means success
  if (ErrorManager::GetInstance().Init() != 0) {
    MS_LOG(WARNING) << "[Internal Error] Init ErrorManager failed.";
  }
  channel_name_ = channel_name;
  device_id_ = device_id;
}

Status DeviceQueueBase::GetAclDataType(DataType d_type, aclDataType *datatype) {
  switch (d_type.value()) {
    case DataType::DE_BOOL:
      *datatype = ACL_BOOL;
      break;
    case DataType::DE_INT8:
      *datatype = ACL_INT8;
      break;
    case DataType::DE_UINT8:
      *datatype = ACL_UINT8;
      break;
    case DataType::DE_INT16:
      *datatype = ACL_INT16;
      break;
    case DataType::DE_UINT16:
      *datatype = ACL_UINT16;
      break;
    case DataType::DE_INT32:
      *datatype = ACL_INT32;
      break;
    case DataType::DE_UINT32:
      *datatype = ACL_UINT32;
      break;
    case DataType::DE_FLOAT16:
      *datatype = ACL_FLOAT16;
      break;
    case DataType::DE_FLOAT32:
      *datatype = ACL_FLOAT;
      break;
    case DataType::DE_FLOAT64:
      *datatype = ACL_DOUBLE;
      break;
    case DataType::DE_INT64:
      *datatype = ACL_INT64;
      break;
    case DataType::DE_UINT64:
      *datatype = ACL_UINT64;
      break;
    default:
      RETURN_STATUS_UNEXPECTED("Invalid data, got unexpected data type.");
  }
  return Status::OK();
}

void DeviceQueueBase::ReportErrorMessage() {
  const std::string &error_message = ErrorManager::GetInstance().GetErrorMessage();
  if (!error_message.empty() && error_message.find(kUnknownErrorString) == std::string::npos) {
    MS_LOG(ERROR) << "Ascend error occurred, error message:\n" << error_message;
  }
}
}  // namespace dataset
}  // namespace mindspore
