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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DEVICE_QUEUE_IMPL_DEVICE_QUEUE_IMPL_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DEVICE_QUEUE_IMPL_DEVICE_QUEUE_IMPL_H_

#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include "acl/acl_tdt.h"
#include "runtime/rt_mem_queue.h"
#include "runtime/dev.h"
#include "runtime/config.h"
#include "graph/def_types.h"
#include "common/util/error_manager/error_manager.h"

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
extern const bool kIsHeterogeneous;

class DeviceQueueBase {
 public:
  virtual Status hostPush(TensorRow ts_row, bool profiling, int32_t *time,
                          acltdtTensorType tdt_type = ACL_TENSOR_DATA_TENSOR) = 0;

  DeviceQueueBase(const std::string &channel_name, int32_t device_id);

  virtual ~DeviceQueueBase() {}

  acltdtChannelHandle *acl_handle_;

 protected:
  Status GetAclDataType(DataType d_type, aclDataType *datatype);

  void ReportErrorMessage();

  std::string channel_name_;

  int32_t device_id_ = 0;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DEVICE_QUEUE_IMPL_DEVICE_QUEUE_IMPL_H_
