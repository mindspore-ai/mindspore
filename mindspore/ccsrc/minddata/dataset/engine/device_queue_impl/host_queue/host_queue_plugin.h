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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DEVICE_QUEUE_IMPL_HOST_QUEUE_HOST_QUEUE_PLUGIN_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DEVICE_QUEUE_IMPL_HOST_QUEUE_HOST_QUEUE_PLUGIN_H_

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
#include "minddata/dataset/engine/device_queue_impl/device_queue_base.h"

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
struct ItemInfo {
  int32_t version;
  int32_t data_type;
  uint32_t cur_cnt;
  uint32_t cnt;
  int32_t tensor_type;
  uint32_t dim_num;
  char reserved[32];
  uint64_t data_len;
};

struct DataItemInfo {
  ItemInfo ctrl_info;
  std::vector<int64_t> dims;
  void *data_ptr;
};

class HostQueueImpl : public DeviceQueueBase {
 public:
  Status hostPush(TensorRow ts_row, bool profiling, int32_t *time, acltdtTensorType tdt_type = ACL_TENSOR_DATA_TENSOR);

  HostQueueImpl(const std::string &channel_name, int32_t device_id);

  ~HostQueueImpl() {}

 private:
  Status HostQueueInit();

  Status SendDataByHostQueue(const TensorRow &tensor_row, const acltdtTensorType &data_type);

  Status SetTransId4MBuf(void **buff);

  Status LaunchTensor2MBuff(const acltdtTensorType &acl_type, const TensorRow &tensor_row, void **buff);
  Status EnqueueData(void *buff, bool *need_resend);

  Status CreateDataItemInfos(const acltdtTensorType &acl_type, const TensorRow &ts_row,
                             std::vector<DataItemInfo> *items);
  Status SerializeDataItemInfos(std::vector<DataItemInfo> *items, void **buff, const acltdtTensorType &acl_type);
  Status AddDataItemInfo(const acltdtTensorType &tdt_data_type, const int32_t &tensor_type, const int64_t *dims,
                         const size_t &dim_size, void *data_ptr, const uint64_t &data_len,
                         std::vector<DataItemInfo> *items);
  void HostQueueFreeBuff(void *buff);

  std::mutex queue_id_to_trans_id_map_mutex;
  std::map<uint32_t, uint64_t> queue_id_to_trans_id_map;
  uint32_t queue_id_ = 0;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DEVICE_QUEUE_IMPL_HOST_QUEUE_HOST_QUEUE_PLUGIN_H_
