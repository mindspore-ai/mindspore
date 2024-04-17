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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_HOST_QUEUE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_HOST_QUEUE_H_

#include <unistd.h>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <queue>
#include "runtime/hardware/device_context_manager.h"
#include "include/backend/data_queue/data_queue.h"
#include "include/backend/data_queue/blocking_queue.h"
#include "runtime/rt.h"
#include "acl/acl_tdt.h"

namespace mindspore {
namespace device {
class AscendHostQueue : public DataQueue {
 public:
  explicit AscendHostQueue(const std::string &channel_name);
  ~AscendHostQueue() override = default;

  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override { return DataQueueStatus::SUCCESS; }
  DataQueueStatus Pop() override { return DataQueueStatus::SUCCESS; }

  static constexpr int64_t kMaxDimSize = 32;
#pragma pack(push, 1)
  struct RuntimeTensorDesc {
    uint64_t data_addr;
    int64_t data_offset_size;
    int64_t dtype;
    int64_t shape[kMaxDimSize + 1];
    int64_t original_shape[kMaxDimSize + 1];
    int64_t format;
    int64_t sub_format;
    uint8_t reserved[456];
  };
#pragma pack(pop)

  struct DataItemInfo {
    struct ItemInfo {
      int32_t version;
      int32_t data_type;
      uint32_t cur_count;
      uint32_t count;
      int32_t tensor_type;
      uint32_t dim_num;
      char reserved[32];
      uint64_t data_len;
    } item_info;
    std::vector<int64_t> dims;
    void *data_ptr;
  };

 private:
  bool HostQueueInit();
  bool SendDataByHostQueue(const std::vector<DataQueueItem> &data);
  bool SetTransId4MBuf(void **buff);
  bool LaunchTensor2MBuff(const std::vector<DataQueueItem> &data, void **buff);
  bool EnqueueData(void *buff, bool *need_resend);
  bool CreateDataItemInfos(const std::vector<DataQueueItem> &data, std::vector<DataItemInfo> *items) const;
  bool SerializeDataItemInfos(std::vector<DataItemInfo> *items, void **buff) const;
  DataItemInfo BuildDataItemInfo(acltdtTensorType acl_data_type, int32_t tensor_type, const int64_t *dims,
                                 size_t dim_size, void *data_ptr, uint64_t data_len) const;
  void HostQueueFreeBuff(void *buff);

  uint32_t device_id_;
  std::mutex queue_id_to_trans_id_map_mutex_;
  std::map<uint32_t, uint64_t> queue_id_to_trans_id_map_;
  uint32_t queue_id_;
  const uint32_t rt_mem_queue_depth_ = 128;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_HOST_QUEUE_H_
