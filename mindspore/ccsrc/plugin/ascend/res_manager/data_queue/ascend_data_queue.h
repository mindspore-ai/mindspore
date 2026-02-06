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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DATA_QUEUE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DATA_QUEUE_H_

#include <unistd.h>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <queue>
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/data_queue/data_queue.h"
#include "acl/acl_tdt.h"

namespace mindspore {
namespace device {
class AscendDataQueueDynamic : public DataQueue {
 public:
  explicit AscendDataQueueDynamic(const std::string &channel_name, const size_t capacity);
  ~AscendDataQueueDynamic() override = default;

  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override;
  DataQueueStatus Pop() override;

 private:
  struct NodeInfo {
    std::vector<DataQueueItem> data_;
  };
  aclrtStream stream_;
  std::unique_ptr<NodeInfo[]> node_info_;
};

namespace tdt_handle {
void AddHandle(acltdtChannelHandle **handle, std::thread *use_thread);
bool DestroyHandle();
void DelHandle(acltdtChannelHandle **handle);
bool IsClosed();
}  // namespace tdt_handle

class WingmanQueue : public DataQueue {
 public:
  explicit WingmanQueue(const std::string &channel_name) : DataQueue(channel_name, 0) {}
  ~WingmanQueue() override = default;
  void Close() override;
  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override;
  DataQueueStatus FrontAsync(std::vector<DataQueueItem> *data) const override;
  DataQueueStatus Pop() override;
  bool IsEmpty() const override { return queue_.empty(); }
  bool IsFull() const override { return false; }
  size_t Size() const override { return queue_.size(); }

 private:
  std::queue<std::vector<DataQueueItem>> queue_;
};

class AscendTdtQueue : public DataQueue {
 public:
  explicit AscendTdtQueue(const std::string &channel_name);
  ~AscendTdtQueue() override;

  bool IsOpen() const override;
  DataQueueStatus Push(std::vector<DataQueueItem> data) override;
  DataQueueStatus Front(std::vector<DataQueueItem> *data) const override { return DataQueueStatus::SUCCESS; }
  DataQueueStatus Pop() override { return DataQueueStatus::SUCCESS; }
  size_t QueryQueueSize() const override;
  std::string QueueType() const override { return queue_type_; }

 private:
  void DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item = true) const;
  bool AssembleTensor2AclDataset(const std::vector<DataQueueItem> &data, acltdtDataset *acl_dataset) const;
  void ParseType(aclDataType acl_data_type, std::string *data_type) const;
  bool Translate(const std::vector<DataQueueItem> &data, acltdtDataset **output_acl_dataset) const;

  acltdtChannelHandle *acl_handle_;
  uint32_t device_id_;
  std::string queue_type_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_BLOCKING_QUEUE_H_
