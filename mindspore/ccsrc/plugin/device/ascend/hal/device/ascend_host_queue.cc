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

#include "plugin/device/ascend/hal/device/ascend_host_queue.h"

#include <string>
#include <map>
#include "graph/def_types.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "utils/log_adapter.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "runtime/device/kernel_runtime.h"
#include "include/backend/distributed/ps/ps_cache/ps_data_prefetch.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace device {
namespace {
const std::map<aclDataType, std::string> kAclTypeToString = {
  {ACL_INT8, "int8"},       {ACL_UINT8, "uint8"},   {ACL_INT16, "int16"},    {ACL_UINT16, "uint16"},
  {ACL_INT32, "int32"},     {ACL_UINT32, "uint32"}, {ACL_INT64, "int64"},    {ACL_UINT64, "uint64"},
  {ACL_FLOAT16, "float16"}, {ACL_FLOAT, "float32"}, {ACL_DOUBLE, "float64"}, {ACL_BOOL, "bool"}};

const std::map<std::string, aclDataType> kStringTypeToAclType = []() -> std::map<std::string, aclDataType> {
  std::map<std::string, aclDataType> ret;
  for (const auto &[acl_type, type_str] : kAclTypeToString) {
    ret.emplace(type_str, acl_type);
  }
  return ret;
}();

constexpr auto kMbufHeadEndOfSequencePos = 128U;
constexpr auto kTransIdOffset = 64UL;
constexpr auto kSleepMilliSeconds = 500;

bool GetAclDataType(const std::string &str_type, aclDataType *acl_type) {
  MS_EXCEPTION_IF_NULL(acl_type);
  auto iter = kStringTypeToAclType.find(str_type);
  if (iter == kStringTypeToAclType.end()) {
    MS_LOG(EXCEPTION) << "Invalid type " << str_type;
  }
  *acl_type = iter->second;
  return true;
}
}  // namespace

AscendHostQueue::AscendHostQueue(const std::string &channel_name)
    : DataQueue(channel_name, 0), queue_id_to_trans_id_map_(), queue_id_(0) {
  // Init ErrorManager
  if (ascend::ErrorManagerAdapter::Init()) {
    MS_LOG(WARNING) << "[Internal Error] Init ErrorManager failed.";
  }
  // get device id
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  device_id_ = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  if (!HostQueueInit()) {
    MS_LOG(WARNING) << "Host queue init failed.";
  }
}

DataQueueStatus AscendHostQueue::Push(std::vector<DataQueueItem> data) {
  if (!SendDataByHostQueue(data)) {
    return DataQueueStatus::INTERNAL_ERROR;
  }

  return DataQueueStatus::SUCCESS;
}

bool AscendHostQueue::HostQueueInit() {
  auto rt_ret = aclrtSetDevice(device_id_);
  if (rt_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call aclrtSetDevice failed, ret = " << rt_ret;
    return false;
  }

  rt_ret = rtMemQueueInit(device_id_);
  if (rt_ret != ACL_RT_SUCCESS && rt_ret != ACL_ERROR_RT_REPEATED_INIT) {
    MS_LOG(ERROR) << "call rtMemQueueInit failed, ret = " << rt_ret;
    return false;
  }

  rtMemQueueAttr_t attr = {};
  auto errno_ret = memset_s(attr.name, RT_MQ_MAX_NAME_LEN, 0, RT_MQ_MAX_NAME_LEN);
  if (errno_ret != EOK) {
    MS_LOG(ERROR) << "call memset_s failed, ret = " << errno_ret;
    return false;
  }
  errno_ret = memcpy_s(attr.name, RT_MQ_MAX_NAME_LEN, channel_name_.c_str(), channel_name_.size() + 1);
  if (errno_ret != EOK) {
    MS_LOG(ERROR) << "call memcpy_s failed, ret = " << errno_ret;
    return false;
  }

  attr.depth = rt_mem_queue_depth_;
  attr.workMode = RT_MQ_MODE_DEFAULT;
  attr.flowCtrlFlag = false;
  attr.flowCtrlDropTime = 0;
  attr.overWriteFlag = false;
  rt_ret = rtMemQueueCreate(device_id_, &attr, &queue_id_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtMemQueueCreate failed, ret = " << rt_ret;
    return false;
  }

  rtMemBuffCfg_t buff_cfg = {};
  rt_ret = rtMbufInit(&buff_cfg);
  if (rt_ret != ACL_ERROR_NONE && rt_ret != ACL_ERROR_RT_REPEATED_INIT) {
    MS_LOG(ERROR) << "Call rtMbufInit failed, ret =" << rt_ret;
    return false;
  }
  std::lock_guard<std::mutex> lock(queue_id_to_trans_id_map_mutex_);
  (void)queue_id_to_trans_id_map_.emplace(queue_id_, 0);

  return true;
}

bool AscendHostQueue::SendDataByHostQueue(const std::vector<DataQueueItem> &data) {
  bool status;
  bool is_need_resend = false;
  void *buff = nullptr;
  if (!LaunchTensor2MBuff(data, &buff)) {
    return false;
  }
  do {
    if (is_need_resend) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMilliSeconds));
    }
    status = EnqueueData(buff, &is_need_resend);
  } while (status && is_need_resend);
  return true;
}

bool AscendHostQueue::SetTransId4MBuf(void **buff) {
  void *head_buff = nullptr;
  uint64_t head_size = 0UL;
  auto ret = rtMbufGetPrivInfo(*buff, &head_buff, &head_size);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtMbufGetPrivInfo failed, ret =" << ret;
    return false;
  }
  uint64_t *trans_id = reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(head_buff) + head_size - kTransIdOffset);
  const std::lock_guard<std::mutex> lk(queue_id_to_trans_id_map_mutex_);
  *trans_id = ++queue_id_to_trans_id_map_[queue_id_];
  MS_LOG(DEBUG) << "host queue[" << queue_id_ << "] set trans id[" << *trans_id << "] success";
  return true;
}

bool AscendHostQueue::LaunchTensor2MBuff(const std::vector<DataQueueItem> &data, void **buff) {
  std::vector<DataItemInfo> items;
  if (!CreateDataItemInfos(data, &items)) {
    return false;
  }
  if (!SerializeDataItemInfos(&items, buff)) {
    return false;
  }
  if (!SetTransId4MBuf(buff)) {
    return false;
  }
  return true;
}

bool AscendHostQueue::EnqueueData(void *buff, bool *need_resend) {
  MS_EXCEPTION_IF_NULL(need_resend);
  *need_resend = false;
  auto rt_error = aclrtSetDevice(device_id_);
  if (rt_error != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call aclrtSetDevice device failed, ret=" << rt_error;
    return false;
  }
  rt_error = rtMemQueueEnQueue(device_id_, queue_id_, buff);
  if (rt_error == ACL_ERROR_NONE) {
    return true;
  } else if (rt_error == ACL_ERROR_RT_QUEUE_FULL) {
    *need_resend = true;
    MS_LOG(DEBUG) << "queue[" << queue_id_ << "] is full, need call rtMemQueueEnQueue again";
  } else {
    HostQueueFreeBuff(buff);
    MS_LOG(ERROR) << "host enqueue queue[" << queue_id_ << "] failed, ret = " << rt_error;
    return false;
  }
  return true;
}

bool AscendHostQueue::CreateDataItemInfos(const std::vector<DataQueueItem> &data,
                                          std::vector<DataItemInfo> *items) const {
  MS_EXCEPTION_IF_NULL(items);
  if (data.empty()) {
    items->emplace_back(BuildDataItemInfo(ACL_TENSOR_DATA_END_OF_SEQUENCE, ACL_BOOL, nullptr, 0UL, nullptr, 0UL));
    return true;
  }

  for (auto &ts : data) {
    aclDataType acl_type;
    if (!GetAclDataType(ts.data_type, &acl_type)) {
      MS_LOG(ERROR) << "Convert type " << ts.data_type << " to acl type failed.";
      return false;
    }

    const auto &shape = ts.shapes;
    void *data_ptr = ts.data_ptr;
    size_t data_size = ts.data_len;

    if (ts.data_type != "string") {
      items->emplace_back(BuildDataItemInfo(ACL_TENSOR_DATA_TENSOR, static_cast<int32_t>(acl_type),
                                            (shape.empty() ? nullptr : &shape[0]), shape.size(), data_ptr, data_size));
    } else {
      MS_LOG(ERROR) << "Create data item failed when send data with type:" << ts.data_type;
    }
  }
  return true;
}

bool AscendHostQueue::SerializeDataItemInfos(std::vector<DataItemInfo> *items, void **buff) const {
  MS_EXCEPTION_IF_NULL(items);
  size_t count = items->size();
  size_t total_size = 0UL;
  for (size_t i = 0UL; i < count; ++i) {
    (*items)[i].item_info.cur_count = i;
    (*items)[i].item_info.count = count;
    total_size +=
      sizeof(DataItemInfo::ItemInfo) + (*items)[i].item_info.dim_num * sizeof(int64_t) + (*items)[i].item_info.data_len;
  }

  total_size += sizeof(RuntimeTensorDesc);
  auto errno_ret = rtMbufAlloc(buff, total_size);
  if (errno_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Call rtMbufAlloc with size[" << total_size << "] failed, ret = " << errno_ret;
    return false;
  }

  void *data = nullptr;
  errno_ret = rtMbufGetBuffAddr(*buff, &data);
  if (errno_ret != ACL_RT_SUCCESS) {
    (void)rtMbufFree(*buff);
    MS_LOG(ERROR) << "Call rtMbufGetBuffAddr with size[" << total_size << "] failed, ret = " << errno_ret;
    return false;
  }

  void *head_buf = nullptr;
  uint64_t head_size = 0UL;
  errno_ret = rtMbufGetPrivInfo(*buff, &head_buf, &head_size);
  if (errno_ret != ACL_RT_SUCCESS) {
    (void)rtMbufFree(*buff);
    MS_LOG(ERROR) << "Call rtMbufGetPrivInfo failed, ret =" << errno_ret;
    return false;
  }
  if ((head_buf != nullptr) && (head_size > kMbufHeadEndOfSequencePos)) {
    MS_LOG(DEBUG) << "Host queue set end_of_sequence mbuf head.";
  }
  data = ::ge::ValueToPtr(::ge::PtrToValue(data) + sizeof(RuntimeTensorDesc));
  size_t offset = 0UL;
  for (size_t i = 0UL; i < count; ++i) {
    errno_ret = memcpy_s(::ge::ValueToPtr(::ge::PtrToValue(data) + offset), sizeof(DataItemInfo::ItemInfo),
                         &((*items)[i].item_info), sizeof(DataItemInfo::ItemInfo));
    if (errno_ret != EOK) {
      (void)rtMbufFree(*buff);
      MS_LOG(ERROR) << "Call memcpy_s failed, ret = " << errno_ret;
      return false;
    }
    offset += sizeof(DataItemInfo::ItemInfo);

    for (size_t j = 0UL; j < (*items)[i].item_info.dim_num; ++j) {
      errno_ret = memcpy_s(::ge::ValueToPtr(::ge::PtrToValue(data) + offset), sizeof(int64_t), &((*items)[i].dims[j]),
                           sizeof(int64_t));
      if (errno_ret != EOK) {
        (void)rtMbufFree(*buff);
        MS_LOG(ERROR) << "Call memcpy_s failed, ret = " << errno_ret;
        return false;
      }
      offset += sizeof(int64_t);
    }

    if ((*items)[i].item_info.data_len == 0UL) {
      continue;
    }

    errno_ret = memcpy_s(::ge::ValueToPtr(::ge::PtrToValue(data) + offset), (*items)[i].item_info.data_len,
                         (*items)[i].data_ptr, (*items)[i].item_info.data_len);
    if (errno_ret != EOK) {
      (void)rtMbufFree(*buff);
      MS_LOG(ERROR) << "call memcpy_s failed, ret = " << errno_ret;
      return false;
    }
    offset += (*items)[i].item_info.data_len;
  }

  return true;
}

AscendHostQueue::DataItemInfo AscendHostQueue::BuildDataItemInfo(acltdtTensorType acl_data_type, int32_t tensor_type,
                                                                 const int64_t *dims, size_t dim_size, void *data_ptr,
                                                                 uint64_t data_len) const {
  DataItemInfo item = {};
  item.item_info.data_type = static_cast<int32_t>(acl_data_type);
  item.item_info.tensor_type = tensor_type;
  item.item_info.dim_num = dim_size;
  item.item_info.data_len = data_len;
  item.dims = std::vector<int64_t>(dims, dims + dim_size);
  item.data_ptr = data_ptr;
  return item;
}

void AscendHostQueue::HostQueueFreeBuff(void *buff) {
  auto rt_ret = rtMbufFree(buff);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtMbufFree failed, ret = " << rt_ret;
  }
}
}  // namespace device
}  // namespace mindspore
