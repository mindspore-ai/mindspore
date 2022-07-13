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
#include "minddata/dataset/engine/device_queue_impl/host_queue/host_queue_plugin.h"
#include "utils/ms_utils.h"
#ifndef ENABLE_SECURITY
#include "minddata/dataset/engine/perf/profiling.h"
#endif
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
constexpr auto kMbufHeadEndOfSequencePos = 128U;
constexpr auto kEndOfSequenceFlag = 0x5A;
constexpr auto kTransIdOffset = 64UL;
constexpr auto kSleepMilliSeconds = 500;
const std::map<acltdtTensorType, int32_t> type_map = {
  {ACL_TENSOR_DATA_TENSOR, 0}, {ACL_TENSOR_DATA_END_OF_SEQUENCE, 1}, {ACL_TENSOR_DATA_ABNORMAL, 2}};

HostQueueImpl::HostQueueImpl(const std::string &channel_name, int32_t device_id)
    : DeviceQueueBase(channel_name, device_id) {
  auto status = HostQueueInit();
  if (status != Status::OK()) {
    MS_LOG(WARNING) << "Host queue init failed.";
  }
}

Status HostQueueImpl::hostPush(TensorRow ts_row, bool profiling, int32_t *time, acltdtTensorType tdt_type) {
#ifndef ENABLE_SECURITY
  double start_time;
  if (profiling) {
    start_time = ProfilingTime::GetCurMilliSecond();
  }
#endif
  // send data by datagw
  auto status = SendDataByHostQueue(ts_row, tdt_type);
  if (status != Status::OK()) {
    RETURN_STATUS_UNEXPECTED("Host queue send data failed.");
  }
#ifndef ENABLE_SECURITY
  if (profiling) {
    double end_time = ProfilingTime::GetCurMilliSecond();
    *time = static_cast<int32_t>(end_time - start_time);
  }
#endif
  return status;
}

Status HostQueueImpl::HostQueueInit() {
  auto ret = rtSetDevice(device_id_);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtSetDevice failed, ret =" << ret;
    RETURN_STATUS_UNEXPECTED("Init:rtSetDevice failed.");
  }

  ret = rtMemQueueInit(device_id_);
  if (ret != ACL_RT_SUCCESS && ret != ACL_ERROR_RT_REPEATED_INIT) {
    MS_LOG(ERROR) << "call rtMemQueueInit failed, ret =" << ret;
    RETURN_STATUS_UNEXPECTED("Init:rtMemQueueInit failed.");
  }

  rtMemQueueAttr_t attr = {};
  auto mem_ret = memset_s(attr.name, RT_MQ_MAX_NAME_LEN, 0, RT_MQ_MAX_NAME_LEN);
  if (mem_ret != EOK) {
    MS_LOG(ERROR) << "call memset_s failed, ret =" << mem_ret;
    RETURN_STATUS_UNEXPECTED("Init:memset_s failed.");
  }
  mem_ret = memcpy_s(attr.name, RT_MQ_MAX_NAME_LEN, channel_name_.c_str(), channel_name_.size() + 1);
  if (mem_ret != EOK) {
    MS_LOG(ERROR) << "call memcpy_s failed, ret =" << mem_ret;
    RETURN_STATUS_UNEXPECTED("Init:memcpy_s failed.");
  }

  attr.depth = 128U;
  attr.workMode = RT_MQ_MODE_DEFAULT;
  attr.flowCtrlFlag = false;
  attr.flowCtrlDropTime = 0U;
  attr.overWriteFlag = false;
  ret = rtMemQueueCreate(device_id_, &attr, &queue_id_);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtMemQueueCreate failed, ret =" << ret;
    RETURN_STATUS_UNEXPECTED("Init:rtMemQueueCreate failed.");
  }

  rtMemBuffCfg_t buff_cfg = {0};
  ret = rtMbufInit(&buff_cfg);
  if (ret != ACL_RT_SUCCESS && ret != ACL_ERROR_RT_REPEATED_INIT) {
    MS_LOG(ERROR) << "call rtMbufInit failed, ret =" << ret;
    RETURN_STATUS_UNEXPECTED("Init:rtMbufInit failed.");
  }
  const std::lock_guard<std::mutex> lk(queue_id_to_trans_id_map_mutex);
  (void)queue_id_to_trans_id_map.insert({queue_id_, 0UL});

  return Status::OK();
}

Status HostQueueImpl::AddDataItemInfo(const acltdtTensorType &tdt_data_type, const int32_t &tensor_type,
                                      const int64_t *dims, const size_t &dim_size, void *data_ptr,
                                      const uint64_t &data_len, std::vector<DataItemInfo> *items) {
  DataItemInfo item = {};
  int32_t data_type = 0;
  auto it = type_map.find(tdt_data_type);
  if (it != type_map.end()) {
    data_type = it->second;
  }
  item.ctrl_info.data_type = data_type;
  item.ctrl_info.tensor_type = tensor_type;
  item.ctrl_info.dim_num = dim_size;
  item.ctrl_info.data_len = data_len;
  item.dims.clear();
  for (size_t i = 0; i < dim_size; ++i) {
    item.dims.push_back(dims[i]);
  }
  item.data_ptr = data_ptr;
  items->push_back(item);
  return Status::OK();
}

Status HostQueueImpl::CreateDataItemInfos(const acltdtTensorType &acl_type, const TensorRow &ts_row,
                                          std::vector<DataItemInfo> *items) {
  if (acl_type != ACL_TENSOR_DATA_TENSOR) {
    return AddDataItemInfo(acl_type, ACL_BOOL, nullptr, 0UL, nullptr, 0UL, items);
  }

  for (auto &ts : ts_row) {
    aclDataType data_type;
    RETURN_IF_NOT_OK(GetAclDataType(ts->type(), &data_type));
    TensorShape tsShape = ts->shape();
    std::shared_ptr<void> dataPtr =
      std::shared_ptr<void>(reinterpret_cast<uchar *>(&(*ts->begin<uint8_t>())), [](const void *elem) {});
    size_t dataLen = ts->SizeInBytes();
    const dsize_t dims = tsShape.Rank();
    std::vector<int64_t> dataShape;
    for (auto i = 0; i < dims; i++) {
      dataShape.emplace_back(tsShape[i]);
    }
    if (ts->type() != DataType::DE_STRING) {
      RETURN_IF_NOT_OK(AddDataItemInfo(ACL_TENSOR_DATA_TENSOR, data_type, (tsShape.empty() ? nullptr : &dataShape[0]),
                                       dims, dataPtr.get(), dataLen, items));
    } else {
      RETURN_STATUS_UNEXPECTED("Create data item failed when send data with type:" + std::to_string(data_type));
    }
  }
  return Status::OK();
}

Status HostQueueImpl::SerializeDataItemInfos(std::vector<DataItemInfo> *items, void **buff,
                                             const acltdtTensorType &acl_type) {
  size_t cnt = items->size();
  size_t total_size = 0UL;
  for (size_t i = 0UL; i < cnt; ++i) {
    (*items)[i].ctrl_info.cur_cnt = i;
    (*items)[i].ctrl_info.cnt = cnt;
    total_size += sizeof(ItemInfo) + (*items)[i].ctrl_info.dim_num * sizeof(int64_t) + (*items)[i].ctrl_info.data_len;
  }

  auto rt_error = rtMbufAlloc(buff, total_size);
  if (rt_error != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtMbufAlloc with size[" << total_size << "] failed, ret = " << rt_error;
    RETURN_STATUS_UNEXPECTED("SerializeDataItemInfo: rtMbufAlloc failed.");
  }

  void *data = nullptr;
  rt_error = rtMbufGetBuffAddr(*buff, &data);
  if (rt_error != ACL_RT_SUCCESS) {
    (void)rtMbufFree(*buff);
    MS_LOG(ERROR) << "call rtMbufGetBuffAddr with size[" << total_size << "] failed, ret = " << rt_error;
    RETURN_STATUS_UNEXPECTED("SerializeDataItemInfo:rtMbufGetBuffAddr failed.");
  }

  void *head_buf = nullptr;
  uint64_t head_size = 0UL;
  rt_error = rtMbufGetPrivInfo(*buff, &head_buf, &head_size);
  if (rt_error != ACL_RT_SUCCESS) {
    (void)rtMbufFree(*buff);
    MS_LOG(ERROR) << "call rtMbufGetPrivInfo failed, ret =" << rt_error;
    RETURN_STATUS_UNEXPECTED("SerializeDataItemInfo:rtMbufGetPrivInfo failed.");
  }
  if ((head_buf != nullptr) && (head_size > kMbufHeadEndOfSequencePos)) {
    MS_LOG(DEBUG) << "host queue set end_of_sequence mbuf head.";
  }

  size_t offset = 0UL;
  for (size_t i = 0UL; i < cnt; ++i) {
    auto mem_ret = memcpy_s(ge::ValueToPtr(ge::PtrToValue(data) + offset), sizeof(ItemInfo), &((*items)[i].ctrl_info),
                            sizeof(ItemInfo));
    if (mem_ret != EOK) {
      (void)rtMbufFree(*buff);
      MS_LOG(ERROR) << "call memcpy_s failed, ret =" << mem_ret;
      RETURN_STATUS_UNEXPECTED("SerializeDataItemInfo:memcpy_s failed.");
    }
    offset += sizeof(ItemInfo);

    for (size_t j = 0UL; j < (*items)[i].ctrl_info.dim_num; ++j) {
      mem_ret = memcpy_s(ge::ValueToPtr(ge::PtrToValue(data) + offset), sizeof(int64_t), &((*items)[i].dims[j]),
                         sizeof(int64_t));
      if (mem_ret != EOK) {
        (void)rtMbufFree(*buff);
        MS_LOG(ERROR) << "call memcpy_s failed, ret =" << mem_ret;
        RETURN_STATUS_UNEXPECTED("SerializeDataItemInfo:memcpy_s failed.");
      }
      offset += sizeof(int64_t);
    }

    if ((*items)[i].ctrl_info.data_len == 0UL) {
      continue;
    }

    mem_ret = memcpy_s(ge::ValueToPtr(ge::PtrToValue(data) + offset), (*items)[i].ctrl_info.data_len,
                       (*items)[i].data_ptr, (*items)[i].ctrl_info.data_len);
    if (mem_ret != EOK) {
      (void)rtMbufFree(*buff);
      MS_LOG(ERROR) << "call memcpy_s failed, ret =" << mem_ret;
      RETURN_STATUS_UNEXPECTED("SerializeDataItemInfo:memcpy_s failed.");
    }
    offset += (*items)[i].ctrl_info.data_len;
  }

  return Status::OK();
}

Status HostQueueImpl::LaunchTensor2MBuff(const acltdtTensorType &acl_type, const TensorRow &tensor_row, void **buff) {
  std::vector<DataItemInfo> items;
  RETURN_IF_NOT_OK(CreateDataItemInfos(acl_type, tensor_row, &items));
  RETURN_IF_NOT_OK(SerializeDataItemInfos(&items, buff, acl_type));
  RETURN_IF_NOT_OK(SetTransId4MBuf(buff));
  return Status::OK();
}

Status HostQueueImpl::SetTransId4MBuf(void **buff) {
  void *head_buff = nullptr;
  uint64_t head_size = 0UL;
  auto ret = rtMbufGetPrivInfo(*buff, &head_buff, &head_size);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtMbufGetPrivInfo failed, ret =" << ret;
    RETURN_STATUS_UNEXPECTED("HostQueueSetTransId:rtMbufGetPrivInfo failed.");
  }
  uint64_t *trans_id = reinterpret_cast<uint64_t *>(static_cast<uint8_t *>(head_buff) + head_size - kTransIdOffset);
  const std::lock_guard<std::mutex> lk(queue_id_to_trans_id_map_mutex);
  *trans_id = ++queue_id_to_trans_id_map[queue_id_];
  MS_LOG(DEBUG) << "host queue[" << queue_id_ << "] set trans id[" << *trans_id << "] success";
  return Status::OK();
}

Status HostQueueImpl::EnqueueData(void *buff, bool *need_resend) {
  *need_resend = false;
  auto rt_error = rtSetDevice(device_id_);
  if (rt_error != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtSetDevice device failed, ret=" << rt_error;
    RETURN_STATUS_UNEXPECTED("rtSetDevice failed.");
  }
  rt_error = rtMemQueueEnQueue(device_id_, queue_id_, buff);
  if (rt_error == RT_ERROR_NONE) {
    return Status::OK();
  } else if (rt_error == ACL_ERROR_RT_QUEUE_FULL) {
    *need_resend = true;
    MS_LOG(DEBUG) << "queue[" << queue_id_ << "] is full, need call rtMemQueueEnQueue again";
  } else {
    HostQueueFreeBuff(buff);
    MS_LOG(ERROR) << "host enqueue queue[" << queue_id_ << "] failed, ret = " << rt_error;
    RETURN_STATUS_UNEXPECTED("host enqueue queue failed.");
  }
  return Status::OK();
}

Status HostQueueImpl::SendDataByHostQueue(const TensorRow &tensor_row, const acltdtTensorType &data_type) {
  Status status;
  bool is_need_resend = false;
  void *buff = nullptr;
  // Status status;
  RETURN_IF_NOT_OK(LaunchTensor2MBuff(data_type, tensor_row, &buff));
  do {
    if (is_need_resend) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMilliSeconds));
    }
    status = EnqueueData(buff, &is_need_resend);
  } while (status == Status::OK() && is_need_resend);
  return Status::OK();
}

void HostQueueImpl::HostQueueFreeBuff(void *buff) {
  auto rt_error = rtMbufFree(buff);
  if (rt_error != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtMbufFree failed, ret=" << rt_error;
  }
}
}  // namespace dataset
}  // namespace mindspore
