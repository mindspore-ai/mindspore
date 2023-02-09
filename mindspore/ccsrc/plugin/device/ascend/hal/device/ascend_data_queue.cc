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

#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include <string>
#include <map>
#include <utility>
#include "graph/def_types.h"
#include "common/util/error_manager/error_manager.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "utils/log_adapter.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"

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

constexpr auto kUnknownErrorString = "Unknown error occurred";
std::vector<std::pair<void **, std::thread *>> g_acl_handle_map = {};

constexpr auto kMbufHeadEndOfSequencePos = 128U;
constexpr auto kEndOfSequenceFlag = 0x5A;
constexpr auto kTransIdOffset = 64UL;
constexpr auto kSleepMilliSeconds = 500;

std::mutex g_acl_destroy_all_mutex = {};
bool g_acl_destroy_all = false;

bool GetAclDataType(const std::string &str_type, aclDataType *acl_type) {
  MS_EXCEPTION_IF_NULL(acl_type);
  auto iter = kStringTypeToAclType.find(str_type);
  if (iter == kStringTypeToAclType.end()) {
    MS_LOG(EXCEPTION) << "Invalid type " << str_type;
  }
  *acl_type = iter->second;
  return true;
}

void CheckRtRetWithError(rtError_t error, const std::string &msg) {
  if (error != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Rt error: " << msg << " | Error number: " << error;
  }
}
}  // namespace

namespace tdt_handle {
void AddHandle(acltdtChannelHandle **handle, std::thread *use_thread) {
  void **void_handle = reinterpret_cast<void **>(handle);
  if (*handle == nullptr) {
    return;
  }

  for (auto iter = g_acl_handle_map.cbegin(); iter != g_acl_handle_map.cend(); ++iter) {
    if (iter->first == void_handle) {
      return;
    }
  }

  g_acl_handle_map.emplace_back(void_handle, use_thread);
  {
    std::lock_guard<std::mutex> lock(g_acl_destroy_all_mutex);
    g_acl_destroy_all = false;
  }
}

void DelHandle(acltdtChannelHandle **handle) {
  void **void_handle = reinterpret_cast<void **>(handle);
  for (auto iter = g_acl_handle_map.begin(); iter != g_acl_handle_map.end();) {
    if (iter->first == void_handle) {
      iter = g_acl_handle_map.erase(iter);
    } else {
      ++iter;
    }
  }
}

bool DestroyHandle() {
  std::lock_guard<std::mutex> lock(g_acl_destroy_all_mutex);
  bool destroy_all = true;
  for (auto &item : g_acl_handle_map) {
    acltdtChannelHandle **handle = reinterpret_cast<acltdtChannelHandle **>(item.first);
    if (*handle != nullptr) {
      aclError stop_status = acltdtStopChannel(*handle);
      if (stop_status != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Failed stop acl data channel and the stop status is " << stop_status << std::endl;
        return false;
      }
      if (item.second != nullptr && item.second->joinable()) {
        item.second->join();
      }
      if (acltdtDestroyChannel(*handle) != ACL_SUCCESS) {
        MS_LOG(INFO) << "acltdtDestroyChannel failed.";
        destroy_all = false;
      } else {
        *handle = nullptr;
      }
    }
  }

  // clear the map container when all the handle has been destroyed
  if (destroy_all) {
    g_acl_handle_map.clear();
    g_acl_destroy_all = true;
  }
  return destroy_all;
}

bool IsClosed() {
  std::lock_guard<std::mutex> lock(g_acl_destroy_all_mutex);
  return g_acl_destroy_all;
}
}  // namespace tdt_handle

AscendDataQueueDynamic::AscendDataQueueDynamic(const std::string &channel_name, const size_t capacity)
    : DataQueue(channel_name, capacity), stream_(nullptr), node_info_(nullptr) {
  auto context_key = device_context_->device_context_key();
  auto runtime_instance = dynamic_cast<ascend::AscendKernelRuntime *>(
    device::KernelRuntimeManager::Instance().GetKernelRuntime(context_key.device_name_, context_key.device_id_));
  node_info_ = std::make_unique<NodeInfo[]>(capacity);
  stream_ = runtime_instance->compute_stream();
}

DataQueueStatus AscendDataQueueDynamic::Push(std::vector<DataQueueItem> data) {
  for (size_t i = 0; i < data.size(); i++) {
    auto &item = data[i];
    if (item.data_ptr == nullptr) {
      MS_LOG(ERROR) << "Invalid Input: ptr: " << item.data_ptr << ", len: " << item.data_len;
      return DataQueueStatus::ERROR_INPUT;
    }
    void *addr = device_context_->device_res_manager_->AllocateMemory(item.data_len);
    if (addr == nullptr) {
      MS_LOG(ERROR) << "Allocate device memory of data queue failed";
    }
    CheckRtRetWithError(
      rtMemcpyAsync(addr, item.data_len, item.data_ptr, item.data_len, RT_MEMCPY_HOST_TO_DEVICE, stream_),
      "Rt Memcpy Error");
    item.device_addr = addr;
  }
  CheckRtRetWithError(rtStreamSynchronize(stream_), "Call runtime rtStreamSynchronize failed");
  node_info_[tail_].data_ = std::move(data);
  tail_ = (tail_ + 1) % (capacity_);
  ++size_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus AscendDataQueueDynamic::Front(std::vector<DataQueueItem> *data) const {
  for (auto &item : node_info_[head_].data_) {
    host_release_(item.data_ptr, item.worker_id);
  }
  *data = node_info_[head_].data_;
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus AscendDataQueueDynamic::Pop() {
  head_ = (head_ + 1) % (capacity_);
  --size_;
  return DataQueueStatus::SUCCESS;
}

AscendTdtQueue::AscendTdtQueue(const std::string &channel_name) : DataQueue(channel_name, 0), acl_handle_(nullptr) {
  // init ErrorManager, 0 means success
  if (ErrorManager::GetInstance().Init() != 0) {
    MS_LOG(WARNING) << "[Internal Error] Init ErrorManager failed.";
  }
  // get device id
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  device_id_ = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);

#if defined(ENABLE_PYTHON) && !defined(ENABLE_ANDROID)
  // There is a python flag in MindSpore to recognize if the runtime env is python.
  // If we only use MD feature, python_env_flag will not set to true,
  // then init.cc will not call ClearResAtexit at the end of MindSpore to clean resource.
  // The original case is [only MD + mbuf + device_queue + send], the ascend stream release
  // failed if we don't call ClearResAtexit first.
  mindspore::python_adapter::set_python_env_flag(true);
#endif

  // create acl tdt handle
  if (!channel_name_.empty()) {
    // When "capacity" is too large, device memory will be exploded
    size_t capacity = 16;
    std::string env_capacity_str = common::GetEnv("MS_DATASET_SINK_QUEUE");
    if (!env_capacity_str.empty()) {
      int32_t env_capacity = atoi(env_capacity_str.c_str());
      if (env_capacity <= 0) {
        MS_LOG(EXCEPTION) << "Invalid data queue capacity.#umsg#User Help Message:#umsg#"
                             "Expect env variable MS_DATASET_SINK_QUEUE > 0."
                          << ascend::GetErrorMessage(true);
      }
      capacity = env_capacity;
    }
    MS_LOG(INFO) << "The capacity of data queue is: " << capacity;
    acl_handle_ = acltdtCreateChannelWithCapacity(device_id_, channel_name_.c_str(), capacity);
    if (acl_handle_ == nullptr) {
      MS_LOG(INFO) << "Select the TDT process.";
      acl_handle_ = acltdtCreateChannel(device_id_, channel_name_.c_str());
      if (acl_handle_ == nullptr) {
        MS_LOG(EXCEPTION) << "Create channel for sending data failed.#umsg#User Help Message:#umsg#"
                             "Please check DEVICE ID setting, DEVICE ID that passed into dataset"
                             "(from context) and training process should be the same."
                          << ascend::GetErrorMessage(true);
      }
    }
    tdt_handle::AddHandle(&acl_handle_, nullptr);
  }

  // a wingman of tdt to help with transferring data shapes on host
  auto wingman_queue = std::make_shared<BlockingQueue>();
  std::shared_ptr<DataQueue> data_queue = std::make_shared<WingmanQueue>(channel_name);
  auto rt = wingman_queue->Create(data_queue);
  if (rt != DataQueueStatus::SUCCESS) {
    MS_LOG(EXCEPTION) << "Wingman queue: " << channel_name << "create failed: " << rt;
  }
  DataQueueMgr::GetInstance().Manage(channel_name, wingman_queue);
}

AscendTdtQueue::~AscendTdtQueue() {
  if (acl_handle_ != nullptr) {
    if (acltdtDestroyChannel(acl_handle_) != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Failed to destroy channel for tdt queue." << ascend::GetErrorMessage(true);
    } else {
      tdt_handle::DelHandle(&acl_handle_);
      acl_handle_ = nullptr;
    }
  }
  if (DataQueueMgr::GetInstance().IsCreated(channel_name_)) {
    DataQueueMgr::GetInstance().Free(channel_name_);
  }
}

bool AscendTdtQueue::IsOpen() const { return !tdt_handle::IsClosed(); }

DataQueueStatus AscendTdtQueue::Push(std::vector<DataQueueItem> data) {
  MS_LOG(DEBUG) << "TDT channel name is " << channel_name_ << ".";
  acltdtDataset *acl_dataset = nullptr;
  auto ret = Translate(data, &acl_dataset);
  if (!ret) {
    DestroyAclDataset(acl_dataset);
    MS_LOG(ERROR) << "Converting into TDT tensor failed!";
    return DataQueueStatus::INTERNAL_ERROR;
  }

  // Data prefetch only when PS mode enables cache.
  if (acltdtGetDatasetSize(acl_dataset) > 0) {
    acltdtDataItem *item0 = acltdtGetDataItem(acl_dataset, 0);
    std::string item_type;
    ParseType(acltdtGetDataTypeFromItem(item0), &item_type);
    if (!ps::PsDataPrefetch::GetInstance().PrefetchData(channel_name_, acltdtGetDataAddrFromItem(item0),
                                                        acltdtGetDataSizeFromItem(item0), item_type)) {
      MS_LOG(ERROR) << "PrefetchData failed in when pre-processing sending data.";
      return DataQueueStatus::INTERNAL_ERROR;
    }
  }
  auto status = acltdtSendTensor(acl_handle_, acl_dataset, -1);
  DestroyAclDataset(acl_dataset);
  if (status != ACL_SUCCESS) {
    // if the device_queue thread had been interrupted by master, just print warning and return success
    if (tdt_handle::IsClosed()) {
      MS_LOG(WARNING) << "Device queue thread had been interrupted by TdtHandle::DestroyHandle, you can ignore "
                      << "the above error: 'failed to send...'. In this scenario, the training ends first without "
                      << "using all epoch(s) data, and the data preprocessing is blocked by the data "
                      << "transmission channel on the device side. So we force the data transmission channel to stop.";
      return DataQueueStatus::SUCCESS;
    }
    MS_LOG(EXCEPTION) << "Tdt Send data failed." << ascend::GetErrorMessage(true);
  }
  auto wingman = DataQueueMgr::GetInstance().GetDataQueue(channel_name_);
  if (wingman != nullptr && wingman->IsOpen() && !data.empty()) {
    (void)wingman->Push(data);
  }
  return DataQueueStatus::SUCCESS;
}

void AscendTdtQueue::ParseType(aclDataType acl_data_type, std::string *data_type) const {
  auto type_iter = kAclTypeToString.find(acl_data_type);
  if (type_iter == kAclTypeToString.end()) {
    MS_LOG(EXCEPTION) << "Got unsupported acl datatype: " << acl_data_type;
  }
  *data_type = type_iter->second;
}

bool AscendTdtQueue::Translate(const std::vector<DataQueueItem> &data, acltdtDataset **output_acl_dataset) const {
  auto acl_dataset = acltdtCreateDataset();
  if (acl_dataset == nullptr) {
    MS_LOG(ERROR) << "Create tdt dataset failed.";
    return false;
  }
  bool status = AssembleTensor2AclDataset(data, acl_dataset);
  if (!status) {
    DestroyAclDataset(acl_dataset);
    MS_LOG(ERROR) << "Assemble tensor row to tdt dataset failed.";
    return false;
  }

  *output_acl_dataset = acl_dataset;
  return true;
}

bool AscendTdtQueue::AssembleTensor2AclDataset(const std::vector<DataQueueItem> &data,
                                               acltdtDataset *acl_dataset) const {
  if (data.empty()) {
    acltdtDataItem *acl_data =
      acltdtCreateDataItem(acltdtTensorType::ACL_TENSOR_DATA_END_OF_SEQUENCE, nullptr, 0, ACL_BOOL, nullptr, 0);
    if (acl_data == nullptr) {
      MS_LOG(ERROR) << "Create data item failed when send empty data.";
      return false;
    }
    if (acltdtAddDataItem(acl_dataset, acl_data) != ACL_SUCCESS) {
      if (acltdtDestroyDataItem(acl_data) != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Destroy data item failed when send empty data.";
      }
      MS_LOG(ERROR) << "Add data item to tdt dataset failed when send data.";
      return false;
    }
    return true;
  }

  for (const auto &ts : data) {
    aclDataType acl_type;
    acltdtDataItem *acl_data = nullptr;
    if (!GetAclDataType(ts.data_type, &acl_type)) {
      MS_LOG(ERROR) << "Convert type " << ts.data_type << " to acl type failed.";
      return false;
    }

    const auto &shape = ts.shapes;
    std::string shape_str = "[";
    for (auto dim : shape) {
      (void)shape_str.append(std::to_string(dim)).append(",");
    }
    shape_str.pop_back();
    (void)shape_str.append("]");

    void *data_ptr = ts.data_ptr;
    size_t data_size = ts.data_len;

    acl_data = acltdtCreateDataItem(acltdtTensorType::ACL_TENSOR_DATA_TENSOR, (shape.empty() ? nullptr : &shape[0]),
                                    shape.size(), acl_type, data_ptr, data_size);
    if (acl_data == nullptr) {
      MS_LOG(ERROR) << "Create data item failed when send data.";
      return false;
    }
    if (acltdtAddDataItem(acl_dataset, acl_data) != ACL_SUCCESS) {
      if (acltdtDestroyDataItem(acl_data) != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Destroy data item failed when send data with type ACL_TENSOR_DATA_TENSOR.";
      }
      MS_LOG(INFO) << "Add data item to tdt dataset failed when send data.";
      return false;
    }

    MS_LOG(DEBUG) << "TDT data type is TDT_TENSOR, tensor type is " << acl_type << ", tensor shape is " << shape_str
                  << ", data length is " << data_size << ".";
  }

  return true;
}

void AscendTdtQueue::DestroyAclDataset(acltdtDataset *acl_dataset, bool include_data_item) const {
  if (include_data_item) {
    for (size_t i = 0; i < acltdtGetDatasetSize(acl_dataset); i++) {
      if (acltdtDestroyDataItem(acltdtGetDataItem(acl_dataset, i)) != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "Destroy data item failed when send data.";
      }
    }
  }

  if (acltdtDestroyDataset(acl_dataset) != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Destroy tdt dataset failed when send data.";
  }
}

AscendHostQueue::AscendHostQueue(const std::string &channel_name)
    : DataQueue(channel_name, 0), queue_id_to_trans_id_map_(), queue_id_(0) {
  // init ErrorManager, 0 means success
  if (ErrorManager::GetInstance().Init() != 0) {
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
  auto rt_ret = rtSetDevice(device_id_);
  if (rt_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtSetDevice failed, ret = " << rt_ret;
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
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtMemQueueCreate failed, ret = " << rt_ret;
    return false;
  }

  rtMemBuffCfg_t buff_cfg = {};
  rt_ret = rtMbufInit(&buff_cfg);
  if (rt_ret != RT_ERROR_NONE && rt_ret != ACL_ERROR_RT_REPEATED_INIT) {
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
  auto rt_error = rtSetDevice(device_id_);
  if (rt_error != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "call rtSetDevice device failed, ret=" << rt_error;
    return false;
  }
  rt_error = rtMemQueueEnQueue(device_id_, queue_id_, buff);
  if (rt_error == RT_ERROR_NONE) {
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
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtMbufFree failed, ret = " << rt_ret;
  }
}

DataQueueStatus WingmanQueue::Push(const std::vector<DataQueueItem> data) {
  queue_.emplace(data);
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus WingmanQueue::Pop() {
  queue_.pop();
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus WingmanQueue::Front(std::vector<DataQueueItem> *data) const {
  *data = queue_.front();
  return DataQueueStatus::SUCCESS;
}

DataQueueStatus WingmanQueue::FrontAsync(std::vector<DataQueueItem> *data) const { return this->Front(data); }

void WingmanQueue::Close() {
  queue_ = {};
  closed_ = true;
}

std::shared_ptr<BlockingQueue> GetTdtWingManQueue(const std::shared_ptr<AnfNode> &node) {
  if (common::AnfAlgo::GetCNodeName(node) != kGetNextOpName) return nullptr;
  auto queue_name = common::AnfAlgo::GetNodeAttr<std::string>(node, "shared_name");
  if (!DataQueueMgr::GetInstance().IsCreated(queue_name)) {
    return nullptr;
  }
  return DataQueueMgr::GetInstance().GetDataQueue(queue_name);
}

void CloseTdtWingManQueue(const std::shared_ptr<AnfNode> &node) {
  if (common::AnfAlgo::GetCNodeName(node) != kGetNextOpName) return;
  auto wingman = GetTdtWingManQueue(node);
  if (wingman && wingman->IsOpen()) {
    wingman->Close();
  }
}

namespace {
std::shared_ptr<DataQueue> CreateAscendDataQueue(const std::string &channel_name, bool, size_t,
                                                 const std::vector<size_t> &) {
  int32_t is_heterogeneous = 0;
  (void)rtGetIsHeterogenous(&is_heterogeneous);
  if (is_heterogeneous != 0) {
    return std::make_shared<AscendHostQueue>(channel_name);
  } else {
    return std::make_shared<AscendTdtQueue>(channel_name);
  }
}

REGISTER_DATA_QUEUE_CREATOR(kAscendDevice, CreateAscendDataQueue);
struct DevicePlugFuncRegister {
  DevicePlugFuncRegister() noexcept {
    DataQueueMgr::SetDestoryTdtHandleHandler([]() -> bool { return tdt_handle::DestroyHandle(); });
  }
} ascend_device_func_register;
}  // namespace
}  // namespace device
}  // namespace mindspore
