/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/device_queue_op.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
DeviceQueueOp::DeviceQueueOp(std::string channel_name, DeviceType device_type, int32_t device_id, int32_t prefetch_size,
                             bool send_epoch_end, int32_t total_batch, bool create_data_info_queue)
    : PipelineOp(1),
      channel_name_(channel_name),
      device_type_(device_type),
      device_id_(device_id),
      prefetch_size_(prefetch_size),
      send_epoch_end_(send_epoch_end),
      stop_send_(false),
      send_finished_(false),
      total_batch_(total_batch),
      create_data_info_queue_(create_data_info_queue),
      data_info_queue_ptr_(nullptr),
      first_fetch_flag_(false),
      first_push_flag_(false) {
#ifdef ENABLE_GPUQUE
  // Get the total device num of current machine
  int32_t device_count = 0;
  cudaGetDeviceCount(&device_count);
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  rank_id_ = cfg->rank_id();  // Get the current rank_id
  if (device_count > 0) {
    rank_id_ = rank_id_ % device_count;
  }
  // Be careful when try to modified these num_workers_ and queue_capacity_,
  // and we suggest num_workers_ * queue_capacity_ not greater than 16, because
  // one worker one circular_pool with 1G pin memory, so num_workers_ * queue_capacity_
  // must limit to avoid memory overload
  num_workers_ = kDeviceQueGpuNumThreads;
  queue_capacity_ = kDeviceQueGpuQueueCapacity;
#endif
#ifdef ENABLE_TDTQUE
  ascend_keep_waiting_ = true;
  tdtInstancePtr = std::make_shared<TdtPlugin>(channel_name_, device_id_);
#endif
#ifdef ENABLE_DUMP_IR
  md_channel_info_ = std::make_shared<MDChannelInfo>(channel_name_);
#endif
}

DeviceQueueOp::~DeviceQueueOp() {
#ifdef ENABLE_DUMP_IR
  std::string rdr_msg = md_channel_info_->ToString();
  if (!send_finished_ && !rdr_msg.empty()) {
    MS_LOG(WARNING) << rdr_msg;
  }
#endif
}

#ifdef ENABLE_GPUQUE
void DeviceQueueOp::ReleaseData(void *addr, int32_t worker_id) {
  if (addr != nullptr) {
    pool_[worker_id]->Deallocate(addr);
  }
}
#endif

Status DeviceQueueOp::EoeReceived(int32_t worker_id) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

Status DeviceQueueOp::FilterMetadata(TensorRow *row) {
  std::unordered_map<std::string, int32_t> current_name_id_map = child_[0]->column_name_id_map();
  TensorRow output;
  TensorRow tmp = *row;
  std::vector<size_t> to_keep_indices;
  for (auto column : current_name_id_map) {
    std::string column_name = column.first;
    // Need to filter meta column start with kDftMetaColumnPrefix
    size_t pos = column_name.find(kDftMetaColumnPrefix);
    if (pos != std::string::npos && pos == 0) {
      continue;
    }
    to_keep_indices.push_back(column.second);
  }
  if (to_keep_indices.size() == 0) {
    std::string err_msg = "No effective column found, maybe all columns are meta column and will be filtered. ";
    err_msg += "If you want to output meta column please rename column name to a new one which is not start with ";
    err_msg += "\"" + std::string(kDftMetaColumnPrefix) + "\"";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::sort(to_keep_indices.begin(), to_keep_indices.end());
  (void)std::transform(to_keep_indices.begin(), to_keep_indices.end(), std::back_inserter(output),
                       [&tmp](const auto &it) { return std::move(tmp[it]); });
  *row = std::move(output);
  return Status::OK();
}

Status DeviceQueueOp::CheckExceptions(const TensorRow &row) const {
  // this method checks if the row meets the conditions to be sent to TDT
  for (const auto &item : row) {
    CHECK_FAIL_RETURN_UNEXPECTED(item->type().IsNumeric(), "Invalid data, cannot send string tensor to device.");
    CHECK_FAIL_RETURN_UNEXPECTED(item->HasData(), "Invalid data, cannot send tensor with no data to device.");
  }
  return Status::OK();
}

Status DeviceQueueOp::operator()() {
#ifndef ENABLE_SECURITY
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(
    "Detect first batch", std::bind(&DeviceQueueOp::DetectFirstBatch, this), nullptr, id()));
#endif
  TaskManager::FindMe()->Post();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);

#ifdef ENABLE_DUMP_IR
  if (md_channel_info_ == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "[Internal ERROR] RDR module init failed.");
  }
#endif
  if (device_type_ == DeviceType::Ascend) {
#ifdef ENABLE_TDTQUE
    if (create_data_info_queue_) {
      // This place has a race condition with GetDataInfo, so the first one
      // arrive here will do the initialize work.
      {
        std::unique_lock<std::mutex> lock(data_info_mutex_);
        if (data_info_queue_ptr_ == nullptr) {
          data_info_queue_ptr_ = std::make_unique<DATA_INFO_QUEUE>(kDataInfoQueueCapacity);
          RETURN_IF_NOT_OK(data_info_queue_ptr_->Register(tree_->AllTasks()));
        }
      }
    }
    if (tdtInstancePtr->acl_handle_ == nullptr) {
      RETURN_STATUS_UNEXPECTED("Create channel for sending data failed, please check DEVICE ID setting.");
    }
    RETURN_IF_NOT_OK(SendDataToAscend());
#endif
  } else if (device_type_ == DeviceType::GPU) {
#ifdef ENABLE_GPUQUE
    RETURN_IF_NOT_OK(SendDataToGPU());
#endif
  } else if (device_type_ == DeviceType::CPU) {
    RETURN_IF_NOT_OK(SendDataToCPU());
  }

  return Status::OK();
}

#ifdef ENABLE_TDTQUE
Status DeviceQueueOp::SendDataToAscend() {
  MS_LOG(INFO) << "Device queue, sending data to Ascend.";
#ifndef ENABLE_SECURITY
  uint64_t batch_start_time = 0;
  uint64_t end_time = 0;
  uint64_t batch_record_start = 0;
  uint64_t batch_record_end = 0;
#endif
  int64_t send_batch = 0;
  int32_t tdt_cost = 0;
#ifndef ENABLE_SECURITY
  int32_t connector_size = 0;
  int32_t connector_capacity = 0;
#endif
  bool is_break_loop = false;

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int64_t sending_num = cfg->sending_batches();  // Get the current sending_num

#ifndef ENABLE_SECURITY
  std::shared_ptr<DeviceQueueTracing> profiling_node;
  bool is_profiling_enable = tree_->GetProfilingManager()->IsProfilingEnable();
  if (is_profiling_enable) {
    std::shared_ptr<Tracing> node;
    RETURN_IF_NOT_OK(tree_->GetProfilingManager()->GetTracingNode(kDeviceQueueTracingName, &node));
    profiling_node = std::dynamic_pointer_cast<DeviceQueueTracing>(node);
    batch_start_time = ProfilingTime::GetCurMilliSecond();
    connector_capacity = ChildOpConnectorCapacity();
  }
#else
  bool is_profiling_enable = false;
#endif
#ifdef ENABLE_DUMP_IR
  md_channel_info_->RecordBatchQueue(ChildOpConnectorSize());
  md_channel_info_->RecordPreprocessBatch(0);
#endif
#ifndef ENABLE_SECURITY
  batch_record_start = ProfilingTime::GetCurMilliSecond();
#endif
  TensorRow curr_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
  first_fetch_flag_ = true;
  while (!curr_row.eof() && !is_break_loop) {
    while (!curr_row.eoe() && !is_break_loop) {
      RETURN_IF_NOT_OK(FilterMetadata(&curr_row));
      RETURN_IF_NOT_OK(CheckExceptions(curr_row));
      WaitContinueSignal();
#ifdef ENABLE_DUMP_IR
      md_channel_info_->RecordBatchQueue(ChildOpConnectorSize());
      md_channel_info_->RecordPreprocessBatch(send_batch);
      md_channel_info_->RecordPushStartTime();
#endif
#ifndef ENABLE_SECURITY
      DetectPerBatchTime(&batch_record_start, &batch_record_end);
#endif
      PrintBeginInfoWhenFirstBatch(first_push_flag_);
      RETURN_IF_NOT_OK(SendRowToTdt(curr_row, is_profiling_enable, &tdt_cost));
      PrintEndInfoWhenFirstBatch(&first_push_flag_);
#ifndef ENABLE_SECURITY
      ProfilingRecorder(is_profiling_enable, profiling_node, send_batch, tdt_cost, &batch_start_time, &end_time,
                        connector_capacity, connector_size);
      batch_record_start = ProfilingTime::GetCurMilliSecond();
#endif
      send_batch++;
#ifdef ENABLE_DUMP_IR
      md_channel_info_->RecordBatchQueue(ChildOpConnectorSize());
      md_channel_info_->RecordPreprocessBatch(send_batch);
      md_channel_info_->RecordPushEndTime();
#endif

      if (total_batch_ > 0 && send_batch >= total_batch_) {
        is_break_loop = true;
        break;
      }

      // wait when sending num is not 0, and sending num no larger than already sending batch
      LimitSendingBatches(send_batch, &sending_num, cfg);

#ifndef ENABLE_SECURITY
      if (is_profiling_enable) {
        connector_size = ChildOpConnectorSize();
        connector_capacity = ChildOpConnectorCapacity();
      }
#endif
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
    }
    if (curr_row.eoe() && send_epoch_end_) {
      TensorRow dummy_row;
      auto status = tdtInstancePtr->hostPush(dummy_row, true, channel_name_, is_profiling_enable, tdt_cost,
                                             ACL_TENSOR_DATA_END_OF_SEQUENCE);
      if (status != Status::OK()) {
        if (stop_send_) {
          send_finished_ = true;
          MS_LOG(INFO) << "stop_send received";
          return Status::OK();
        }
        return Status(StatusCode::kMDTDTPushFailure,
                      "TDT Push data into device Failed, check the first error or TraceBack first, following are"
                      " several possible checking way: 1) if training is not ready, still in network graph compiling"
                      " stage, check error raised by Network used operator or environment configuration. 2) if"
                      " interrupt in middle process of training, may check whether dataset sending num and network"
                      " training num mismatch. 3) if this error raised in end of training, ignore this. 4) other cases,"
                      " try find ascend host log or checking info log etc or search this in mindspore's FAQ.");
      }
      MS_LOG(INFO) << "an epoch has already sent, now stop send data.";
      stop_send_ = true;
    }
#ifndef ENABLE_SECURITY
    if (is_profiling_enable) {
      connector_size = ChildOpConnectorSize();
      connector_capacity = ChildOpConnectorCapacity();
      tree_->SetEpochEnd();
    }
#endif
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
  }

  // now we use this flag to judge whether exception raised.
  if (stop_send_ || !TaskManager::FindMe()->Interrupted()) {
    send_finished_ = true;
  }
  tree_->SetFinished();
  MS_LOG(INFO) << "Device queue send " << send_batch << " batch.";

  return Status::OK();
}

void DeviceQueueOp::WaitContinueSignal() const {
  while (stop_send_ && ascend_keep_waiting_) {
    MS_LOG(DEBUG) << "stop_send flag is set, waiting for continue signal...";
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

void DeviceQueueOp::LimitSendingBatches(int64_t send_batch, int64_t *sending_num, std::shared_ptr<ConfigManager> cfg) {
  while (send_batch >= *sending_num) {
    *sending_num = cfg->sending_batches();
    if (*sending_num == 0) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    MS_LOG(INFO) << "Wait for 10 milliseconds, as needed send batch is: " << *sending_num
                 << ", and current sending batch is:" << send_batch;
  }
}

Status DeviceQueueOp::SendRowToTdt(TensorRow curr_row, bool is_profiling_enable, int32_t *tdt_cost) {
  auto status = tdtInstancePtr->hostPush(curr_row, true, channel_name_, is_profiling_enable, *tdt_cost);
  if (status != Status::OK()) {
    if (stop_send_) {
      MS_LOG(INFO) << "stop_send received";
      return Status::OK();
    }
    return Status(StatusCode::kMDTDTPushFailure,
                  "TDT Push data into device Failed, check the first error or TraceBack first, following are"
                  " several possible checking way: 1) if training is not ready, still in network graph compiling"
                  " stage, check error raised by Network used operator or environment configuration. 2) if"
                  " interrupt in middle process of training, may check whether dataset sending num and network"
                  " training num mismatch. 3) if this error raised in end of training, ignore this. 4) other cases,"
                  " try find ascend host log or checking info log ects or search this in mindspore's FAQ.");
  }
  if (create_data_info_queue_) {
    DATA_INFO data_info;
    (void)std::transform(curr_row.begin(), curr_row.end(), std::back_inserter(data_info),
                         [](const std::shared_ptr<Tensor> &ts) { return std::make_pair(ts->type(), ts->shape()); });
    RETURN_IF_NOT_OK(data_info_queue_ptr_->Add(data_info));
  }
  return Status::OK();
}
#endif

#ifdef ENABLE_TDTQUE
Status DeviceQueueOp::GetDataInfo(DATA_INFO *data_info) {
  if (!create_data_info_queue_) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "DataInfo queue is not created.");
  }
  // This place has a race condition with operator(), so the first one
  // arrive here will do the initialize work.
  {
    std::unique_lock<std::mutex> lock(data_info_mutex_);
    if (data_info_queue_ptr_ == nullptr) {
      data_info_queue_ptr_ = std::make_unique<DATA_INFO_QUEUE>(kDataInfoQueueCapacity);
      RETURN_IF_NOT_OK(data_info_queue_ptr_->Register(tree_->AllTasks()));
    }
  }
  RETURN_IF_NOT_OK(data_info_queue_ptr_->PopFront(data_info));
  return Status::OK();
}
#else
Status DeviceQueueOp::GetDataInfo(DATA_INFO *data_info) {
  return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "GetDataInfo is not supported yet.");
}
#endif

#ifdef ENABLE_GPUQUE
Status DeviceQueueOp::SetThreadDevice() {
  // Without cudaSetDevice cuda memory will allocate on GPU:0 as default
  // and will overload in distribute scenario.
  auto ret = cudaSetDevice(rank_id_);
  if (ret != cudaSuccess) {
    std::string err;
    err += "cudaSetDevice failed, ret[";
    err += std::to_string(static_cast<int>(ret));
    err += "], ";
    err += cudaGetErrorString(ret);
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err);
  }
  return Status::OK();
}

Status DeviceQueueOp::LaunchParallelCopyThread() {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  // Every thread use cuda api should SetThreadDevice
  RETURN_IF_NOT_OK(SetThreadDevice());
  // CircularPool may not safe under multi-threads scenario, so one worker with one pool
  for (int i = 0; i < num_workers_; i++) {
    std::shared_ptr<MemoryPool> pool;
    RETURN_IF_NOT_OK(CircularPool::CreateCircularPool(&pool, -1, kDeviceQueGpuThreadMemory, false, true));
    pool_.push_back(pool);
  }
  gpu_item_connector_ = std::make_unique<GpuItemConnector>(num_workers_, 1, queue_capacity_);
  receive_queues_.Init(num_workers_, queue_capacity_);
  RETURN_IF_NOT_OK(receive_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&DeviceQueueOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask("Push data to GPU queue",
                                                      std::bind(&DeviceQueueOp::PushDataToGPU, this), nullptr, id()));

  return Status::OK();
}

Status DeviceQueueOp::PushDataToGPU() {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  // Every thread use cuda api should SetThreadDevice
  RETURN_IF_NOT_OK(SetThreadDevice());
  TaskManager::FindMe()->Post();
#ifndef ENABLE_SECURITY
  uint64_t batch_start_time = 0;
  int32_t push_cost = 0;
  int32_t connector_size = 0;
  int32_t connector_capacity = 0;
  std::shared_ptr<DeviceQueueTracing> profiling_node;
  bool is_profiling_enable = tree_->GetProfilingManager()->IsProfilingEnable();
  if (is_profiling_enable) {
    std::shared_ptr<Tracing> node;
    RETURN_IF_NOT_OK(tree_->GetProfilingManager()->GetTracingNode(kDeviceQueueTracingName, &node));
    profiling_node = std::dynamic_pointer_cast<DeviceQueueTracing>(node);
    batch_start_time = ProfilingTime::GetCurMilliSecond();
    connector_capacity = gpu_item_connector_->capacity();
  }
#endif
#ifdef ENABLE_DUMP_IR
  md_channel_info_->RecordBatchQueue(gpu_item_connector_->size());
  md_channel_info_->RecordPreprocessBatch(0);
#endif
  std::vector<device::DataItemGpu> items;
  RETURN_IF_NOT_OK(gpu_item_connector_->Pop(0, &items));
  int64_t send_batch = 0;
  bool is_open = false;
  uint32_t handle = INVALID_HANDLE;
  auto release_function = std::bind(&DeviceQueueOp::ReleaseData, this, std::placeholders::_1, std::placeholders::_2);
  while (!items.empty() && !GpuBufferMgr::GetInstance().IsClosed()) {
#ifdef ENABLE_DUMP_IR
    md_channel_info_->RecordBatchQueue(gpu_item_connector_->size());
    md_channel_info_->RecordPreprocessBatch(send_batch);
    md_channel_info_->RecordPushStartTime();
#endif
    if (!is_open) {
      std::vector<size_t> data_size;
      for (int32_t index = 0; index < items.size(); index++) {
        data_size.push_back(items[index].data_len_);
      }
      handle = GpuBufferMgr::GetInstance().Open(0, channel_name_, data_size, release_function);
      if (handle == INVALID_HANDLE) {
        return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                      "[Internal ERROR] Failed to open channel for sending data.");
      }
      is_open = true;
    }

    // Data prefetch only when PS mode enables cache.
    if (!ps::PsDataPrefetch::GetInstance().PrefetchData(channel_name_, items[0].data_ptr_, items[0].data_len_,
                                                        items[0].data_type_)) {
      return Status(StatusCode::kMDTimeOut, __LINE__, __FILE__,
                    "Failed to prefetch data in current PS mode(cache data when sending).");
    }
    RETURN_IF_NOT_OK(RetryPushData(handle, items));
    send_batch++;
#ifndef ENABLE_SECURITY
    if (is_profiling_enable) {
      uint64_t end_time = ProfilingTime::GetCurMilliSecond();
      // record push data time
      profiling_node->Record(TIME, TDT_PUSH_TIME, send_batch, push_cost, end_time);
      int32_t batch_cost = (int32_t)(end_time - batch_start_time);
      // record batch time
      profiling_node->Record(TIME, BATCH_TIME, send_batch, batch_cost, end_time);
      // record pipeline time
      profiling_node->Record(TIME, PIPELINE_TIME, send_batch, batch_cost - push_cost, end_time);
      batch_start_time = end_time;
      // record connector depth
      profiling_node->Record(CONNECTOR_DEPTH, connector_capacity, send_batch, connector_size, end_time);
      connector_size = gpu_item_connector_->size();
      connector_capacity = gpu_item_connector_->capacity();
    }
#endif
#ifdef ENABLE_DUMP_IR
    md_channel_info_->RecordBatchQueue(gpu_item_connector_->size());
    md_channel_info_->RecordPreprocessBatch(send_batch);
    md_channel_info_->RecordPushEndTime();
#endif
    if (total_batch_ > 0 && send_batch >= total_batch_) {
      break;
    }
    if (!TaskManager::FindMe()->Interrupted() && !GpuBufferMgr::GetInstance().IsClosed()) {
      auto rc = gpu_item_connector_->Pop(0, &items);
      // If the batches send by dataset are more than gpu calculate, gpu will core for no signal notify.
      if (rc.IsError()) {
        GpuBufferMgr::GetInstance().Close(handle);
        GpuBufferMgr::GetInstance().CloseConfirm();
        return rc;
      }
    } else {
      break;
    }
  }

  // now we use this flag to judge whether exception raised.
  if (!TaskManager::FindMe()->Interrupted() && !GpuBufferMgr::GetInstance().IsClosed()) {
    send_finished_ = true;
  }
  tree_->SetFinished();
  MS_LOG(INFO) << "Device queue send " << send_batch << " batch.";

  GpuBufferMgr::GetInstance().Close(handle);
  GpuBufferMgr::GetInstance().CloseConfirm();
  return Status::OK();
}

Status DeviceQueueOp::RetryPushData(unsigned int handle, const std::vector<DataItemGpu> &items) {
  bool flag_log = false;
  while (!GpuBufferMgr::GetInstance().IsClosed() && !TaskManager::FindMe()->Interrupted()) {
    BlockQueueStatus_T ret = GpuBufferMgr::GetInstance().Push(handle, items, WAIT_TIME);
    if (ret) {
      if (ret == BlockQueueStatus_T::ERROR_INPUT) {
        return Status(
          StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
          "Invalid data, the types or shapes of current row is different with previous row(i.e. do batch operation but "
          "drop_reminder is False, or without resize image into the same size, these will cause shapes differs).");
      } else {
        if (!stop_send_) {
          if (!flag_log) {
            MS_LOG(DEBUG) << "Retry pushing data...";
            flag_log = true;
          }
          continue;
        }
        break;
      }
    } else {
      break;
    }
  }
  return Status::OK();
}

// WorkEntry of DeviceQueueOp just do multi_threads memcpy for performance optimization.
Status DeviceQueueOp::WorkerEntry(int32_t worker_id) {
  // Every thread use cuda api should SetThreadDevice
  RETURN_IF_NOT_OK(SetThreadDevice());
  TaskManager::FindMe()->Post();
  TensorRow current_row;
  uint32_t batch_num = 0;
  RETURN_IF_NOT_OK(receive_queues_[worker_id]->PopFront(&current_row));
  while (!current_row.quit() && !GpuBufferMgr::GetInstance().IsClosed()) {
    std::vector<device::DataItemGpu> items;
    for (int i = 0; i < current_row.size(); i++) {
      device::DataItemGpu data_item;
      data_item.data_len_ = static_cast<size_t>(current_row[i]->SizeInBytes());
      data_item.data_ptr_ = nullptr;
      data_item.worker_id_ = worker_id;
      items.push_back(data_item);
    }
    RETURN_IF_NOT_OK(MallocForGPUData(&items, current_row, worker_id));
    RETURN_IF_NOT_OK(gpu_item_connector_->Add(worker_id, std::move(items)));
    batch_num++;

    RETURN_IF_NOT_OK(receive_queues_[worker_id]->PopFront(&current_row));
  }

  MS_LOG(INFO) << "Device queue worker id " << worker_id << "proc " << batch_num << "batch.";
  // Add empty vector as quit flag.
  std::vector<device::DataItemGpu> items;
  RETURN_IF_NOT_OK(gpu_item_connector_->Add(worker_id, std::move(items)));
  return Status::OK();
}

Status DeviceQueueOp::SendDataToGPU() {
  RETURN_IF_NOT_OK(LaunchParallelCopyThread());
  MS_LOG(INFO) << "Device queue, sending data to GPU.";
#ifndef ENABLE_SECURITY
  uint64_t batch_record_start, batch_record_end;
  batch_record_start = ProfilingTime::GetCurMilliSecond();
#endif
  TensorRow current_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
  first_fetch_flag_ = true;
  int64_t num_buf = 0;
  bool is_break_loop = false;
  while (!current_row.eof() && !is_break_loop && !GpuBufferMgr::GetInstance().IsClosed()) {
    while (!current_row.eoe() && !is_break_loop && !GpuBufferMgr::GetInstance().IsClosed()) {
      RETURN_IF_NOT_OK(FilterMetadata(&current_row));
      RETURN_IF_NOT_OK(CheckExceptions(current_row));
#ifndef ENABLE_SECURITY
      DetectPerBatchTime(&batch_record_start, &batch_record_end);
#endif
      PrintBeginInfoWhenFirstBatch(first_push_flag_);
      RETURN_IF_NOT_OK(receive_queues_[num_buf++ % num_workers_]->Add(std::move(current_row)));
      PrintEndInfoWhenFirstBatch(&first_push_flag_);
#ifndef ENABLE_SECURITY
      batch_record_start = ProfilingTime::GetCurMilliSecond();
#endif
      if (!TaskManager::FindMe()->Interrupted() && !GpuBufferMgr::GetInstance().IsClosed()) {
        RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
      } else {
        is_break_loop = true;
      }
    }

    if (!TaskManager::FindMe()->Interrupted() && !GpuBufferMgr::GetInstance().IsClosed()) {
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
    } else {
      is_break_loop = true;
    }
  }

  for (uint32_t index = 0; index < num_workers_; index++) {
    TensorRow quit_flag(TensorRow::kFlagQuit);
    RETURN_IF_NOT_OK(receive_queues_[num_buf++ % num_workers_]->Add(std::move(quit_flag)));
  }

  MS_LOG(INFO) << "Device queue receive " << num_buf - num_workers_ << " batch.";
  return Status::OK();
}

Status DeviceQueueOp::MallocForGPUData(std::vector<device::DataItemGpu> *items, const TensorRow &curr_row,
                                       const int32_t &worker_id) {
  int i = 0;
  for (auto &sub_item : *items) {
    auto rc = pool_[worker_id]->Allocate(sub_item.data_len_, &sub_item.data_ptr_);
    if (rc.IsError() || sub_item.data_ptr_ == nullptr) {
      return Status(StatusCode::kMDOutOfMemory, __LINE__, __FILE__, "Memory malloc failed.");
    }
    if (curr_row[i] == nullptr) {
      MS_LOG(ERROR) << "The pointer curr_row[" << i << "] is null";
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "TensorRow 'curr_row' contains nullptr.");
    }
    sub_item.data_type_ = curr_row[i]->type().ToString();
    const unsigned char *column_data = curr_row[i]->GetBuffer();
    if (memcpy_s(sub_item.data_ptr_, sub_item.data_len_, column_data,
                 static_cast<uint32_t>(curr_row[i++]->SizeInBytes())) != 0) {
      MS_LOG(ERROR) << "memcpy_s failed!";
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "memcpy failed when using memcpy_s do copy.");
    }
  }

  return Status::OK();
}
#endif

Status DeviceQueueOp::SendDataToCPU() {
  MS_LOG(INFO) << "Device queue, sending data to CPU.";
  int64_t total_batch = 0;

  while (!(child_iterator_->EofHandled())) {
    TensorRow curr_row;
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));

    if (!first_fetch_flag_) {
      first_fetch_flag_ = true;
    }
    if (!curr_row.empty()) {
      for (auto &tensor : curr_row) {
        MS_LOG(DEBUG) << "Feature size is " << tensor->SizeInBytes() << ".";
      }
      total_batch++;
      if (stop_send_) break;
    }
  }

  MS_LOG(INFO) << "Device queue total batch is " << total_batch << ".";

  return Status::OK();
}

void DeviceQueueOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nChannel name: " << channel_name_ << "\nPrefetch size: " << prefetch_size_ << "\n\n";
  }
}

#ifndef ENABLE_SECURITY
void DeviceQueueOp::ProfilingRecorder(bool is_profiling_enable, std::shared_ptr<DeviceQueueTracing> profiling_node,
                                      int64_t send_batch, int32_t tdt_cost, uint64_t *batch_start_time,
                                      uint64_t *end_time, int32_t connector_capacity, int32_t connector_size) {
  // Record the pipeline profiling info
  if (is_profiling_enable) {
    *end_time = ProfilingTime::GetCurMilliSecond();
    // record push tdt time
    profiling_node->Record(TIME, TDT_PUSH_TIME, send_batch + 1, tdt_cost, *end_time);
    int32_t batch_cost = (int32_t)(*end_time - *batch_start_time);
    // record batch time
    profiling_node->Record(TIME, BATCH_TIME, send_batch + 1, batch_cost, *end_time);
    // record pipeline time
    profiling_node->Record(TIME, PIPELINE_TIME, send_batch + 1, batch_cost - tdt_cost, *end_time);
    *batch_start_time = *end_time;
    // record connector depth
    profiling_node->Record(CONNECTOR_DEPTH, connector_capacity, send_batch + 1, connector_size, *end_time);
  }
}

Status DeviceQueueOp::DetectFirstBatch() {
  TaskManager::FindMe()->Post();
  uint8_t count_num = 0;
  uint64_t temp_start_time = ProfilingTime::GetCurMilliSecond();
  while (true) {
    RETURN_IF_INTERRUPTED();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    uint64_t temp_end_time = ProfilingTime::GetCurMilliSecond();
    // if fetch first batch, or detect 3 or more times and unable fetch first batch, exist with already printed Warning
    if (first_fetch_flag_ == true || count_num > 2) {
      break;
    } else if (temp_end_time - temp_start_time > kTimeOutMilliSeconds) {
      count_num++;
      MS_LOG(WARNING) << "Bad performance attention, it waits more than 25 seconds and unable to fetch first Batch of "
                         "data from dataset pipeline, which might result `GetNext` timeout problem. You may test "
                         "dataset processing performance (with creating dataset iterator) and optimize it. Notes: "
                         "shuffle operation is turn on for loading Dataset in default, which may effect first batch "
                         "loading time.";
    }
  }
  return Status::OK();
}

void DeviceQueueOp::DetectPerBatchTime(const uint64_t *start_time, uint64_t *end_time) {
  *end_time = ProfilingTime::GetCurMilliSecond();
  if (*end_time - *start_time > kTimeOutMilliSeconds) {
    MS_LOG(WARNING) << "Bad performance attention, it takes more than 25 seconds to fetch a batch of data from dataset "
                       "pipeline, which might result `GetNext` timeout problem. You may test dataset processing"
                       " performance(with creating dataset iterator) and optimize it.";
  }
}

void DeviceQueueOp::PrintBeginInfoWhenFirstBatch(const bool &first_push_flag) {
  if (first_push_flag != true) {
    MS_LOG(INFO) << "Loading dataset and begin to push first batch into device ...";
  }
}

void DeviceQueueOp::PrintEndInfoWhenFirstBatch(bool *first_push_flag) {
  if (!first_push_flag) {
    MS_LOG(WARNING) << "First batch flag: first_push_flag is nullptr";
    return;
  }
  if (*first_push_flag != true) {
    MS_LOG(INFO) << "Loading dataset and push first batch into device successful.";
    *first_push_flag = true;
  }
}
#endif
}  // namespace dataset
}  // namespace mindspore
