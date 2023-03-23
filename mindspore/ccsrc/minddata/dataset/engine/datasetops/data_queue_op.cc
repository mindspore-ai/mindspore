/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/data_queue_op.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>

#include "minddata/dataset/engine/gpu_item_connector.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/task_manager.h"
#ifdef WITH_BACKEND
#include "mindspore/ccsrc/include/backend/data_queue/data_queue_mgr.h"
#endif
#ifndef _WIN32
#include "mindspore/ccsrc/ps/ps_cache/ps_data/ps_data_prefetch.h"
#endif
#ifdef WITH_BACKEND
#include "utils/ms_context.h"
#endif
namespace mindspore {
namespace dataset {
namespace {
std::vector<DataQueueItem> ConvertTensorRowToDataQueueItem(const TensorRow &row) {
  std::vector<device::DataQueueItem> items;
  for (auto &i : row) {
    device::DataQueueItem data_item;
    data_item.data_len = static_cast<size_t>(i->SizeInBytes());
    data_item.shapes = i->shape().AsVector();
    data_item.data_ptr = const_cast<void *>(static_cast<const void *>(i->GetBuffer()));
    data_item.data_type = i->type().ToString();
    items.emplace_back(std::move(data_item));
  }
  return items;
}
}  // namespace
DataQueueOp::DataQueueOp(const std::string channel_name, DeviceType device_type, int32_t device_id, bool send_epoch_end,
                         int32_t total_batch, bool create_data_info_queue)
    : PipelineOp(1),
      ascend_keep_waiting_(true),
      num_workers_(kDeviceQueGpuNumThreads),
      queue_capacity_(kDeviceQueGpuQueueCapacity),
      channel_name_(channel_name),
      device_type_(device_type),
      device_id_(device_id),
      send_epoch_end_(send_epoch_end),
      stop_send_(false),
      send_finished_(false),
      total_batch_(total_batch),
      create_data_info_queue_(create_data_info_queue),
      data_info_queue_ptr_(nullptr),
      first_fetch_flag_(false),
      first_push_flag_(false) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  dynamic_shape_ = cfg->dynamic_shape();

  // Be careful when try to modified these num_workers_ and queue_capacity_,
  // and we suggest num_workers_ * queue_capacity_ not greater than 16, because
  // one worker one circular_pool with 1G pin memory, so num_workers_ * queue_capacity_
  // must limit to avoid memory overload
#ifdef WITH_BACKEND
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    ascend_data_queue_ =
      device::DataQueueMgr::GetInstance().CreateDataQueue(kAscendDevice, channel_name, dynamic_shape_, 0, {});
  }
#endif
#ifdef ENABLE_DUMP_IR
  md_channel_info_ = std::make_shared<MDChannelInfo>(channel_name_);
#endif
}

DataQueueOp::~DataQueueOp() {
#ifdef ENABLE_DUMP_IR
  // BFS iter execution tree to get send epoch from EpochControl Op
  std::vector<std::shared_ptr<DatasetOp>> child_node = this->Children();
  size_t node_index = 0;
  int32_t num_epochs = 0;
  while (child_node.size() != 0 && node_index < child_node.size()) {
    auto node = child_node[node_index];
    if (node->Name() == kEpochCtrlOp) {
      EpochCtrlOp *op = dynamic_cast<EpochCtrlOp *>(node.get());
      if (op != nullptr) {
        num_epochs = op->NumEpochs();
        break;
      }
    }
    auto child_child_node = node->Children();
    if (!child_child_node.empty()) {
      std::copy(child_child_node.begin(), child_child_node.end(), std::back_inserter(child_node));
    }
    ++node_index;
  }

  // won't print rdr if call stop_send manually or send infinite epoch
  std::string rdr_msg = md_channel_info_->ToString();
  if (!send_finished_ && !rdr_msg.empty() && num_epochs != -1) {
    MS_LOG(WARNING) << rdr_msg;
  }
#endif
}

void DataQueueOp::ReleaseData(void *addr, int32_t worker_id) {
  if (addr != nullptr && worker_id >= 0 && worker_id < pool_.size()) {
    pool_[worker_id]->Deallocate(addr);
  }
}

Status DataQueueOp::EoeReceived(int32_t) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

Status DataQueueOp::FilterMetadata(TensorRow *row) const {
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

Status DataQueueOp::CheckExceptions(const TensorRow &row) const {
  // this method checks if the row meets the conditions to be sent to TDT
  for (const auto &item : row) {
    CHECK_FAIL_RETURN_UNEXPECTED(item->type().IsNumeric(),
                                 "Invalid datatype, cannot send string, or Python dict to device.");
    CHECK_FAIL_RETURN_UNEXPECTED(item->HasData(), "Invalid data, the data send to device is null.");
  }
  return Status::OK();
}

Status DataQueueOp::operator()() {
#ifndef ENABLE_SECURITY
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask("Detect first batch",
                                                      std::bind(&DataQueueOp::DetectFirstBatch, this), nullptr, id()));
#endif
  TaskManager::FindMe()->Post();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);

#ifdef ENABLE_DUMP_IR
  if (md_channel_info_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] RDR module init failed.");
  }
#endif
  if (device_type_ == DeviceType::Ascend) {
#ifdef WITH_BACKEND
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
    RETURN_IF_NOT_OK(SendDataToAscend());

#endif
  } else if (device_type_ == DeviceType::GPU) {
#ifdef WITH_BACKEND
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
    RETURN_IF_NOT_OK(SendDataToGPU());
#endif
  } else if (device_type_ == DeviceType::CPU) {
    RETURN_IF_NOT_OK(SendDataToCPU());
  }

  return Status::OK();
}

Status DataQueueOp::SendDataToAscend() {
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
  bool is_profiling_enable = GlobalContext::profiling_manager()->IsProfilingEnable(tree_);
  if (is_profiling_enable) {
    std::shared_ptr<Tracing> node;
    RETURN_IF_NOT_OK(GlobalContext::profiling_manager()->GetTracingNode(kDeviceQueueTracingName, &node));
    profiling_node = std::dynamic_pointer_cast<DeviceQueueTracing>(node);
    batch_start_time = ProfilingTime::GetCurMilliSecond();
    connector_capacity = ChildOpConnectorCapacity();
  }
#else
  bool is_profiling_enable = false;
#endif
#ifdef ENABLE_DUMP_IR
  RETURN_IF_NOT_OK(md_channel_info_->RecordBatchQueue(ChildOpConnectorSize()));
  RETURN_IF_NOT_OK(md_channel_info_->RecordPreprocessBatch(0));
#endif
#ifndef ENABLE_SECURITY
  batch_record_start = ProfilingTime::GetCurMilliSecond();
#endif
  TensorRow curr_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
  first_fetch_flag_ = true;

  MS_LOG(INFO) << "Begin to send data to device, channel name: " << channel_name_;

  while (!curr_row.eof() && !is_break_loop) {
    while (!curr_row.eoe() && !is_break_loop) {
      RETURN_IF_NOT_OK(FilterMetadata(&curr_row));
      RETURN_IF_NOT_OK(CheckExceptions(curr_row));
      WaitContinueSignal();
#ifdef ENABLE_DUMP_IR
      RETURN_IF_NOT_OK(md_channel_info_->RecordBatchQueue(ChildOpConnectorSize()));
      RETURN_IF_NOT_OK(md_channel_info_->RecordPreprocessBatch(send_batch));
      RETURN_IF_NOT_OK(md_channel_info_->RecordPushStartTime());
#endif
#ifndef ENABLE_SECURITY
      DetectPerBatchTime(&batch_record_start, &batch_record_end);
#endif
      PrintBeginInfoWhenFirstBatch(first_push_flag_);
      // when training stopped, handle might have been destroyed immediately
      if (ascend_data_queue_ != nullptr && !ascend_data_queue_->IsOpen()) {
        MS_LOG(WARNING) << "Thread has already been terminated.";
        is_break_loop = true;
        continue;
      }
      RETURN_IF_NOT_OK(SendRowToTdt(curr_row, is_profiling_enable, &tdt_cost));
      PrintEndInfoWhenFirstBatch(&first_push_flag_);
#ifndef ENABLE_SECURITY
      ProfilingRecorder(is_profiling_enable, profiling_node, send_batch, tdt_cost, &batch_start_time, &end_time,
                        connector_capacity, connector_size);
      batch_record_start = ProfilingTime::GetCurMilliSecond();
#endif
      send_batch++;
      MS_LOG(INFO) << "Have sent " << send_batch << " batch(es) to device, channel name: " << channel_name_;
#ifdef ENABLE_DUMP_IR
      RETURN_IF_NOT_OK(md_channel_info_->RecordBatchQueue(ChildOpConnectorSize()));
      RETURN_IF_NOT_OK(md_channel_info_->RecordPreprocessBatch(send_batch));
      RETURN_IF_NOT_OK(md_channel_info_->RecordPushEndTime());
#endif

      if (total_batch_ > 0 && send_batch >= total_batch_) {
        is_break_loop = true;
        break;
      }

      // wait when sending num is not 0, and sending num no larger than already sending batch
      LimitSendingBatches(send_batch, &sending_num, cfg);

#ifndef ENABLE_SECURITY
      RecordProfilingData(is_profiling_enable, false, &connector_size, &connector_capacity, &send_batch);
#endif
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
#ifndef ENABLE_SECURITY
      uint64_t batch_fetch_end = ProfilingTime::GetCurMilliSecond();
#endif
      if (ascend_data_queue_->QueueType() == "Ascend_MBUF") {
        // Queue control logic for mbuf in host, to prevent from hang/exit abnormally
        // case 1: If mbuf queue memory + next row memory < 2G then continue send, else suspend;
        // case 2: Based on case 1, if element nums in mbuf < max_queue_size then continue send, else suspend;
        // case 3: If row memory >= 1G, can only send 1 row each time, queue_size will always in [0, 1];
        // note:
        // why need queue control: acltdtSendTensor will hang when queue is full, need to break this thread by ourselves
        // how about dynamic shape: yes, memory_per_batch_ collect memory of rows in different shapes.
        // how about row too large(>2G): we can promise the first row will be sent and hang in this while, but we dont
        //     know if the device will out of memory. If not oom, send next row, otherwise device returns error.

        // Calculate the memory of next row before sending
        size_t queue_size = ascend_data_queue_->QueryQueueSize();
        double row_memory = curr_row.SizeInBytes() / 1024. / 1024. / 1024.;
        memory_per_batch_.push_back(row_memory);

        const double max_queue_memory = 2.;
        const size_t max_queue_size = 100;
        const int64_t send_interval = 1000;
        while ((row_memory + CalMbufQueueMemory(queue_size) >= max_queue_memory || queue_size >= max_queue_size) &&
               queue_size != 0) {
          RETURN_IF_INTERRUPTED();
          MS_LOG(DEBUG) << "Mbuf queue size: " << queue_size << ", max queue limit: " << max_queue_size << ". "
                        << "Next row memory: " << row_memory << ", Mbuf memory: " << CalMbufQueueMemory(queue_size);

          queue_size = ascend_data_queue_->QueryQueueSize();
          std::this_thread::sleep_for(std::chrono::microseconds(send_interval));
        }
      }
#ifndef ENABLE_SECURITY
      uint64_t queue_wait_end = ProfilingTime::GetCurMilliSecond();
      // Skip the time looping in the mbuf queue control, FetchNextTensorRow time is what we need
      batch_record_start = batch_record_start + (queue_wait_end - batch_fetch_end);
#endif
    }

    // send epoch end flag: ACL_TENSOR_DATA_END_OF_SEQUENCE to tdt
    RETURN_IF_NOT_OK(SendEpochEndToAscend(curr_row, is_profiling_enable, &tdt_cost, &is_break_loop));

#ifndef ENABLE_SECURITY
    RecordProfilingData(is_profiling_enable, true, &connector_size, &connector_capacity, &send_batch);
#endif
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
  }

  // now we use this flag to judge whether exception raised.
  if (stop_send_ || !TaskManager::FindMe()->Interrupted()) {
    send_finished_ = true;
  }
  tree_->SetFinished();
  MS_LOG(INFO) << "ExecutionTree finished. Device queue sent number of batches: " << send_batch;

  return Status::OK();
}

void DataQueueOp::RecordProfilingData(bool is_profiling_enable, bool end_of_epoch, int32_t *connector_size,
                                      int32_t *connector_capacity, const int64_t *send_batch) const {
  if (is_profiling_enable) {
    *connector_size = ChildOpConnectorSize();
    *connector_capacity = ChildOpConnectorCapacity();
  }
  if (end_of_epoch) {
    tree_->SetEpochEnd();
    GlobalContext::profiling_manager()->RecordEndOfEpoch(*send_batch);
  }
}

double DataQueueOp::CalMbufQueueMemory(size_t realtime_queue_size) {
  while (memory_per_batch_.size() > realtime_queue_size) {
    memory_per_batch_.pop_front();
  }
  return std::accumulate(memory_per_batch_.begin(), memory_per_batch_.end(), 0.);
}

Status DataQueueOp::SendEpochEndToAscend(const TensorRow &curr_row, const bool &is_profiling_enable, int32_t *tdt_cost,
                                         bool *is_break_loop) {
  RETURN_UNEXPECTED_IF_NULL(tdt_cost);
  RETURN_UNEXPECTED_IF_NULL(is_break_loop);
  if (curr_row.eoe() && send_epoch_end_ && ascend_data_queue_->IsOpen()) {
    TensorRow dummy_row;
#ifndef ENABLE_SECURITY
    double start_time = 0;
    if (is_profiling_enable) {
      start_time = ProfilingTime::GetCurMilliSecond();
    }
#endif
    auto status = ascend_data_queue_->Push({});
#ifndef ENABLE_SECURITY
    if (is_profiling_enable) {
      double end_time = ProfilingTime::GetCurMilliSecond();
      RETURN_UNEXPECTED_IF_NULL(tdt_cost);
      *tdt_cost = static_cast<int32_t>(end_time - start_time);
    }
#endif

    RETURN_IF_NOT_OK(CheckPushStatus(status, stop_send_, &send_finished_, is_break_loop));
    MS_LOG(INFO) << "an epoch has already sent, now stop send data.";
    stop_send_ = true;
  }
  return Status::OK();
}

void DataQueueOp::WaitContinueSignal() const {
  while (stop_send_ && ascend_keep_waiting_) {
    MS_LOG(DEBUG) << "stop_send flag is set, waiting for continue signal...";
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

void DataQueueOp::LimitSendingBatches(int64_t send_batch, int64_t *sending_num,
                                      const std::shared_ptr<ConfigManager> &cfg) const {
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

Status DataQueueOp::SendRowToTdt(TensorRow curr_row, bool is_profiling_enable, int32_t *tdt_cost) {
  std::vector<device::DataQueueItem> items = ConvertTensorRowToDataQueueItem(curr_row);
#ifndef ENABLE_SECURITY
  double start_time = 0;
  if (is_profiling_enable) {
    start_time = ProfilingTime::GetCurMilliSecond();
  }
#endif
  auto status = ascend_data_queue_->Push(items);
#ifndef ENABLE_SECURITY
  if (is_profiling_enable) {
    double end_time = ProfilingTime::GetCurMilliSecond();
    RETURN_UNEXPECTED_IF_NULL(tdt_cost);
    *tdt_cost = static_cast<int32_t>(end_time - start_time);
  }
#endif
  if (status != device::DataQueueStatus::SUCCESS) {
    if (stop_send_) {
      MS_LOG(INFO) << "stop_send received";
      return Status::OK();
    }
    RETURN_STATUS_ERROR(
      StatusCode::kMDTDTPushFailure,
      "TDT Push data into device Failed, check the first error or TraceBack first, more checking advises are: "
      "1) if training is not ready, error might raised by network computing operator or environment configuration. "
      "2) other cases, checking info level log or search this error in mindspore's FAQ for detail solution.");
  }
  if (create_data_info_queue_) {
    DATA_INFO data_info;
    (void)std::transform(curr_row.begin(), curr_row.end(), std::back_inserter(data_info),
                         [](const std::shared_ptr<Tensor> &ts) { return std::make_pair(ts->type(), ts->shape()); });
    RETURN_IF_NOT_OK(data_info_queue_ptr_->Add(data_info));
  }
  return Status::OK();
}

Status DataQueueOp::CheckPushStatus(DataQueueStatus status, bool stop_send, bool *send_finished, bool *is_break_loop) {
  if (status != DataQueueStatus::SUCCESS) {
    if (stop_send) {
      *send_finished = true;
      MS_LOG(INFO) << "stop_send received";
      return Status::OK();
    }
    // when training stopped, handle might have been destroyed immediately
    if (!ascend_data_queue_->IsOpen()) {
      *is_break_loop = true;
      MS_LOG(WARNING) << "Thread has already been terminated.";
      return Status::OK();
    }
    RETURN_STATUS_ERROR(
      StatusCode::kMDTDTPushFailure,
      "TDT Push data into device Failed, check the first error or TraceBack first, more checking advises are: "
      "1) if training is not ready, error might raised by network computing operator or environment configuration. "
      "2) other cases, checking info level log or search this error in mindspore's FAQ for detail solution.");
  }
  return Status::OK();
}

Status DataQueueOp::GetDataInfo(DATA_INFO *data_info) {
#ifdef WITH_BACKEND
  RETURN_UNEXPECTED_IF_NULL(MsContext::GetInstance());
  if (!create_data_info_queue_) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] DataInfo queue is not created.");
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
#endif
  return Status::OK();
}

Status DataQueueOp::SetThreadDevice() {
#ifdef WITH_BACKEND
  (void)device::DataQueueMgr::GetInstance().SetThreadDevice(channel_name_);
#endif
  return Status::OK();
}

Status DataQueueOp::CreateDynamicDataQueue() {
#ifdef WITH_BACKEND
  if (dynamic_shape_) {
    auto ret = device::DataQueueMgr::GetInstance().CreateDynamicBufQueue(channel_name_, kDynamicHostQueueCapacity);
    if (ret != DataQueueStatus::SUCCESS && ret != DataQueueStatus::QUEUE_EXIST) {
      RETURN_STATUS_ERROR(StatusCode::kMEFailed, "Create dynamic data queue failed");
    }
  }
#endif
  return Status::OK();
}

Status DataQueueOp::LaunchParallelCopyThread() {
#ifdef WITH_BACKEND
  RETURN_UNEXPECTED_IF_NULL(tree_);
  // Every thread use cuda api should SetThreadDevice
  RETURN_IF_NOT_OK(SetThreadDevice());
  // CircularPool may not safe under multi-threads scenario, so one worker with one pool
  for (int i = 0; i < num_workers_; i++) {
    std::shared_ptr<MemoryPool> pool;
    RETURN_UNEXPECTED_IF_NULL(MsContext::GetInstance());
    RETURN_IF_NOT_OK(CircularPool::CreateCircularPool(
      &pool, -1, kDeviceQueGpuThreadMemory, false,
      MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice));
    pool_.push_back(pool);
  }
  gpu_connector_ = std::make_unique<GpuConnector>(num_workers_, 1, queue_capacity_);
  receive_queues_.Init(num_workers_, queue_capacity_);
  RETURN_IF_NOT_OK(receive_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(static_cast<int>(num_workers_),
                                        std::bind(&DataQueueOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask("Push data to GPU queue",
                                                      std::bind(&DataQueueOp::PushDataToGPU, this), nullptr, id()));
#endif
  return Status::OK();
}

bool DataQueueOp::NoExceptionRaised() const {
#ifdef WITH_BACKEND
  return !TaskManager::FindMe()->Interrupted() && !device::DataQueueMgr::GetInstance().IsClosed();
#else
  return !TaskManager::FindMe()->Interrupted();
#endif
}

Status DataQueueOp::PushDataToGPU() {
#ifdef WITH_BACKEND
  RETURN_UNEXPECTED_IF_NULL(tree_);
  // Every thread use cuda api should SetThreadDevice
  RETURN_IF_NOT_OK(SetThreadDevice());
  TaskManager::FindMe()->Post();
#ifndef ENABLE_SECURITY
  uint64_t batch_start_time = 0;
  uint64_t end_time = 0;
  uint64_t push_cost = 0;
  std::shared_ptr<DeviceQueueTracing> profiling_node;
  bool is_profiling_enable = GlobalContext::profiling_manager()->IsProfilingEnable(tree_);
  if (is_profiling_enable) {
    std::shared_ptr<Tracing> node;
    RETURN_IF_NOT_OK(GlobalContext::profiling_manager()->GetTracingNode(kDeviceQueueTracingName, &node));
    profiling_node = std::dynamic_pointer_cast<DeviceQueueTracing>(node);
    batch_start_time = ProfilingTime::GetCurMilliSecond();
  }
#endif
#ifdef ENABLE_DUMP_IR
  md_channel_info_->RecordBatchQueue(gpu_connector_->size());
  md_channel_info_->RecordPreprocessBatch(0);
#endif
  GpuConnectorItem item;
  RETURN_IF_NOT_OK(gpu_connector_->Pop(0, &item));
  auto items = std::move(item.data_item);
  bool eoe_flag = item.eoe_flag;
  int64_t send_batch = 0;
  auto release_function = std::bind(&DataQueueOp::ReleaseData, this, std::placeholders::_1, std::placeholders::_2);
  auto ret = device::DataQueueMgr::GetInstance().Open(channel_name_, release_function);
  if (ret != DataQueueStatus::SUCCESS) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to open channel for sending data.");
  }

  while (!(items.empty() && !eoe_flag) && !device::DataQueueMgr::GetInstance().IsClosed()) {
    if (!eoe_flag) {
#ifdef ENABLE_DUMP_IR
      md_channel_info_->RecordBatchQueue(gpu_connector_->size());
      md_channel_info_->RecordPreprocessBatch(send_batch);
      md_channel_info_->RecordPushStartTime();
#endif
      // Data prefetch only when PS mode enables cache.
#ifndef _WIN32
      if (!ps::PsDataPrefetch::GetInstance().PrefetchData(channel_name_, items[0].data_ptr, items[0].data_len,
                                                          items[0].data_type)) {
        RETURN_STATUS_ERROR(StatusCode::kMDTimeOut,
                            "[Internal ERROR] Failed to prefetch data in current PS mode(cache data when sending).");
      }
#endif
      RETURN_IF_NOT_OK(RetryPushData(items, is_profiling_enable, &push_cost));
#ifndef ENABLE_SECURITY
      ProfilingRecorder(is_profiling_enable, profiling_node, send_batch, push_cost, &batch_start_time, &end_time,
                        gpu_connector_->capacity(), gpu_connector_->size());
#endif
      send_batch++;
      MS_LOG(INFO) << "Have sent " << send_batch << " batch(es) to device, channel name: " << channel_name_;
#ifdef ENABLE_DUMP_IR
      md_channel_info_->RecordBatchQueue(gpu_connector_->size());
      md_channel_info_->RecordPreprocessBatch(send_batch);
      md_channel_info_->RecordPushEndTime();
#endif
      if (total_batch_ > 0 && send_batch >= total_batch_) {
        break;
      }
    } else {
#ifndef ENABLE_SECURITY
      if (is_profiling_enable) {
        tree_->SetEpochEnd();
        GlobalContext::profiling_manager()->RecordEndOfEpoch(send_batch);
      }
#endif
    }
    if (NoExceptionRaised()) {
      auto rc = gpu_connector_->Pop(0, &item);
      items = std::move(item.data_item);
      eoe_flag = item.eoe_flag;
      // If the batches send by dataset are more than gpu calculate, gpu will core for no signal notify.
      if (rc.IsError()) {
        device::DataQueueMgr::GetInstance().Close(channel_name_);
        device::DataQueueMgr::GetInstance().CloseConfirm();
        return rc;
      }
    } else {
      break;
    }
  }

  // now we use this flag to judge whether exception raised.
  if (NoExceptionRaised()) {
    send_finished_ = true;
  }
  tree_->SetFinished();
  MS_LOG(INFO) << "ExecutionTree finished.  Device queue pushed number of batches: " << send_batch;

  device::DataQueueMgr::GetInstance().Close(channel_name_);
  device::DataQueueMgr::GetInstance().CloseConfirm();
  return Status::OK();
}

// WorkEntry of DataQueueOp just do multi_threads memcpy for performance optimization.
Status DataQueueOp::WorkerEntry(int32_t worker_id) {
  // Every thread use cuda api should SetThreadDevice
  RETURN_IF_NOT_OK(SetThreadDevice());
  TaskManager::FindMe()->Post();
  TensorRow current_row;
  uint32_t batch_num = 0;
  RETURN_IF_NOT_OK(receive_queues_[worker_id]->PopFront(&current_row));
  while (!current_row.quit() && !device::DataQueueMgr::GetInstance().IsClosed()) {
    GpuConnectorItem connector_item = {{}, current_row.eoe()};
    if (!connector_item.eoe_flag) {
      std::vector<device::DataQueueItem> items;
      for (auto &i : current_row) {
        device::DataQueueItem data_item;
        data_item.data_len = static_cast<size_t>(i->SizeInBytes());
        data_item.shapes = i->shape().AsVector();
        data_item.data_ptr = nullptr;
        data_item.worker_id = worker_id;
        items.push_back(data_item);
      }

      RETURN_IF_NOT_OK(MallocForGPUData(&items, current_row, worker_id));
      connector_item.data_item = std::move(items);
      batch_num++;
    } else {
      MS_LOG(INFO) << "EOE Detected";
    }
    RETURN_IF_NOT_OK(gpu_connector_->Add(worker_id, std::move(connector_item)));
    RETURN_IF_NOT_OK(receive_queues_[worker_id]->PopFront(&current_row));
  }

  MS_LOG(INFO) << "Device queue worker id " << worker_id << " processed number of batches: " << batch_num;
  // Add empty data_item vector with eoe_flag=false as quit flag.
  GpuConnectorItem connector_item = {{}, false};
  RETURN_IF_NOT_OK(gpu_connector_->Add(worker_id, std::move(connector_item)));
#endif
  return Status::OK();
}

Status DataQueueOp::SendDataToGPU() {
#ifdef WITH_BACKEND
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

  MS_LOG(INFO) << "Begin to send data to device, channel name: " << channel_name_;

  while (!current_row.eof() && !is_break_loop && !device::DataQueueMgr::GetInstance().IsClosed()) {
    while (!current_row.eoe() && !is_break_loop && !device::DataQueueMgr::GetInstance().IsClosed()) {
      RETURN_IF_NOT_OK(FilterMetadata(&current_row));
      RETURN_IF_NOT_OK(CheckExceptions(current_row));
#ifndef ENABLE_SECURITY
      DetectPerBatchTime(&batch_record_start, &batch_record_end);
#endif

      if (create_data_info_queue_) {
        DATA_INFO data_info;
        (void)std::transform(current_row.begin(), current_row.end(), std::back_inserter(data_info),
                             [](const std::shared_ptr<Tensor> &ts) { return std::make_pair(ts->type(), ts->shape()); });
        RETURN_IF_NOT_OK(data_info_queue_ptr_->Add(data_info));
      }

      PrintBeginInfoWhenFirstBatch(first_push_flag_);
      RETURN_IF_NOT_OK(receive_queues_[num_buf++ % num_workers_]->Add(std::move(current_row)));
      PrintEndInfoWhenFirstBatch(&first_push_flag_);
#ifndef ENABLE_SECURITY
      batch_record_start = ProfilingTime::GetCurMilliSecond();
#endif
      if (NoExceptionRaised()) {
        RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
      } else {
        is_break_loop = true;
      }
    }

    if (current_row.eoe()) {
      MS_LOG(INFO) << "EOE Detected";
      TensorRow eoe_flag(TensorRow::kFlagEOE);
      RETURN_IF_NOT_OK(receive_queues_[num_buf++ % num_workers_]->Add(std::move(eoe_flag)));
    }

    if (NoExceptionRaised()) {
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
    } else {
      is_break_loop = true;
    }
  }

  for (uint32_t index = 0; index < num_workers_; index++) {
    MS_LOG(INFO) << "Adding quit flag to Workers";
    TensorRow quit_flag(TensorRow::kFlagQuit);
    RETURN_IF_NOT_OK(receive_queues_[num_buf++ % num_workers_]->Add(std::move(quit_flag)));
  }

  MS_LOG(INFO) << "Device queue received number of batches and EOEs: " << (num_buf - num_workers_);
#else
  MS_LOG(WARNING) << "Gpu queue is not supported in ut tests.";
#endif
  return Status::OK();
}

Status DataQueueOp::MallocForGPUData(std::vector<device::DataQueueItem> *items, const TensorRow &curr_row,
                                     const int32_t &worker_id) {
  size_t i = 0;
  for (auto &sub_item : *items) {
    auto rc = pool_[static_cast<size_t>(worker_id)]->Allocate(sub_item.data_len, &sub_item.data_ptr);
    if (rc.IsError() || sub_item.data_ptr == nullptr) {
      RETURN_STATUS_OOM("Memory malloc failed, check memory usage.");
    }
    if (curr_row[i] == nullptr) {
      MS_LOG(ERROR) << "[Internal ERROR] The pointer curr_row[" << i << "] is null";
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] TensorRow 'curr_row' contains nullptr.");
    }
    sub_item.data_type = curr_row[i]->type().ToString();
    const unsigned char *column_data = curr_row[i]->GetBuffer();
    if (memcpy_s(sub_item.data_ptr, sub_item.data_len, column_data,
                 static_cast<uint32_t>(curr_row[i++]->SizeInBytes())) != 0) {
      MS_LOG(ERROR) << "[Internal ERROR] memcpy_s failed.";
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] memcpy_s failed.");
    }
  }

  return Status::OK();
}

Status DataQueueOp::ClearDevice() {
#ifdef WITH_BACKEND
  MS_LOG(INFO) << "Clearing the data in GPU device: " << device_id_ << " channel: " << channel_name_;
  auto release_function = std::bind(&DataQueueOp::ReleaseData, this, std::placeholders::_1, std::placeholders::_2);
  auto ret = device::DataQueueMgr::GetInstance().Open(channel_name_, release_function);
  if (ret != DataQueueStatus::SUCCESS) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to open channel for clearing the device.");
  }

  ret = device::DataQueueMgr::GetInstance().Clear(channel_name_);
  CHECK_FAIL_RETURN_UNEXPECTED(ret == DataQueueStatus::SUCCESS, "Failed to clear the device.");
  device::DataQueueMgr::GetInstance().Close(channel_name_);
  device::DataQueueMgr::GetInstance().CloseConfirm();
#endif
  return Status::OK();
}

Status DataQueueOp::SendDataToCPU() {
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
      if (stop_send_) {
        break;
      }
    }
  }

  MS_LOG(INFO) << "Device queue total batch is " << total_batch << ".";

  return Status::OK();
}

void DataQueueOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nChannel name: " << channel_name_ << "\n\n";
  }
}

#ifndef ENABLE_SECURITY
void DataQueueOp::ProfilingRecorder(bool is_profiling_enable, const std::shared_ptr<DeviceQueueTracing> &profiling_node,
                                    int64_t send_batch, int32_t tdt_cost, uint64_t *batch_start_time,
                                    uint64_t *end_time, int32_t connector_capacity, int32_t connector_size) const {
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

Status DataQueueOp::DetectFirstBatch() {
  TaskManager::FindMe()->Post();
  uint8_t count_num = 0;
  uint64_t temp_start_time = ProfilingTime::GetCurMilliSecond();
  constexpr int check_interval = 200;
  while (true) {
    RETURN_IF_INTERRUPTED();
    std::this_thread::sleep_for(std::chrono::milliseconds(check_interval));
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

void DataQueueOp::DetectPerBatchTime(const uint64_t *start_time, uint64_t *end_time) const {
  *end_time = ProfilingTime::GetCurMilliSecond();
  if (*end_time - *start_time > kTimeOutMilliSeconds) {
    MS_LOG(WARNING) << "Bad performance attention, it takes more than 25 seconds to fetch a batch of data from dataset "
                       "pipeline, which might result `GetNext` timeout problem. You may test dataset processing"
                       " performance(with creating dataset iterator) and optimize it.";
  }
}
#endif

void DataQueueOp::PrintBeginInfoWhenFirstBatch(const bool &first_push_flag) const {
  if (first_push_flag != true) {
    MS_LOG(INFO) << "Loading dataset and begin to push first batch into device ...";
  }
}

void DataQueueOp::PrintEndInfoWhenFirstBatch(bool *first_push_flag) const {
  if (!first_push_flag) {
    MS_LOG(WARNING) << "First batch flag: first_push_flag is nullptr";
    return;
  }
  if (*first_push_flag != true) {
    MS_LOG(INFO) << "Loading dataset and push first batch into device successful.";
    *first_push_flag = true;
  }
}

Status DataQueueOp::RetryPushData(const std::vector<DataQueueItem> &items, const bool profiling, uint64_t *push_time) {
#ifdef WITH_BACKEND
  bool flag_log = false;
#ifndef ENABLE_SECURITY
  uint64_t start_time = 0;
  if (profiling) {
    start_time = ProfilingTime::GetCurMilliSecond();
  }
#endif
  while (!device::DataQueueMgr::GetInstance().IsClosed() && !TaskManager::FindMe()->Interrupted()) {
    DataQueueStatus ret = device::DataQueueMgr::GetInstance().Push(channel_name_, items, WAIT_TIME);
    if (ret != DataQueueStatus::SUCCESS) {
      if (ret == DataQueueStatus::ERROR_INPUT) {
        device::DataQueueMgr::GetInstance().Close(channel_name_);
        device::DataQueueMgr::GetInstance().CloseConfirm();
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
#ifndef ENABLE_SECURITY
  if (profiling) {
    *push_time = ProfilingTime::GetCurMilliSecond() - start_time;
  }
#endif
#endif
  return Status::OK();
}

Status DataQueueOp::SendDataToAscendDynamic() {
#ifdef WITH_BACKEND
  MS_LOG(DEBUG) << "Dynamic Data queue, sending data to Ascend.";
  int64_t send_batch = 0;
  uint64_t data_queue_cost = 0;
  bool is_break_loop = false;

  RETURN_IF_NOT_OK(CreateDynamicDataQueue());
  std::function<void(void *, int32_t)> release_function([](void *, int32_t) { return; });
  auto ret = device::DataQueueMgr::GetInstance().Open(channel_name_, release_function);
  if (ret != DataQueueStatus::SUCCESS) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "[Internal ERROR] Failed to open channel for sending data.");
  }

  TensorRow curr_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
  first_fetch_flag_ = true;

  while (!curr_row.eof() && !is_break_loop) {
    while (!curr_row.eoe() && !is_break_loop) {
      RETURN_IF_NOT_OK(FilterMetadata(&curr_row));
      RETURN_IF_NOT_OK(CheckExceptions(curr_row));
      std::vector<device::DataQueueItem> items = ConvertTensorRowToDataQueueItem(curr_row);
      RETURN_IF_NOT_OK(RetryPushData(items, false, &data_queue_cost));
      if (create_data_info_queue_) {
        DATA_INFO data_info;
        (void)std::transform(curr_row.begin(), curr_row.end(), std::back_inserter(data_info),
                             [](const std::shared_ptr<Tensor> &ts) { return std::make_pair(ts->type(), ts->shape()); });
        RETURN_IF_NOT_OK(data_info_queue_ptr_->Add(data_info));
      }
      send_batch++;
      if (total_batch_ > 0 && send_batch >= total_batch_) {
        is_break_loop = true;
        break;
      }

      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
    }

    if (curr_row.eoe() && send_epoch_end_) {
      MS_LOG(INFO) << "an epoch has already sent, now stop send data.";
      stop_send_ = true;
    }

    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&curr_row));
  }

  // now we use this flag to judge whether exception raised.
  if (stop_send_ || !TaskManager::FindMe()->Interrupted()) {
    send_finished_ = true;
  }
  tree_->SetFinished();
  MS_LOG(INFO) << "ExecutionTree finished. Device queue sent number of batches: " << send_batch;

  device::DataQueueMgr::GetInstance().Close(channel_name_);
  device::DataQueueMgr::GetInstance().CloseConfirm();
#endif
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
