/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <iomanip>
#include <iostream>
#include <memory>
#include "dataset/core/config_manager.h"
#include "dataset/core/global_context.h"
#include "dataset/engine/datasetops/device_queue_op.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/dataset_iterator.h"
#include "dataset/engine/opt/pass.h"
#include "dataset/engine/perf/profiling.h"
#include "dataset/engine/perf/device_queue_tracing.h"
#include "dataset/util/status.h"
#include "dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
DeviceQueueOp::DeviceQueueOp(std::string channel_name, DeviceType device_type, int32_t device_id, int32_t prefetch_size,
                             int32_t op_connector_size, int64_t num_batch)
    : PipelineOp(op_connector_size),
      channel_name_(channel_name),
      device_type_(device_type),
      device_id_(device_id),
      prefetch_size_(prefetch_size),
      num_batch_(num_batch) {}

DeviceQueueOp::~DeviceQueueOp() {}

#ifdef ENABLE_GPUQUE
void ReleaseData(void *addr) {
  if (addr != nullptr) {
    free(addr);
  }
}
#endif

DeviceQueueOp::Builder::Builder(int32_t prefetch_size)
    : builder_prefetch_size_(prefetch_size),
      builder_device_id_(0),
      builder_device_type_(DeviceType::CPU),
      builder_channel_name_(""),
      builder_num_batch_(0) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status DeviceQueueOp::EoeReceived(int32_t worker_id) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

Status DeviceQueueOp::operator()() {
  TaskManager::FindMe()->Post();

  if (device_type_ == DeviceType::Ascend) {
#ifdef ENABLE_TDTQUE
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

Status DeviceQueueOp::CheckExceptions(const std::unique_ptr<DataBuffer> &buffer) const {
  // this method checks if the buffer meets the conditions to be sent to TDT
  if (buffer->NumRows() != 0) {
    TensorRow row;
    buffer->GetRow(0, &row);
    for (const auto &item : row) {
      CHECK_FAIL_RETURN_UNEXPECTED(item->type().IsNumeric(), "Cannot send tensor of string type to device.");
    }
  }
  return Status::OK();
}

#ifdef ENABLE_TDTQUE
Status DeviceQueueOp::SendDataToAscend() {
  MS_LOG(INFO) << "Device queue, sending data to Ascend.";
  int64_t total_batch = 0;
  bool is_break_loop = false;
  double batch_start_time, end_time;
  int32_t batch_cost, tdt_cost;
  int32_t connector_size = 0;
  int32_t connector_capacity;
  std::shared_ptr<DeviceQueueTracing> profiling_node;
  bool isProfilingEnable = tree_->GetProfilingManager()->IsProfilingEnable();
  if (isProfilingEnable) {
    std::shared_ptr<Tracing> node;
    RETURN_IF_NOT_OK(tree_->GetProfilingManager()->GetTracingNode(kDeviceQueueTracingName, &node));
    profiling_node = std::dynamic_pointer_cast<DeviceQueueTracing>(node);
    batch_start_time = ProfilingTime::GetCurMilliSecond();
    connector_capacity = ChildOpConnectorCapacity();
  }
  std::unique_ptr<DataBuffer> current_buffer;
  RETURN_IF_NOT_OK(GetNextInput(&current_buffer));

  while (!current_buffer->eof() && !is_break_loop) {
    while (!current_buffer->eoe() && !is_break_loop) {
      RETURN_IF_NOT_OK(CheckExceptions(current_buffer));
      TensorRow currRow;
      for (int row_id = 0; row_id < current_buffer->NumRows() && !is_break_loop; row_id++) {
        RETURN_IF_NOT_OK(current_buffer->GetRow(row_id, &currRow));
        auto status = tdtInstancePtr->hostPush(currRow, true, channel_name_, isProfilingEnable, tdt_cost);
        if (status == TdtStatus::FAILED) {
          return Status(StatusCode::kTDTPushFailure, "TDT Push Failed");
        }

        if (isProfilingEnable) {
          end_time = ProfilingTime::GetCurMilliSecond();
          // record push tdt time
          profiling_node->Record(TIME, TDT_PUSH_TIME, total_batch + 1, tdt_cost);
          batch_cost = (int32_t)(end_time - batch_start_time);
          // record batch time
          profiling_node->Record(TIME, BATCH_TIME, total_batch + 1, batch_cost);
          // record pipeline time
          profiling_node->Record(TIME, PIPELINE_TIME, total_batch + 1, batch_cost - tdt_cost);
          batch_start_time = end_time;
          // record connector depth
          profiling_node->Record(CONNECTOR_DEPTH, connector_capacity, total_batch + 1, connector_size);
        }
        total_batch++;
        if (num_batch_ > 0 && total_batch == num_batch_) {
          is_break_loop = true;
        }
      }
      if (isProfilingEnable) {
        connector_size = ChildOpConnectorSize();
        connector_capacity = ChildOpConnectorCapacity();
      }
      RETURN_IF_NOT_OK(GetNextInput(&current_buffer));
    }
    if (isProfilingEnable) {
      connector_size = ChildOpConnectorSize();
      connector_capacity = ChildOpConnectorCapacity();
    }
    RETURN_IF_NOT_OK(GetNextInput(&current_buffer));
  }

  tree_->SetFinished();
  MS_LOG(INFO) << "Device queue total batch is " << total_batch << ", number of batches is " << num_batch_ << ".";

  return Status::OK();
}
#endif

#ifdef ENABLE_GPUQUE
Status DeviceQueueOp::SendDataToGPU() {
  MS_LOG(INFO) << "Device queue, sending data to GPU.";
  int64_t total_batch = 0;
  bool is_break_loop = false;
  bool is_open = false;
  uint32_t handle = INVALID_HANDLE;

  std::unique_ptr<DataBuffer> current_buffer;
  RETURN_IF_NOT_OK(GetNextInput(&current_buffer));

  while (!current_buffer->eof() && !is_break_loop && !GpuBufferMgr::GetInstance().IsClosed()) {
    while (!current_buffer->eoe() && !is_break_loop && !GpuBufferMgr::GetInstance().IsClosed()) {
      RETURN_IF_NOT_OK(CheckExceptions(current_buffer));
      TensorRow curr_row;  // batch data
      for (int row_id = 0;
           row_id < current_buffer->NumRows() && !is_break_loop && !GpuBufferMgr::GetInstance().IsClosed(); row_id++) {
        RETURN_IF_NOT_OK(current_buffer->GetRow(row_id, &curr_row));

        std::vector<size_t> data_size;
        for (int i = 0; i < curr_row.size(); i++) {
          data_size.push_back(static_cast<size_t>(curr_row[i]->SizeInBytes()));
        }
        if (!is_open) {
          handle = GpuBufferMgr::GetInstance().Open(0, channel_name_, data_size, ReleaseData);
          if (handle == INVALID_HANDLE) {
            return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "open failed");
          }
          is_open = true;
        }
        RETURN_IF_NOT_OK(RetryPushGPUData(data_size, curr_row, handle));
        total_batch++;
        if (num_batch_ > 0 && total_batch == num_batch_) {
          is_break_loop = true;
        }
      }
      if (!TaskManager::FindMe()->Interrupted())
        RETURN_IF_NOT_OK(GetNextInput(&current_buffer));
      else
        is_break_loop = true;
    }
    if (!TaskManager::FindMe()->Interrupted())
      RETURN_IF_NOT_OK(GetNextInput(&current_buffer));
    else
      is_break_loop = true;
  }

  MS_LOG(INFO) << "Device queue total batch is " << total_batch << ", number of batches is " << num_batch_ << ".";

  GpuBufferMgr::GetInstance().Close(handle);

  GpuBufferMgr::GetInstance().CloseConfirm();

  return Status::OK();
}

Status DeviceQueueOp::RetryPushGPUData(const std::vector<size_t> &data_size, const TensorRow &curr_row,
                                       uint32_t handle) {
  std::vector<device::DataItemGpu> items;
  for (int i = 0; i < data_size.size(); i++) {
    device::DataItemGpu data_item;
    data_item.data_len_ = data_size[i];
    data_item.data_ptr_ = nullptr;
    items.push_back(data_item);
  }

  while (!GpuBufferMgr::GetInstance().IsClosed() && !TaskManager::FindMe()->Interrupted()) {
    RETURN_IF_NOT_OK(MallocForGPUData(&items, curr_row));
    BlockQueueStatus_T ret = GpuBufferMgr::GetInstance().Push(handle, items, WAIT_TIME);
    if (ret) {
      for (int i = 0; i < items.size(); i++) {
        free(items[i].data_ptr_);
      }
      if (ret == BlockQueueStatus_T::ERROR_INPUT) {
        return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "invalid input Data, please check it.");
      } else {
        MS_LOG(WARNING) << "Retry pushing data...";
        continue;
      }
    } else {
      break;
    }
  }
  return Status::OK();
}

Status DeviceQueueOp::MallocForGPUData(std::vector<device::DataItemGpu> *items, const TensorRow &curr_row) {
  int i = 0;
  for (auto &sub_item : *items) {
    sub_item.data_ptr_ = (unsigned char *)malloc(sub_item.data_len_);
    if (sub_item.data_ptr_ == nullptr) {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "memory malloc failed.");
    }
    (void)memset_s(sub_item.data_ptr_, sub_item.data_len_, 0, sub_item.data_len_);
    const unsigned char *column_data = curr_row[i]->GetBuffer();
    if (memcpy_s(sub_item.data_ptr_, sub_item.data_len_, column_data,
                 static_cast<uint32_t>(curr_row[i++]->SizeInBytes())) != 0) {
      MS_LOG(ERROR) << "memcpy_s failed!";
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "memcpy_s failed.");
    }
  }

  return Status::OK();
}
#endif

Status DeviceQueueOp::SendDataToCPU() {
  MS_LOG(INFO) << "Device queue, sending data to CPU.";
  int64_t total_batch = 0;

  std::unique_ptr<ChildIterator> child_iterator = std::make_unique<ChildIterator>(this, 0, 0);
  while (!(child_iterator->eof_handled())) {
    TensorRow curr_row;
    RETURN_IF_NOT_OK(child_iterator->FetchNextTensorRow(&curr_row));

    if (!curr_row.empty()) {
      MS_LOG(DEBUG) << "Feature size is " << curr_row[0]->SizeInBytes() << ".";
      MS_LOG(DEBUG) << "Label size is " << curr_row[1]->SizeInBytes() << ".";
      total_batch++;
      if (num_batch_ > 0 && total_batch == num_batch_) {
        break;
      }
    }
  }

  MS_LOG(INFO) << "Device queue total batch is " << total_batch << ", number of batches is " << num_batch_ << ".";

  return Status::OK();
}

void DeviceQueueOp::Print(std::ostream &out, bool show_all) const {
  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <DeviceQueueOp>:";
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

// Visitor accept method for NodePass
Status DeviceQueueOp::Accept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->RunOnNode(std::static_pointer_cast<DeviceQueueOp>(shared_from_this()), modified);
}

}  // namespace dataset
}  // namespace mindspore
