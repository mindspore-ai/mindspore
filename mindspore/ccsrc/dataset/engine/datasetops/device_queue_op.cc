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
#include "dataset/engine/datasetops/device_queue_op.h"
#include <iomanip>
#include <iostream>
#include <memory>

#include "dataset/core/config_manager.h"
#include "dataset/core/global_context.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/dataset_iterator.h"
#include "dataset/util/status.h"
#include "dataset/util/task_manager.h"

#ifdef ENABLE_TDTQUE
#include "tdt/tsd_client.h"
#endif

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
  return Status::OK();
}

#ifdef ENABLE_TDTQUE
Status DeviceQueueOp::SendDataToAscend() {
  MS_LOG(INFO) << "Device queue, sending data to Ascend.";
  int64_t total_batch = 0;
  bool is_break_loop = false;

  std::unique_ptr<DataBuffer> current_buffer;
  RETURN_IF_NOT_OK(GetNextInput(&current_buffer));

  while (!current_buffer->eof() && !is_break_loop) {
    while (!current_buffer->eoe() && !is_break_loop) {
      RETURN_IF_NOT_OK(CheckExceptions(current_buffer));
      TensorRow currRow;
      for (int row_id = 0; row_id < current_buffer->NumRows() && !is_break_loop; row_id++) {
        RETURN_IF_NOT_OK(current_buffer->GetRow(row_id, &currRow));
        auto status = tdtInstancePtr->hostPush(currRow, true, channel_name_);
        if (status == TdtStatus::FAILED) {
          return Status(StatusCode::kTDTPushFailure, "TDT Push Failed");
        }
        total_batch++;
        if (num_batch_ > 0 && total_batch == num_batch_) {
          is_break_loop = true;
        }
      }
      RETURN_IF_NOT_OK(GetNextInput(&current_buffer));
    }
    RETURN_IF_NOT_OK(GetNextInput(&current_buffer));
  }

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
        if (curr_row.size() < 2) {
          return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Invalid tensor size");
        }
        uint32_t feature_size = static_cast<uint32_t>(curr_row[0]->SizeInBytes());
        uint32_t label_size = static_cast<uint32_t>(curr_row[1]->SizeInBytes());
        if (!is_open) {
          handle = GpuBufferMgr::GetInstance().Open(0, channel_name_, feature_size, label_size, ReleaseData);
          if (handle == INVALID_HANDLE) {
            return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "open failed");
          }
          is_open = true;
        }
        RETURN_IF_NOT_OK(RetryPushGPUData(feature_size, label_size, curr_row, handle));
        total_batch++;
        if (num_batch_ > 0 && total_batch == num_batch_) {
          is_break_loop = true;
        }
      }
      RETURN_IF_NOT_OK(GetNextInput(&current_buffer));
    }
    RETURN_IF_NOT_OK(GetNextInput(&current_buffer));
  }

  MS_LOG(INFO) << "Device queue total batch is " << total_batch << ", number of batches is " << num_batch_ << ".";

  GpuBufferMgr::GetInstance().Close(handle);

  GpuBufferMgr::GetInstance().CloseConfirm();

  return Status::OK();
}

Status DeviceQueueOp::RetryPushGPUData(uint32_t feature_size, uint32_t label_size, const TensorRow &curr_row,
                                       uint32_t handle) {
  unsigned char *feature_addr = nullptr;
  unsigned char *label_addr = nullptr;
  while (true && !GpuBufferMgr::GetInstance().IsClosed()) {
    RETURN_IF_NOT_OK(MallocForGPUData(&feature_addr, feature_size, &label_addr, label_size, curr_row));
    auto ret = GpuBufferMgr::GetInstance().Push(handle, feature_addr, feature_size, label_addr, label_size, WAIT_TIME);
    if (ret) {
      free(feature_addr);
      free(label_addr);
      MS_LOG(WARNING) << "Retry pushing data...";
      continue;
    } else {
      break;
    }
  }
  return Status::OK();
}

Status DeviceQueueOp::MallocForGPUData(unsigned char **feature_addr, uint32_t feature_size, unsigned char **label_addr,
                                       uint32_t label_size, const TensorRow &curr_row) {
  *feature_addr = (unsigned char *)malloc(feature_size);
  if (*feature_addr == nullptr) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "feature memory malloc failed.");
  }
  (void)memset_s(*feature_addr, feature_size, 0, feature_size);
  unsigned char *feature = curr_row[0]->StartAddr();
  if (memcpy_s(*feature_addr, feature_size, feature, static_cast<uint32_t>(curr_row[0]->SizeInBytes())) != 0) {
    MS_LOG(ERROR) << "Feature memcpy_s failed!";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "feature memcpy_s failed.");
  }

  *label_addr = (unsigned char *)malloc(label_size);
  if (*label_addr == nullptr) {
    free(*feature_addr);
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "label memory malloc failed.");
  }
  (void)memset_s(*label_addr, label_size, 0, label_size);
  unsigned char *label = curr_row[1]->StartAddr();
  if (memcpy_s(*label_addr, label_size, label, static_cast<uint32_t>(curr_row[1]->SizeInBytes())) != 0) {
    MS_LOG(ERROR) << "Label memcpy_s failed!";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "label memcpy_s failed.");
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
}  // namespace dataset
}  // namespace mindspore
