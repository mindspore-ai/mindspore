/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/receive_bridge_op.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "minddata/dataset/callback/callback_param.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/tensor_row.h"

#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/monitor.h"

namespace mindspore {
namespace dataset {
// Constructor of ReceiveBridgeOp
ReceiveBridgeOp::ReceiveBridgeOp(int32_t op_connector_size, SharedMemoryQueue receive_queue, MessageQueue msg_queue)
    : ParallelOp(1, op_connector_size),
      receive_queue_(receive_queue),
      msg_queue_(msg_queue),
      subprocess_pid_(-1),
      subprocess_status_(true) {
  receive_info_.normal_row_.sample_ = 0;
  receive_info_.normal_row_.row_step_ = ReceiveBridgeOp::RowStep::kNone;
  receive_info_.eoe_row_.sample_ = 0;
  receive_info_.eoe_row_.row_step_ = ReceiveBridgeOp::RowStep::kNone;
  receive_info_.eof_row_.sample_ = 0;
  receive_info_.eof_row_.row_step_ = ReceiveBridgeOp::RowStep::kNone;
}

ReceiveBridgeOp::~ReceiveBridgeOp() {
  receive_queue_.SetReleaseFlag(true);
  msg_queue_.SetReleaseFlag(true);

  std::string err_msg = "Dataset ReceiveOp normal_row: " + std::to_string(receive_info_.normal_row_.sample_) +
                        ", status: " + std::to_string(receive_info_.normal_row_.row_step_) +
                        ", eoe_row: " + std::to_string(receive_info_.eoe_row_.sample_) +
                        ", status: " + std::to_string(receive_info_.eoe_row_.row_step_) +
                        ", eof_row: " + std::to_string(receive_info_.eof_row_.sample_) +
                        ", status: " + std::to_string(receive_info_.eof_row_.row_step_);
  if (receive_info_.normal_row_.row_step_ == 0 ||
      receive_info_.normal_row_.row_step_ == ReceiveBridgeOp::RowStep::kNone) {
    return;
  }
  if (receive_info_.normal_row_.row_step_ != ReceiveBridgeOp::RowStep::kAfterReceiveMsg) {
    MS_LOG(WARNING) << err_msg;
  }
  if (receive_info_.eoe_row_.row_step_ == ReceiveBridgeOp::RowStep::kNone) {
    return;
  }
  if (receive_info_.normal_row_.row_step_ == ReceiveBridgeOp::RowStep::kAfterReceiveMsg &&
      receive_info_.eoe_row_.row_step_ != ReceiveBridgeOp::RowStep::kAfterReceiveMsg) {
    MS_LOG(WARNING) << err_msg;
  }
  if (receive_info_.eof_row_.row_step_ == ReceiveBridgeOp::RowStep::kNone) {
    return;
  }
  if (receive_info_.normal_row_.row_step_ == ReceiveBridgeOp::RowStep::kAfterReceiveMsg &&
      receive_info_.eoe_row_.row_step_ == ReceiveBridgeOp::RowStep::kAfterReceiveMsg &&
      receive_info_.eof_row_.row_step_ != ReceiveBridgeOp::RowStep::kAfterReceiveMsg) {
    MS_LOG(WARNING) << err_msg;
  }
}

// A print method typically used for debugging
void ReceiveBridgeOp::Print(std::ostream &out, bool show_all) const {
  // Call the super class for displaying any common 1-liner info
  ParallelOp::Print(out, show_all);
  // Then show any custom derived-internal 1-liner info for this op
  out << "\n";
}

Status ReceiveBridgeOp::MonitorIndependentDatasetProcess() {
  TaskManager::FindMe()->Post();
  while (!tree_->isFinished()) {
    RETURN_IF_INTERRUPTED();
    if (MonitorSubprocess(subprocess_pid_) != Status::OK() && receive_info_.eof_row_.row_step_ == 0) {
      MS_LOG(WARNING) << "The independent dataset process: " << std::to_string(subprocess_pid_)
                      << " exits, and the main process exits.";
      subprocess_status_ = false;
      break;
    }

    // get error flag from dataset independent process
    if (msg_queue_.MsgRcv(111, IPC_NOWAIT) > 0) {
      MS_LOG(ERROR) << "The independent dataset process occur errors.";
      break;
    }
    sleep(1);  // check the independent dataset process status in every 1s
  }

  // del the message queue id
  msg_queue_.SetReleaseFlag(true);
  msg_queue_.~MessageQueue();

  // Quit all workers, this code might never be reached if EpochCtrl is -1.
  for (int32_t wkr_id = 0; wkr_id < num_workers_; wkr_id++) {
    RETURN_IF_NOT_OK(SendQuitFlagToWorker(NextWorkerID()));
  }

  return Status::OK();
}

// This class functor will provide the master loop that drives the logic for performing the work
Status ReceiveBridgeOp::operator()() {
  RETURN_IF_NOT_OK(RegisterAndLaunchThreads());

  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(
    "ReceiveBridge-MonitorIndependentDatasetProcess",
    std::bind(&ReceiveBridgeOp::MonitorIndependentDatasetProcess, this), nullptr, id()));

  // Synchronize with TaskManager
  TaskManager::FindMe()->Post();

  // Get msg from the independent dataset process by msg_queue_
  receive_info_.normal_row_.row_step_ = ReceiveBridgeOp::RowStep::kBeginReceiveMsg;
  auto status = msg_queue_.MsgRcv(kWorkerSendDataMsg);
  if (status != Status::OK()) {
    tree_->SetFinished();
    if (subprocess_status_ == false) {
      RETURN_STATUS_UNEXPECTED("Independent Dataset Process reports an error and exit.");
    } else {
      return status;
    }
  }
  receive_info_.normal_row_.row_step_ = ReceiveBridgeOp::RowStep::kAfterReceiveMsg;

  TensorRow new_row;
  RETURN_IF_NOT_OK(receive_queue_.ToTensorRow(&new_row, msg_queue_.shm_id_, msg_queue_.shm_size_));

  while (!new_row.eof()) {
    while (!new_row.eoe()) {
      RETURN_IF_INTERRUPTED();
      receive_info_.normal_row_.sample_ += 1;
      // The NextWorkerID() should always 0, because ReceiveBridgeOp is a single thread op
      RETURN_IF_NOT_OK(worker_in_queues_[NextWorkerID()]->Add(std::move(new_row)));

      MS_LOG(INFO) << "Dataset ReceiveOp normal_row: " << std::to_string(receive_info_.normal_row_.sample_)
                   << ", status: " << receive_info_.normal_row_.row_step_
                   << ", eoe_row: " << std::to_string(receive_info_.eoe_row_.sample_)
                   << ", status: " << receive_info_.eoe_row_.row_step_
                   << ", eof_row: " << std::to_string(receive_info_.eof_row_.sample_)
                   << ", status: " << receive_info_.eof_row_.row_step_;

      // Send msg to the independent dataset process by msg_queue_
      receive_info_.normal_row_.row_step_ = ReceiveBridgeOp::RowStep::kBeginSendMsg;
      RETURN_IF_NOT_OK(msg_queue_.MsgSnd(kMasterSendDataMsg));
      receive_info_.normal_row_.row_step_ = ReceiveBridgeOp::RowStep::kAfterSendMsg;

      receive_info_.normal_row_.row_step_ = ReceiveBridgeOp::RowStep::kBeginReceiveMsg;
      // Get msg from the independent dataset process by msg_queue_
      status = msg_queue_.MsgRcv(kWorkerSendDataMsg);
      if (status != Status::OK()) {
        tree_->SetFinished();
        if (subprocess_status_ == false) {
          RETURN_STATUS_UNEXPECTED("Independent Dataset Process reports an error and exit.");
        } else {
          return status;
        }
      }
      receive_info_.normal_row_.row_step_ = ReceiveBridgeOp::RowStep::kAfterReceiveMsg;

      RETURN_IF_NOT_OK(receive_queue_.ToTensorRow(&new_row, msg_queue_.shm_id_, msg_queue_.shm_size_));
    }

    receive_info_.eoe_row_.sample_ += 1;

    // Propagate the eoe row to worker
    RETURN_IF_NOT_OK(worker_in_queues_[NextWorkerID()]->Add(std::move(new_row)));
    UpdateRepeatAndEpochCounter();

    // Send msg to the independent dataset process by msg_queue_
    receive_info_.eoe_row_.row_step_ = ReceiveBridgeOp::RowStep::kBeginSendMsg;
    RETURN_IF_NOT_OK(msg_queue_.MsgSnd(kMasterSendDataMsg));
    receive_info_.eoe_row_.row_step_ = ReceiveBridgeOp::RowStep::kAfterSendMsg;

    // Get msg from the independent dataset process by msg_queue_
    receive_info_.eoe_row_.row_step_ = ReceiveBridgeOp::RowStep::kBeginReceiveMsg;
    status = msg_queue_.MsgRcv(kWorkerSendDataMsg);
    if (status != Status::OK()) {
      tree_->SetFinished();
      if (subprocess_status_ == false) {
        RETURN_STATUS_UNEXPECTED("Independent Dataset Process reports an error and exit.");
      } else {
        return status;
      }
    }
    receive_info_.eoe_row_.row_step_ = ReceiveBridgeOp::RowStep::kAfterReceiveMsg;

    RETURN_IF_NOT_OK(receive_queue_.ToTensorRow(&new_row, msg_queue_.shm_id_, msg_queue_.shm_size_));
  }
  receive_info_.eof_row_.sample_ += 1;
  receive_info_.eof_row_.row_step_ = ReceiveBridgeOp::RowStep::kAfterReceiveMsg;

  // End() is commented out because it might never be called due to the lack of EOF when EpochCtrl is -1
  // Handle eof logic, this code might never be reached if epoch_ctrl = -1.
  RETURN_IF_NOT_OK(worker_in_queues_[NextWorkerID()]->Add(std::move(new_row)));

  // Quit all workers, this code might never be reached if EpochCtrl is -1.
  for (int32_t wkr_id = 0; wkr_id < num_workers_; wkr_id++) {
    RETURN_IF_NOT_OK(SendQuitFlagToWorker(NextWorkerID()));
  }

  return Status::OK();
}

// Private function for worker/thread to loop continuously. It comprises the main
// logic of ReceiveBridgeOp: getting the data from previous Op, validating user specified column names,
// applying a list of TensorOps to each of the data, process the results and then
// pushing them back to ReceiveBridgeOp's output Connector to be fetched by the next Op.
Status ReceiveBridgeOp::WorkerEntry(int32_t worker_id) {
  // Handshake with TaskManager that thread creation is successful.
  TaskManager::FindMe()->Post();

  RETURN_IF_NOT_OK(CollectOpInfoStart(this->NameWithID(), "ReceiveBridgeGet"));
  TensorRow in_row;
  // Fetch next data from parent node
  RETURN_IF_NOT_OK(worker_in_queues_[static_cast<const int>(worker_id)]->PopFront(&in_row));
  RETURN_IF_NOT_OK(CollectOpInfoEnd(this->NameWithID(), "ReceiveBridgeGet", {{"TensorRowFlags", in_row.FlagName()}}));
  RETURN_IF_NOT_OK(CollectOpInfoStart(this->NameWithID(), "ReceiveBridgeProcess"));

  // Now that init work is done, drop into the main fetching loop.
  // receive op does not use child iterator, and it needs to manually handle eoe and eof's itself
  // rather than use the base-class defaults.
  while (true) {
    // Handle special logic where row carries a ctrl flag.
    if (in_row.Flags() != TensorRow::kFlagNone) {
      RETURN_IF_NOT_OK(
        CollectOpInfoEnd(this->NameWithID(), "ReceiveBridgeProcess", {{"TensorRowFlags", in_row.FlagName()}}));
      if (in_row.quit()) {
        break;
      }
    }
    // Push the row onto the connector for next operator to consume.
    RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(std::move(in_row)));

    RETURN_IF_NOT_OK(CollectOpInfoStart(this->NameWithID(), "ReceiveBridgeGet"));
    // Fetch next data from parent node
    RETURN_IF_NOT_OK(worker_in_queues_[static_cast<const int>(worker_id)]->PopFront(&in_row));
    RETURN_IF_NOT_OK(CollectOpInfoEnd(this->NameWithID(), "ReceiveBridgeGet", {{"TensorRowFlags", in_row.FlagName()}}));
    RETURN_IF_NOT_OK(CollectOpInfoStart(this->NameWithID(), "ReceiveBridgeProcess"));
  }

  return Status::OK();
}

Status ReceiveBridgeOp::SendQuitFlagToWorker(int32_t worker_id) {
  TensorRow quit_flag(TensorRow::kFlagQuit);
  RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->Add(std::move(quit_flag)));
  return Status::OK();
}

Status ReceiveBridgeOp::GetNextRowPullMode(TensorRow *const row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_[0]->GetNextRowPullMode(&new_row));
  if (new_row.eoe()) {
    UpdateRepeatAndEpochCounter();
  }
  (*row) = std::move(new_row);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
