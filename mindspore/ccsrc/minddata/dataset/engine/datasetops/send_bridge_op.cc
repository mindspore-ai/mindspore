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
#include "minddata/dataset/engine/datasetops/send_bridge_op.h"

#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
// Constructor of SendBridgeOp
SendBridgeOp::SendBridgeOp(int32_t op_connector_size, SharedMemoryQueue send_queue, MessageQueue msg_queue)
    : ParallelOp(1, op_connector_size), send_queue_(send_queue), msg_queue_(msg_queue) {
  send_info_.normal_row_.sample_ = 0;
  send_info_.normal_row_.row_step_ = SendBridgeOp::RowStep::kNone;
  send_info_.eoe_row_.sample_ = 0;
  send_info_.eoe_row_.row_step_ = SendBridgeOp::RowStep::kNone;
  send_info_.eof_row_.sample_ = 0;
  send_info_.eof_row_.row_step_ = SendBridgeOp::RowStep::kNone;
}

SendBridgeOp::~SendBridgeOp() {
  std::string err_msg = "Dataset SendOp normal_row: " + std::to_string(send_info_.normal_row_.sample_) +
                        ", status: " + std::to_string(send_info_.normal_row_.row_step_) +
                        ", eoe_row: " + std::to_string(send_info_.eoe_row_.sample_) +
                        ", status: " + std::to_string(send_info_.eoe_row_.row_step_) +
                        ", eof_row: " + std::to_string(send_info_.eof_row_.sample_) +
                        ", status: " + std::to_string(send_info_.eof_row_.row_step_);
  if (send_info_.normal_row_.row_step_ == 0 || send_info_.normal_row_.row_step_ == SendBridgeOp::RowStep::kNone) {
    return;
  }
  if (send_info_.normal_row_.row_step_ != SendBridgeOp::RowStep::kAfterReceiveMsg) {
    MS_LOG(WARNING) << err_msg;
  }
  if (send_info_.eoe_row_.row_step_ == SendBridgeOp::RowStep::kNone) {
    return;
  }
  if (send_info_.normal_row_.row_step_ == SendBridgeOp::RowStep::kAfterReceiveMsg &&
      send_info_.eoe_row_.row_step_ != SendBridgeOp::RowStep::kAfterReceiveMsg) {
    MS_LOG(WARNING) << err_msg;
  }
  if (send_info_.eof_row_.row_step_ == SendBridgeOp::RowStep::kNone) {
    return;
  }
  if (send_info_.normal_row_.row_step_ == SendBridgeOp::RowStep::kAfterReceiveMsg &&
      send_info_.eoe_row_.row_step_ == SendBridgeOp::RowStep::kAfterReceiveMsg &&
      send_info_.eof_row_.row_step_ != SendBridgeOp::RowStep::kAfterSendMsg) {
    MS_LOG(WARNING) << err_msg;
  }
}

// A print method typically used for debugging
void SendBridgeOp::Print(std::ostream &out, bool show_all) const {
  // Call the super class for displaying any common 1-liner info
  ParallelOp::Print(out, show_all);
  // Then show any custom derived-internal 1-liner info for this op
  out << "\n";
}

// This class functor will provide the master loop that drives the logic for performing the work
Status SendBridgeOp::operator()() {
  RETURN_IF_NOT_OK(RegisterAndLaunchThreads());

  // Synchronize with TaskManager
  TaskManager::FindMe()->Post();

  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));

  while (!new_row.eof()) {
    while (!new_row.eoe()) {
      // The NextWorkerID() should always 0, because SendBridgeOp is a single thread op
      RETURN_IF_NOT_OK(worker_in_queues_[NextWorkerID()]->Add(std::move(new_row)));
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    }

    // Propagate the eoe row to worker
    RETURN_IF_NOT_OK(worker_in_queues_[NextWorkerID()]->Add(std::move(new_row)));
    UpdateRepeatAndEpochCounter();
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  }
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
// logic of SendBridgeOp: getting the data from previous Op, validating user specified column names,
// applying a list of TensorOps to each of the data, process the results and then
// pushing them back to SendBridgeOp's output Connector to be fetched by the next Op.
Status SendBridgeOp::WorkerEntry(int32_t worker_id) {
  // Handshake with TaskManager that thread creation is successful.
  TaskManager::FindMe()->Post();

  send_info_.normal_row_.sample_ = 0;

  RETURN_IF_NOT_OK(CollectOpInfoStart(this->NameWithID(), "SendBridgeGet"));
  TensorRow in_row;
  // Fetch next data from parent node
  RETURN_IF_NOT_OK(worker_in_queues_[static_cast<const int>(worker_id)]->PopFront(&in_row));
  RETURN_IF_NOT_OK(CollectOpInfoEnd(this->NameWithID(), "SendBridgeGet", {{"TensorRowFlags", in_row.FlagName()}}));
  RETURN_IF_NOT_OK(CollectOpInfoStart(this->NameWithID(), "SendBridgeProcess"));

  uint64_t *current_sample = nullptr;
  SendBridgeOp::RowStep *current_step = nullptr;

  // Now that init work is done, drop into the main fetching loop.
  // send op does not use child iterator, and it needs to manually handle eoe and eof's itself
  // rather than use the base-class defaults.
  while (true) {
    // Handle special logic where row carries a ctrl flag.
    if (in_row.Flags() != TensorRow::kFlagNone) {
      if (in_row.quit()) {
        break;
      }
    }

    current_sample = &send_info_.normal_row_.sample_;
    current_step = &send_info_.normal_row_.row_step_;
    if (in_row.eoe()) {
      current_sample = &send_info_.eoe_row_.sample_;
      current_step = &send_info_.eoe_row_.row_step_;
    } else if (in_row.eof()) {
      current_sample = &send_info_.eof_row_.sample_;
      current_step = &send_info_.eof_row_.row_step_;
    }

    // Copy the in_row to shared memory
    RETURN_IF_NOT_OK(send_queue_.FromTensorRow(in_row));

    // Send msg to the main process by msg_queue_
    *current_step = SendBridgeOp::RowStep::kBeginSendMsg;
    RETURN_IF_NOT_OK(msg_queue_.MsgSnd(kWorkerSendDataMsg, send_queue_.GetShmID(), send_queue_.GetShmSize()));
    *current_step = SendBridgeOp::RowStep::kAfterSendMsg;

    *current_sample += 1;

    MS_LOG(INFO) << "Dataset SendOp normal_row: " << std::to_string(send_info_.normal_row_.sample_)
                 << ", status: " << send_info_.normal_row_.row_step_
                 << ", eoe_row: " << std::to_string(send_info_.eoe_row_.sample_)
                 << ", status: " << send_info_.eoe_row_.row_step_
                 << ", eof_row: " << std::to_string(send_info_.eof_row_.sample_)
                 << ", status: " << send_info_.eof_row_.row_step_;

    // all tensor row had been sent to main process from independent process
    if (in_row.eof()) {
      break;
    }
    // Get msg from the main process by msg_queue_
    *current_step = SendBridgeOp::RowStep::kBeginReceiveMsg;
    auto ret = msg_queue_.MsgRcv(kMasterSendDataMsg);
    if (ret != Status::OK()) {
      *current_step = SendBridgeOp::RowStep::kAfterReceiveMsg;
      tree_->SetFinished();  // the independent dataset process will exit
      if (msg_queue_.state_ != MessageQueue::State::kReleased) {
        return ret;
      }
      return Status::OK();
    }
    *current_step = SendBridgeOp::RowStep::kAfterReceiveMsg;

    RETURN_IF_NOT_OK(
      CollectOpInfoEnd(this->NameWithID(), "SendBridgeProcess", {{"TensorRowFlags", in_row.FlagName()}}));
    RETURN_IF_NOT_OK(CollectOpInfoStart(this->NameWithID(), "SendBridgeGet"));

    // Fetch next data from parent node
    RETURN_IF_NOT_OK(worker_in_queues_[static_cast<const int>(worker_id)]->PopFront(&in_row));

    RETURN_IF_NOT_OK(CollectOpInfoEnd(this->NameWithID(), "SendBridgeGet", {{"TensorRowFlags", in_row.FlagName()}}));
    RETURN_IF_NOT_OK(CollectOpInfoStart(this->NameWithID(), "SendBridgeProcess"));
  }

  RETURN_IF_NOT_OK(CollectOpInfoEnd(this->NameWithID(), "SendBridgeProcess", {{"TensorRowFlags", in_row.FlagName()}}));
  return Status::OK();
}

Status SendBridgeOp::SendQuitFlagToWorker(int32_t worker_id) {
  TensorRow quit_flag(TensorRow::kFlagQuit);
  RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->Add(std::move(quit_flag)));
  return Status::OK();
}

Status SendBridgeOp::GetNextRowPullMode(TensorRow *const row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_[0]->GetNextRowPullMode(&new_row));
  if (new_row.eoe()) {
    UpdateRepeatAndEpochCounter();
  }
  (*row) = std::move(new_row);
  return Status::OK();
}

MessageQueue::State SendBridgeOp::MessageQueueState() { return msg_queue_.MessageQueueState(); }

MessageQueue SendBridgeOp::GetMessageQueue() { return msg_queue_; }
}  // namespace dataset
}  // namespace mindspore
