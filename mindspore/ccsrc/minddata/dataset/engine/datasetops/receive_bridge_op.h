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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RECEIVE_BRIDGE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RECEIVE_BRIDGE_OP_H_

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/api/python/python_mp.h"
#include "minddata/dataset/callback/ds_callback.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/wait_post.h"
#include "minddata/dataset/core/shared_memory_queue.h"
#include "minddata/dataset/core/message_queue.h"

namespace mindspore {
namespace dataset {
class ReceiveBridgeOp : public ParallelOp<TensorRow, TensorRow> {
 public:
  enum RowStep {
    kNone = 0,
    kBeginReceiveMsg,
    kAfterReceiveMsg,
    kBeginSendMsg,
    kAfterSendMsg,
  };
  struct RowStatus {
    uint64_t sample_;
    RowStep row_step_;
  };
  struct ReceiveInfo {
    RowStatus normal_row_;
    RowStatus eoe_row_;
    RowStatus eof_row_;
  };

  ReceiveBridgeOp(int32_t op_connector_size, SharedMemoryQueue receive_queue, MessageQueue msg_queue);

  // Destructor
  ~ReceiveBridgeOp();

  // A print method typically used for debugging
  // @param out The output stream to write output to
  // @param show_all A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out reference to the output stream being overloaded
  // @param mo reference to the ReceiveBridgeOp to display
  // @return the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const ReceiveBridgeOp &mo) {
    mo.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // This main thread creates local queues, pulls TensorRow from the previous
  // op's Connector and distributes them to the local queues. Workers pull from the local queues.
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kReceiveBridgeOp; }

  /// Send quit flag row to worker at worker_id to make it exit
  /// \param worker_id id of the worker
  /// \return Status code
  Status SendQuitFlagToWorker(int32_t worker_id) override;

  Status GetNextRowPullMode(TensorRow *const row) override;

  void SetSubprocessID(int pid) { subprocess_pid_ = pid; }

  Status MonitorIndependentDatasetProcess();

 private:
  std::unique_ptr<ChildIterator> child_iterator_;  // An iterator for fetching.

  // Private function for worker/thread to loop continuously. It comprises the main
  // logic of ReceiveBridgeOp: getting the data from previous Op, validating user specified column names,
  // applying a list of TensorOps to each of the data, process the results and then
  // pushing them back to ReceiveBridgeOp's output Connector to be fetched by the next Op.
  // @param worker_id The id assigned to this thread/worker upon creation.
  // @return Status The status code returned
  Status WorkerEntry(int32_t worker_id) override;  //  In: workerId assigned by tree_

 private:
  SharedMemoryQueue receive_queue_;  // get data from independent process
  MessageQueue msg_queue_;           // get msg from independent process
  ReceiveInfo receive_info_;         // receive info, including msgrcv and msgsnd status
  int subprocess_pid_;               // the independent dataset process id
  bool subprocess_status_;           // the independent dataset process status, 1: running, 0: exit

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RECEIVE_BRIDGE_OP_H_
