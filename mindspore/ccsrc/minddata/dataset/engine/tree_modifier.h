/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_MODIFIER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_MODIFIER_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/tree_adapter.h"

constexpr int64_t queue = 10;

namespace mindspore {
namespace dataset {
class DatasetNode;

/// A pure virtual class to be used as a base for all pipeline modification requests.
class ChangeRequest {
 public:
  /// Default constructor
  ChangeRequest() = default;
  virtual ~ChangeRequest() = default;

  /// Pure virtual method. Subclasses should override this function and implement the actual change to the give
  /// operator.
  /// \param op pointer to the operator that the change will be applied on
  /// \return Status return Status code
  virtual Status ApplyChange(DatasetOp *op) = 0;
};

using ChangeRequestPtr = std::shared_ptr<ChangeRequest>;

/// ChangeRequest to add n workers to an operator.
class ChangeNumWorkersRequest : public ChangeRequest {
 public:
  /// Constructor
  /// \param num_workers number of workeres to be added to the operator. Default to 1.
  explicit ChangeNumWorkersRequest(int32_t num_workers = 1) : num_workers_(num_workers) {}
  virtual ~ChangeNumWorkersRequest() = default;

  /// Actual change to add n workers
  /// \param op pointer to the operator that the change will be applied on
  /// \return Status return Status code
  Status ApplyChange(DatasetOp *op) override;

 private:
  int32_t num_workers_;
};

/// ChangeRequest to change the size of the oupout connector of an operators.
class ResizeConnectorRequest : public ChangeRequest {
 public:
  /// Constructor
  /// \param new_size new queue size.
  explicit ResizeConnectorRequest(int32_t new_size) : new_size_(new_size) {}
  virtual ~ResizeConnectorRequest() = default;

  /// Actual change to resize the output connector of the given operator
  /// \param op pointer to the operator that the change will be applied on
  /// \return Status return Status code
  Status ApplyChange(DatasetOp *op) override {
    RETURN_IF_NOT_OK(op->OutputConnector()->Resize(new_size_));
    return Status::OK();
  }

 private:
  int32_t new_size_;
};

/// A callback class used by Aututune to queue changes for operators
class AutotuneCallback : public DSCallback {
 public:
  AutotuneCallback(int32_t step_size, DatasetOp *op)
      : DSCallback(step_size), op_(op), change_request_queue_(std::make_unique<Queue<ChangeRequestPtr>>(queue)) {}
  virtual ~AutotuneCallback() = default;

  Status DSNStepBegin(const CallbackParam &cb_param) override;
  Status DSBegin(const CallbackParam &cb_param) override;
  Status DSEpochBegin(const CallbackParam &cb_param) override;
  Status DSEnd(const CallbackParam &cb_param) override;
  Status DSEpochEnd(const CallbackParam &cb_param) override;
  Status DSNStepEnd(const CallbackParam &cb_param) override;

  bool IsBeginNeeded() override;
  bool IsEpochBeginNeeded() override;
  bool IsNStepBeginNeeded() override;
  bool IsEndNeeded() override;
  bool IsEpochEndNeeded() override;
  bool IsNStepEndNeeded() override;

  ///  Push a change request to the queue of the callback.
  /// \param change_request Shared pointer to the change request to be pushed to the queue.
  /// \return Status return Status code
  Status PushChangeRequest(ChangeRequestPtr change_request);

 private:
  DatasetOp *op_;
  std::unique_ptr<Queue<ChangeRequestPtr>> change_request_queue_;
};

/// Main class to handle modification of the ExecutionTree used by AutoTune
class TreeModifier {
  // friend with TreeAdapter to access the ExecutionTree
  friend TreeAdapter;

 public:
  /// Constructor to create a TreeModifier given a TreeAdapter
  /// \param adapter TreeAdapter
  explicit TreeModifier(const TreeAdapter *adapter);

  /// Constructor to create a TreeModifier given an ExecutionTree
  /// \param tree ExecutionTree
  explicit TreeModifier(ExecutionTree *tree) : tree_(tree) {
    // loop over all ops to create AutotuneCallback and register it.
    for (auto itr = tree_->begin(); itr != tree_->end(); ++itr) {
      auto cb = std::make_shared<AutotuneCallback>(1, itr.get().get());
      itr->AddCallbacks({cb});
      (void)callbacks.insert(std::make_pair(itr->id(), cb));
    }
  }

  /// Add changeRequest to the callback associated with the op.
  /// \param op_id Operator ID
  /// \param change_request Pointer to the change request
  /// \return Status return Status code
  Status AddChangeRequest(int32_t op_id, const ChangeRequestPtr &change_request) {
    num_requests_++;
    RETURN_IF_NOT_OK(callbacks[op_id]->PushChangeRequest(change_request));
    return Status::OK();
  }

  /// \brief Get the number of change requests received
  /// \return Number of change requests received
  uint64_t GetRequestsCount() const { return num_requests_; }

 private:
  ExecutionTree *tree_;
  std::map<int32_t, std::shared_ptr<AutotuneCallback>> callbacks;
  uint64_t num_requests_ = 0;  // counter for number of requests received
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_MODIFIER_H_
