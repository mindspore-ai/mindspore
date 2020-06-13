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
#ifndef DATASET_ENGINE_DATASETOPS_DATASET_OP_H_
#define DATASET_ENGINE_DATASETOPS_DATASET_OP_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "dataset/core/constants.h"
#include "dataset/engine/db_connector.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Forward declare
class ExecutionTree;

class DataBuffer;

class NodePass;

// The base class DatasetOp is the main tree node.  It is an abstract class, so
// the actual implementation of the operators will be derived from here.
class DatasetOp : public std::enable_shared_from_this<DatasetOp> {
  // Allow execution tree to access internal members
  friend class ExecutionTree;

 public:
  static constexpr int32_t kInvalidOperatorId = -1;

  // Flags that control operator runtime behaviours
  enum OpControlFlags {
    kDeOpNone = 0,
    kDeOpRepeated = 1,        // Operator is a leaf node in a repeat path
    kDeOpLastRepeat = 1 << 1  // We are in the last repeat loop
  };

  // Flags that control operator runtime behaviours
  enum OpState { kDeOpRunning = 0, kDeOpIdle = 1, kDeOpTerminated };

  // Constructor
  // @param op_connector_size - The size for the output connector of this operator.
  explicit DatasetOp(int32_t op_connector_size);

  // Destructor
  virtual ~DatasetOp() { tree_ = nullptr; }

  // Adds a operator to become our child.
  // @param child - shared pointer to the child to add.
  Status AddChild(std::shared_ptr<DatasetOp> child);

  // Getter function to get a shared pointer to our child
  // @param child_index - An operator can have n children. Indicates choose which child to return.
  std::shared_ptr<DatasetOp> child(int32_t child_index) const;

  // Creates the connector within this operator
  // @param num_producers - number of threads that write into this connector
  // @param num_consumers - number of threads that read from this connector
  void CreateConnector(int32_t num_producers, int32_t num_consumers);

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  virtual void Print(std::ostream &out, bool show_all) const;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param dO - reference to the DatasetOp to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const DatasetOp &dO) {
    dO.Print(out, false);
    return out;
  }

  // Class functor operator ().
  // DatasetOps operate by launching a thread (see ExecutionTree).
  // This pure virtual version makes the requirement that derived classes must provide a functor
  // that will execute their main runtime loop code.
  // @return Status - The error code return
  virtual Status operator()() = 0;

  // Gets the next buffer from the given child
  // @notes See GetNextInput for similar function that has built-in message handling
  // @param p_buffer - The shared pointer for the fetched buffer to return (by reference)
  // @param worker_id - The worker id
  // @return Status - The error code return
  virtual Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id) {
    return GetNextBuffer(p_buffer, worker_id, false);
  }

  // Gets the next buffer from the given child
  // @notes See GetNextInput for similar function that has built-in message handling
  // @param p_buffer - The shared pointer for the fetched buffer to return (by reference)
  // @return Status - The error code return
  virtual Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer) { return GetNextBuffer(p_buffer, 0, false); }

  // Gets the next buffer from the given child
  // @notes See GetNextInput for similar function that has built-in message handling
  // @param p_buffer - The shared pointer for the fetched buffer to return (by reference)
  // @param worker_id - The worker id
  // @param retry_if_eoe Set this flag to true to allow calling pop() again after the first pop() returns EOE.
  // @return Status - The error code return
  virtual Status GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe);

  // Gets the next buffer from the given child .  This function also has built-in eoe and eof
  // message handling so that child classes don't have to manually code pass-through logic when
  // those messages are received.
  // @param p_buffer - The shared pointer for the fetched buffer to return (by reference)
  // @param worker_id - The worker id
  // @return Status - The error code return
  Status GetNextInput(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id = 0, int32_t child_index = 0);

  // Performs handling for when an eoe message is received.
  // The base class implementation simply flows the eoe message to output. Derived classes
  // may override if they need to perform special eoe handling.
  // @param worker_id - The worker id
  // @return Status - The error code return
  virtual Status EoeReceived(int32_t worker_id);

  // Performs handling for when an eof message is received.
  // The base class implementation simply flows the eof message to output. Derived classes
  // may override if they need to perform special eof handling.
  // @param worker_id - The worker id
  // @return Status - The error code return
  virtual Status EofReceived(int32_t worker_id);

  // Derived classes may implement the reset function if the operator is stateful and needs
  // specific reset handling that is not contained in this common code version of the reset
  // @return Status - The error code return
  virtual Status Reset();

  // This calls the reset function on this subtree in pre-order
  // @return Status - The error code return
  virtual Status ResetSubtree() {
    RETURN_IF_NOT_OK(Reset());
    for (const auto &c : child_) {
      RETURN_IF_NOT_OK(c->ResetSubtree());
    }
    return Status::OK();
  }

  // During tree prepare phase, operators may have specific pre-operations to perform depending on
  // their role.
  // @notes Derived versions of this function should always call it's superclass version first
  // before providing their own implementations.
  virtual Status PrepareNodePreAction();

  // During tree prepare phase, operators may have specific post-operations to perform depending on
  // their role.
  // @notes Derived versions of this function should always call it's superclass version first
  // before providing their own implementations.
  virtual Status PrepareNodePostAction();

  // Getter function
  // @return The operator id
  int32_t id() const { return operator_id_; }

  // Getter function
  // @return The prepare flags
  virtual uint32_t PrepareFlags() const;

  // Getter function
  // @return The number of workers in this op
  virtual int32_t num_workers() const = 0;

  // Getter function
  // @return The number of threads consuming from previous op.
  virtual int32_t num_consumers() const = 0;

  // Getter function
  // @return The number of threads producing to the output connector.
  virtual int32_t num_producers() const = 0;

  // Getter function
  // @return T/F if this is an inlined operator
  bool inlined() const { return (oc_queue_size_ == 0); }

  // Setter function
  // @return Sets the control flags
  void set_control_flag(uint64_t flag) { BitSet(&op_ctrl_flags_, flag); }

  // Register the internal worker connectors. No op unless it is a parallel op
  // @return Status
  virtual Status RegisterWorkerConnectors() { return Status::OK(); }

  // Getter for the column name mapping
  // @return The returned map
  std::unordered_map<std::string, int32_t> column_name_id_map() const { return column_name_id_map_; }

  // Checks if the column name map has been set up yet for this op
  // @return - T/F if the operator has the map set up
  bool HasColumnNameMap() const { return (column_name_id_map_.empty()); }

  // gives a string output for the column map for handy debug printing
  // @return - the column name map as a string
  std::string ColumnNameMapAsString() const;

  // Getter function
  // @return connector size of current op
  int32_t ConnectorSize() const {
    if (!inlined()) {
      return out_connector_->size();
    }
    // Return -1 for inlined op
    return -1;
  }

  // Getter function
  // @return connector size of current op
  int32_t ConnectorCapacity() const {
    if (!inlined()) {
      return out_connector_->capacity();
    }
    // Return -1 for inlined op
    return -1;
  }

  // Getter function
  // @return connector size of child op
  int32_t ChildOpConnectorSize(int32_t child_index = 0) const { return child_[child_index]->ConnectorSize(); }

  // Getter function
  // @return connector capacity of child op
  int32_t ChildOpConnectorCapacity(int32_t child_index = 0) const { return child_[child_index]->ConnectorCapacity(); }

  // Children Getter
  // @return Vector of Children
  std::vector<std::shared_ptr<DatasetOp>> Children() const { return child_; }

  // Base method for NodePass visit.
  // Subclass needs to override this if it requires special node visit access.
  // Check "dataset/engine/opt/pass.h" for more details.
  // @return Statue of the node visit
  virtual Status Accept(NodePass *p, bool *modified);

  // Op name getter
  // @return Name of the current Op
  virtual std::string Name() const { return "DatasetOp"; }

  // Execution Tree getter
  // @return Pointer to the ExecutionTree the current op belongs to, no ownership
  ExecutionTree *Tree() { return tree_; }

 protected:
  // Adds a parent operator to this operator
  // @notes External callers do not have access to this function.
  // @param parent - The parent node to add
  void AddParent(const DatasetOp *parent);

  // A helper function for providing an assignment of the column name map.
  // This grabs the map from child 0 and assigns it into this op.
  // Can only be used if number of children is 1.
  // @return - Status
  Status AssignColMapFromChild();

  std::vector<std::shared_ptr<DatasetOp>> child_;                // Child nodes
  std::vector<const DatasetOp *> parent_;                        // Parent nodes. No ownership and read-only
  int32_t oc_queue_size_;                                        // Capacity for each out_connector_
  int32_t operator_id_;                                          // Generated id for the node
  ExecutionTree *tree_;                                          // Back pointer to our tree.
  OpState state_;                                                // The state of the operator, Running, Idle, Terminated
  uint32_t op_ctrl_flags_;                                       // Flags for the operator
  std::unique_ptr<DbConnector> out_connector_;                   // Output Connector
  std::unordered_map<std::string, int32_t> column_name_id_map_;  // Mapping between col index and col name
  bool first_fetch_;                                             // For use when setting column map
  std::mutex column_name_map_mutex_;                             // For protecting shared access to the column map

 private:
  // Sets the operator id.
  // @notes No public interface.  Only the class itself, or it's friend the execution tree can set
  // this
  // @param op_id - the Id value to set into the operator
  void set_id(int32_t op_id) { operator_id_ = op_id; }

  // Sets the tree into the op so that the operator has a back pointer to the tree.
  // @param tree - the tree to assign to the op.
  void set_tree(ExecutionTree *tree) { tree_ = tree; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_DATASET_OP_H_
