/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/iterator.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/include/datasets.h"

namespace mindspore {
namespace dataset {
namespace api {

// Get the next row from the data pipeline.
void Iterator::GetNextRow(TensorMap *row) {
  Status rc = iterator_->GetNextAsMap(row);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetNextRow: Failed to get next row. Error status: " << rc;
    row->clear();
  }
}

// Get the next row from the data pipeline.
void Iterator::GetNextRow(TensorVec *row) {
  TensorRow tensor_row;
  Status rc = iterator_->FetchNextTensorRow(&tensor_row);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetNextRow: Failed to get next row. Error status: " << rc;
    row->clear();
  }
  // Generate a vector as return
  row->clear();
  std::copy(tensor_row.begin(), tensor_row.end(), std::back_inserter(*row));
}

// Shut down the data pipeline.
void Iterator::Stop() {
  // Releasing the iterator_ unique_ptre. This should trigger the destructor of iterator_.
  iterator_.reset();

  // Release ownership of tree_ shared pointer. This will decrement the ref count.
  tree_.reset();
}

// Function to build and launch the execution tree.
Status Iterator::BuildAndLaunchTree(std::shared_ptr<Dataset> ds) {
  // One time init
  Status rc;
  rc = GlobalInit();
  RETURN_IF_NOT_OK(rc);

  // Instantiate the execution tree
  tree_ = std::make_shared<ExecutionTree>();

  // Iterative BFS converting Dataset tree into runtime Execution tree.
  std::queue<std::pair<std::shared_ptr<Dataset>, std::shared_ptr<DatasetOp>>> q;

  if (ds == nullptr) {
    RETURN_STATUS_UNEXPECTED("Input is null pointer");
  } else {
    // Convert the current root node.
    auto root_ops = ds->Build();
    if (root_ops.empty()) {
      RETURN_STATUS_UNEXPECTED("Node operation returned nothing");
    }

    // Iterate through all the DatasetOps returned by Dataset's Build(), associate them
    // with the execution tree and add the child and parent relationship between the nodes
    // Note that some Dataset objects might return more than one DatasetOps
    // e.g. MapDataset will return [ProjectOp, MapOp] if project_columns is set for MapDataset
    std::shared_ptr<DatasetOp> prev_op = nullptr;
    for (auto op : root_ops) {
      RETURN_IF_NOT_OK(tree_->AssociateNode(op));
      if (prev_op != nullptr) {
        RETURN_IF_NOT_OK(prev_op->AddChild(op));
      }
      prev_op = op;
    }
    // Add the last DatasetOp to the queue to be BFS.
    q.push(std::make_pair(ds, root_ops.back()));

    // Traverse down to the children and convert them to the corresponding DatasetOps (i.e. execution tree nodes)
    while (!q.empty()) {
      auto node_pair = q.front();
      q.pop();
      // Iterate through all the direct children of the first element in our BFS queue
      for (auto child : node_pair.first->children) {
        auto child_ops = child->Build();
        if (child_ops.empty()) {
          RETURN_STATUS_UNEXPECTED("Node operation returned nothing");
        }
        auto node_op = node_pair.second;
        // Iterate through all the DatasetOps returned by calling Build on the last Dataset object, associate them
        // with the execution tree and add the child and parent relationship between the nodes
        // Note that some Dataset objects might return more than one DatasetOps
        // e.g. MapDataset will return MapOp and ProjectOp if project_columns is set for MapDataset
        for (auto child_op : child_ops) {
          RETURN_IF_NOT_OK(tree_->AssociateNode(child_op));
          RETURN_IF_NOT_OK(node_op->AddChild(child_op));
          node_op = child_op;
        }
        // Add the child and the last element of the returned DatasetOps (which is now the leaf node in our current
        // execution tree) to the BFS queue
        q.push(std::make_pair(child, child_ops.back()));
      }
    }
    RETURN_IF_NOT_OK(tree_->AssignRoot(root_ops.front()));
  }

  // Launch the execution tree.
  RETURN_IF_NOT_OK(tree_->Prepare());
  RETURN_IF_NOT_OK(tree_->Launch());
  iterator_ = std::make_unique<DatasetIterator>(tree_);
  RETURN_UNEXPECTED_IF_NULL(iterator_);

  return rc;
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
