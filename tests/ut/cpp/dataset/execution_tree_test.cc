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
#include <string>
#include "dataset/util/circular_pool.h"
#include "dataset/core/client.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/engine/datasetops/shuffle_op.h"
#include "dataset/engine/datasetops/source/tf_reader_op.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestExecutionTree : public UT::DatasetOpTesting {
 public:

};


// Construct some tree nodes and play with them
TEST_F(MindDataTestExecutionTree, TestExecutionTree1) {
  MS_LOG(INFO) << "Doing MindDataTestExecutionTree1.";

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();
  ASSERT_NE(my_tree, nullptr);
  uint32_t shuffle_size = 32;
  uint32_t connector_size = 8;


  std::shared_ptr<ShuffleOp> leaf_op1 =
      std::make_shared<ShuffleOp>(shuffle_size, 0, connector_size, false, 32);
  ASSERT_NE(leaf_op1, nullptr);
  my_tree->AssociateNode(leaf_op1);
  shuffle_size = 16;
  std::shared_ptr<ShuffleOp> leaf_op2 =
      std::make_shared<ShuffleOp>(shuffle_size, 0,  connector_size, false, 32);
  ASSERT_NE(leaf_op2, nullptr);
  my_tree->AssociateNode(leaf_op2);
  shuffle_size = 8;
  std::shared_ptr<ShuffleOp> parent_op =
      std::make_shared<ShuffleOp>(shuffle_size, 0, connector_size, false, 32);
  ASSERT_NE(parent_op, nullptr);
  my_tree->AssociateNode(parent_op);

  // It's up to you if you want to use move semantic or not here.
  // By using move, we transfer ownership of the child to the parent.
  // If you do not use move,
  // we get a reference count bump to the pointer and you have
  // your own pointer to it, plus the parent has a copy of the pointer.
  parent_op->AddChild(std::move(leaf_op1));
  parent_op->AddChild(std::move(leaf_op2));
  shuffle_size = 4;
  std::shared_ptr<DatasetOp> root_op =
      std::make_shared<ShuffleOp>(shuffle_size, 0, connector_size, false, 32);
  my_tree->AssignRoot(root_op);
  root_op->AddChild(parent_op);
  ASSERT_NE(root_op, nullptr);
  // Testing Iterator
  MS_LOG(INFO) << "Testing Tree Iterator from root.";
  for (auto itr = my_tree->begin(); itr != my_tree->end(); ++itr) {
    itr->Print(std::cout, false);
  }
  MS_LOG(INFO) << "Finished testing Tree Iterator from root.";
  MS_LOG(INFO) << "Testing Tree Iterator from parentOp.";
  for (auto itr = my_tree->begin(parent_op); itr != my_tree->end(); ++itr) {
    itr->Print(std::cout, false);
  }
  MS_LOG(INFO) << "Finished testing Tree Iterator from parentOp.";

  // At this point, since move semantic was used,
  // I don't have any operator access myself now.
  // Ownership is fully transferred into the tree.

  // explicitly drive tree destruction rather than
  // wait for descoping (to examine in debugger
  // just to see it work)
  my_tree.reset();
  MS_LOG(INFO) << "Done.";
}

// Construct some tree nodes and play with them
TEST_F(MindDataTestExecutionTree, TestExecutionTree2) {
  MS_LOG(INFO) << "Doing MindDataTestExecutionTree2.";
  Status rc;
  auto my_tree = std::make_shared<ExecutionTree>();

  std::string dataset_path = datasets_root_path_ + "/testDataset1/testDataset1.data";
  std::shared_ptr<TFReaderOp> my_tfreader_op;
  TFReaderOp::Builder()
      .SetDatasetFilesList({dataset_path})
      .SetRowsPerBuffer(2)
      .SetWorkerConnectorSize(2)
      .SetNumWorkers(2)
      .Build(&my_tfreader_op);

  my_tree->AssociateNode(my_tfreader_op);
  my_tree->AssignRoot(my_tfreader_op);

  // prepare the tree
  my_tree->Prepare();

  // Launch the tree execution to kick off threads
  // and start running the pipeline
  MS_LOG(INFO) << "Launching my tree.";
  my_tree->Launch();

  // Simulate a parse of data from our pipeline.
  std::shared_ptr<DatasetOp> root_node = my_tree->root();

  // Start the loop of reading from our pipeline using iterator
  MS_LOG(INFO) << "Testing DatasetIterator in testTree2.";
  DatasetIterator di(my_tree);
  TensorRow buffer;
  rc = di.FetchNextTensorRow(&buffer);
  EXPECT_TRUE(rc.IsOk());

  while (!buffer.empty()) {
    rc = di.FetchNextTensorRow(&buffer);
    EXPECT_TRUE(rc.IsOk());
  }
}

// Construct some tree nodes and play with them
TEST_F(MindDataTestExecutionTree, TestExecutionTree3) {
  MS_LOG(INFO) << "Doing MindDataTestExecutionTree3.";
}
