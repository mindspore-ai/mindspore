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
#include "minddata/dataset/util/circular_pool.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

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

  std::shared_ptr<ShuffleOp> leaf_op1 = std::make_shared<ShuffleOp>(shuffle_size, 0, connector_size, false);
  ASSERT_NE(leaf_op1, nullptr);
  my_tree->AssociateNode(leaf_op1);
  shuffle_size = 16;
  std::shared_ptr<ShuffleOp> leaf_op2 = std::make_shared<ShuffleOp>(shuffle_size, 0, connector_size, false);
  ASSERT_NE(leaf_op2, nullptr);
  my_tree->AssociateNode(leaf_op2);
  shuffle_size = 8;
  std::shared_ptr<ShuffleOp> parent_op = std::make_shared<ShuffleOp>(shuffle_size, 0, connector_size, false);
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
  std::shared_ptr<DatasetOp> root_op = std::make_shared<ShuffleOp>(shuffle_size, 0, connector_size, false);
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
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  auto op_connector_size = config_manager->op_connector_size();
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  std::vector<std::string> columns_to_load = {};
  std::vector<std::string> files = {dataset_path};
  std::shared_ptr<TFReaderOp> my_tfreader_op = std::make_shared<TFReaderOp>(
    1, 2, 0, files, std::move(schema), op_connector_size, columns_to_load, false, 1, 0, false);
  rc = my_tfreader_op->Init();
  ASSERT_OK(rc);
  rc = my_tree->AssociateNode(my_tfreader_op);
  ASSERT_OK(rc);
  rc = my_tree->AssignRoot(my_tfreader_op);
  ASSERT_OK(rc);

  // prepare the tree
  rc = my_tree->Prepare();
  ASSERT_OK(rc);

  // Launch the tree execution to kick off threads
  // and start running the pipeline
  MS_LOG(INFO) << "Launching my tree.";
  rc = my_tree->Launch();
  ASSERT_OK(rc);

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
TEST_F(MindDataTestExecutionTree, TestExecutionTree3) { MS_LOG(INFO) << "Doing MindDataTestExecutionTree3."; }
