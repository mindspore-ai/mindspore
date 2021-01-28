/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "utils/ms_utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include <memory>
#include <vector>
#include <iostream>

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestShuffleOp : public UT::DatasetOpTesting {

};


// Test info:
// - Dataset from testDataset1 has 10 rows, 2 columns.
// - RowsPerBuffer buffer setting of 2 divides evenly into total rows.
// - Shuffle size is multiple of rows per buffer.
//
// Tree:  shuffle over TFReader
//
//    ShuffleOp
//        |
//    TFReaderOp
//
TEST_F(MindDataTestShuffleOp, TestShuffleBasic1) {
  Status rc;
  MS_LOG(INFO) << "UT test TestShuffleBasic1.";

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testDataset1/testDataset1.data";
  std::shared_ptr<TFReaderOp> my_tfreader_op;
  rc = TFReaderOp::Builder()
      .SetDatasetFilesList({dataset_path})
      .SetRowsPerBuffer(2)
      .SetWorkerConnectorSize(16)
      .SetNumWorkers(1)
      .Build(&my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  std::shared_ptr<ShuffleOp> my_shuffle_op;
  rc = ShuffleOp::Builder().SetRowsPerBuffer(2).SetShuffleSize(4).Build(&my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());

  // Set children/root layout.
  rc = my_shuffle_op->AddChild(my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "Launching tree and begin iteration.";
  rc = my_tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  EXPECT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count << ".";

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: " << ss.str() << ".";
    }
    rc = di.FetchNextTensorRow(&tensor_list);
    EXPECT_TRUE(rc.IsOk());
    row_count++;
  }
  ASSERT_EQ(row_count, 10);

}

// Test info:
// - Dataset from testDataset1 has 10 rows, 2 columns.
// - RowsPerBuffer buffer setting of 3 does not divide evenly into total rows, thereby causing
//   partially filled buffers.
// - Shuffle size is not a multiple of rows per buffer.
// - User has provided a non-default seed value.
//
// Tree: shuffle over TFReader
//
//    ShuffleOp
//       |
//    TFReaderOp
//
TEST_F(MindDataTestShuffleOp, TestShuffleBasic2) {
  Status rc;
  MS_LOG(INFO) << "UT test TestShuffleBasic2.";

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testDataset1/testDataset1.data";
  std::shared_ptr<TFReaderOp> my_tfreader_op;
  rc = TFReaderOp::Builder()
      .SetDatasetFilesList({dataset_path})
      .SetRowsPerBuffer(3)
      .SetWorkerConnectorSize(16)
      .SetNumWorkers(2)
      .Build(&my_tfreader_op);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  std::shared_ptr<ShuffleOp> my_shuffle_op;
  rc = ShuffleOp::Builder().SetShuffleSize(4).SetShuffleSeed(100).SetRowsPerBuffer(3).Build(&my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());

  // Set children/root layout.
  rc = my_shuffle_op->AddChild(my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "Launching tree and begin iteration.";
  rc = my_tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  EXPECT_TRUE(rc.IsOk());
  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count << ".";

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: " << ss.str() << ".";
    }
    rc = di.FetchNextTensorRow(&tensor_list);
    EXPECT_TRUE(rc.IsOk());
    row_count++;
  }
  ASSERT_EQ(row_count, 10);
}

// Test info:
// - Dataset from testDataset1 has 10 rows, 2 columns.
// - RowsPerBuffer buffer setting of 3 does not divide evenly into total rows, thereby causing
//   partially filled buffers
// - Shuffle size captures the entire dataset size (actually sets a value that is larger than the
//   amount of rows in the dataset.
//
// Tree: shuffle over TFReader
//
//    ShuffleOp
//        |
//    TFReaderOp
//
TEST_F(MindDataTestShuffleOp, TestShuffleBasic3) {
  Status rc;
  MS_LOG(INFO) << "UT test TestShuffleBasic3.";

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testDataset1/testDataset1.data";
  std::shared_ptr<TFReaderOp> my_tfreader_op;
  rc = TFReaderOp::Builder()
      .SetDatasetFilesList({dataset_path})
      .SetRowsPerBuffer(3)
      .SetWorkerConnectorSize(16)
      .SetNumWorkers(2)
      .Build(&my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  my_tree->AssociateNode(my_tfreader_op);
  std::shared_ptr<ShuffleOp> my_shuffle_op;
  rc = ShuffleOp::Builder().SetShuffleSize(100).SetRowsPerBuffer(3).Build(&my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());

  // Set children/root layout.
  rc = my_shuffle_op->AddChild(my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "Launching tree and begin iteration.";
  rc = my_tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  EXPECT_TRUE(rc.IsOk());
  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count << ".";

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str()) << ".";
    }
    rc = di.FetchNextTensorRow(&tensor_list);
    EXPECT_TRUE(rc.IsOk());
    row_count++;
  }
  ASSERT_EQ(row_count, 10);
}


// Test info:
// - Dataset from testDataset1 has 10 rows, 2 columns.
// - RowsPerBuffer buffer setting of 3 does not divide evenly into total rows thereby causing
//   partially filled buffers
// - Shuffle size is not a multiple of rows per buffer.
// - shuffle seed is given, and subsequent epochs will change the seed each time.
// - Repeat count of 2
//
// Tree: Repeat over shuffle over TFReader
//
//    Repeat
//       |
//    shuffle
//       |
//    TFReaderOp
//
TEST_F(MindDataTestShuffleOp, TestRepeatShuffle) {
  Status rc;
  MS_LOG(INFO) << "UT test TestRepeatShuffle.";

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testDataset1/testDataset1.data";
  std::shared_ptr<TFReaderOp> my_tfreader_op;
  rc = TFReaderOp::Builder()
      .SetDatasetFilesList({dataset_path})
      .SetRowsPerBuffer(3)
      .SetWorkerConnectorSize(16)
      .SetNumWorkers(2)
      .Build(&my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  std::shared_ptr<ShuffleOp> my_shuffle_op;
  rc = ShuffleOp::Builder()
      .SetShuffleSize(4)
      .SetShuffleSeed(100)
      .SetRowsPerBuffer(3)
      .SetReshuffleEachEpoch(true)
      .Build(&my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  uint32_t numRepeats = 2;
  std::shared_ptr<RepeatOp> my_repeat_op;
  rc = RepeatOp::Builder(numRepeats).Build(&my_repeat_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_repeat_op);
  EXPECT_TRUE(rc.IsOk());

  // Set children/root layout.
  my_shuffle_op->set_total_repeats(numRepeats);
  my_shuffle_op->set_num_repeats_per_epoch(numRepeats);
  rc = my_repeat_op->AddChild(my_shuffle_op);
  EXPECT_TRUE(rc.IsOk());
  my_tfreader_op->set_total_repeats(numRepeats);
  my_tfreader_op->set_num_repeats_per_epoch(numRepeats);
  rc = my_shuffle_op->AddChild(my_tfreader_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_repeat_op);
  EXPECT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "Launching tree and begin iteration.";
  rc = my_tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  EXPECT_TRUE(rc.IsOk());
  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count << ".";

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << *tensor_list[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str()) << ".";
    }
    rc = di.FetchNextTensorRow(&tensor_list);
    EXPECT_TRUE(rc.IsOk());
    row_count++;
  }
  ASSERT_EQ(row_count, 20);
}
