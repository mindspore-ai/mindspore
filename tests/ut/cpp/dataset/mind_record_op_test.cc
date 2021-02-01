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
#include <iostream>
#include <memory>
#include <vector>
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "utils/ms_utils.h"
#include "gtest/gtest.h"
#include "minddata/mindrecord/include/shard_category.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "utils/log_adapter.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestMindRecordOp : public UT::DatasetOpTesting {
};

TEST_F(MindDataTestMindRecordOp, TestMindRecordBasic) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordBasic";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::shared_ptr<MindRecordOp> my_mindrecord_op;
  MindRecordOp::Builder builder;
  builder.SetDatasetFile({mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"})
      .SetLoadDataset(true)
      .SetRowsPerBuffer(3)
      .SetNumMindRecordWorkers(4)
      .SetColumnsToLoad(column_list);
  rc = builder.Build(&my_mindrecord_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << (*my_mindrecord_op);

  my_tree->AssociateNode(my_mindrecord_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_mindrecord_op);

  MS_LOG(INFO) << "Launching tree and begin iteration";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << (*tensor_list[i]) << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str());
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
}

TEST_F(MindDataTestMindRecordOp, TestMindRecordSample) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordSample";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::vector<std::shared_ptr<mindspore::mindrecord::ShardOperator>> operators;
  operators.push_back(std::make_shared<mindspore::mindrecord::ShardSample>(4));

  std::shared_ptr<MindRecordOp> my_mindrecord_op;
  MindRecordOp::Builder builder;
  builder.SetDatasetFile({mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"})
      .SetLoadDataset(true)
      .SetRowsPerBuffer(3)
      .SetNumMindRecordWorkers(4)
      .SetColumnsToLoad(column_list)
      .SetOperators(operators);
  rc = builder.Build(&my_mindrecord_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << (*my_mindrecord_op);

  my_tree->AssociateNode(my_mindrecord_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_mindrecord_op);

  MS_LOG(INFO) << "Launching tree and begin iteration";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << (*tensor_list[i]) << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str());
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
}

TEST_F(MindDataTestMindRecordOp, TestMindRecordShuffle) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordShuffle";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::vector<std::shared_ptr<mindspore::mindrecord::ShardOperator>> operators;
  operators.push_back(std::make_shared<mindspore::mindrecord::ShardShuffle>(1));

  std::shared_ptr<MindRecordOp> my_mindrecord_op;
  MindRecordOp::Builder builder;
  builder.SetDatasetFile({mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"})
      .SetLoadDataset(true)
      .SetRowsPerBuffer(3)
      .SetNumMindRecordWorkers(4)
      .SetColumnsToLoad(column_list)
      .SetOperators(operators);
  rc = builder.Build(&my_mindrecord_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << (*my_mindrecord_op);

  my_tree->AssociateNode(my_mindrecord_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_mindrecord_op);

  MS_LOG(INFO) << "Launching tree and begin iteration";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << (*tensor_list[i]) << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str());
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
}

TEST_F(MindDataTestMindRecordOp, TestMindRecordCategory) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordCategory";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::vector<std::shared_ptr<mindspore::mindrecord::ShardOperator>> operators;
  std::vector<std::pair<std::string, std::string>> categories;
  categories.push_back(std::make_pair("label", "490"));
  categories.push_back(std::make_pair("label", "171"));
  operators.push_back(std::make_shared<mindspore::mindrecord::ShardCategory>(categories));

  std::shared_ptr<MindRecordOp> my_mindrecord_op;
  MindRecordOp::Builder builder;
  builder.SetDatasetFile({mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"})
      .SetLoadDataset(true)
      .SetRowsPerBuffer(3)
      .SetNumMindRecordWorkers(4)
      .SetColumnsToLoad(column_list)
      .SetOperators(operators);
  rc = builder.Build(&my_mindrecord_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << (*my_mindrecord_op);

  my_tree->AssociateNode(my_mindrecord_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_mindrecord_op);

  MS_LOG(INFO) << "Launching tree and begin iteration";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << (*tensor_list[i]) << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str());
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
}

TEST_F(MindDataTestMindRecordOp, TestMindRecordRepeat) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordRepeat";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::shared_ptr<MindRecordOp> my_mindrecord_op;
  MindRecordOp::Builder builder;
  builder.SetDatasetFile({mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"})
      .SetLoadDataset(true)
      .SetRowsPerBuffer(3)
      .SetNumMindRecordWorkers(4)
      .SetColumnsToLoad(column_list);
  rc = builder.Build(&my_mindrecord_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << (*my_mindrecord_op);

  rc = my_tree->AssociateNode(my_mindrecord_op);
  EXPECT_TRUE(rc.IsOk());

  uint32_t num_repeats = 2;
  std::shared_ptr<RepeatOp> my_repeat_op;
  rc = RepeatOp::Builder(num_repeats)
      .Build(&my_repeat_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_repeat_op);
  EXPECT_TRUE(rc.IsOk());

  my_mindrecord_op->set_total_repeats(num_repeats);
  my_mindrecord_op->set_num_repeats_per_epoch(num_repeats);
  rc = my_repeat_op->AddChild(my_mindrecord_op);
  EXPECT_TRUE(rc.IsOk());


  // Set children/root layout.
  rc = my_tree->AssignRoot(my_repeat_op);
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << (*tensor_list[i]) << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str());
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
}


TEST_F(MindDataTestMindRecordOp, TestMindRecordBlockReaderRepeat) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordBlockReaderRepeat";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::shared_ptr<MindRecordOp> my_mindrecord_op;
  MindRecordOp::Builder builder;
  builder.SetDatasetFile({mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"})
      .SetLoadDataset(true)
      .SetRowsPerBuffer(3)
      .SetNumMindRecordWorkers(4)
      .SetColumnsToLoad(column_list);
  rc = builder.Build(&my_mindrecord_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << (*my_mindrecord_op);

  rc = my_tree->AssociateNode(my_mindrecord_op);
  EXPECT_TRUE(rc.IsOk());

  uint32_t num_repeats = 2;
  std::shared_ptr<RepeatOp> my_repeat_op;
  rc = RepeatOp::Builder(num_repeats)
      .Build(&my_repeat_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_repeat_op);
  EXPECT_TRUE(rc.IsOk());

  my_mindrecord_op->set_total_repeats(num_repeats);
  my_mindrecord_op->set_num_repeats_per_epoch(num_repeats);
  rc = my_repeat_op->AddChild(my_mindrecord_op);
  EXPECT_TRUE(rc.IsOk());

  // Set children/root layout.
  rc = my_tree->AssignRoot(my_repeat_op);
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << (*tensor_list[i]) << std::endl;
      MS_LOG(INFO) << "Tensor print: " << common::SafeCStr(ss.str());
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
}

TEST_F(MindDataTestMindRecordOp, TestMindRecordInvalidColumnList) {
  // single MindRecord op and nothing else
  //
  //    MindRecordOp

  MS_LOG(INFO) << "UT test TestMindRecordInvalidColumnList";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.

  std::vector<std::string> column_list;
  std::string label_col_name("file_name_2");
  column_list.push_back(label_col_name);
  label_col_name = "label";
  column_list.push_back(label_col_name);

  std::shared_ptr<MindRecordOp> my_mindrecord_op;
  MindRecordOp::Builder builder;
  builder.SetDatasetFile({mindrecord_root_path_ + "/testMindDataSet/testImageNetData/imagenet.mindrecord0"})
      .SetLoadDataset(true)
      .SetRowsPerBuffer(3)
      .SetNumMindRecordWorkers(4)
      .SetColumnsToLoad(column_list);
  rc = builder.Build(&my_mindrecord_op);
  ASSERT_TRUE(rc.IsError());
  ASSERT_TRUE(rc.ToString().find_first_of("illegal column list") != std::string::npos);
}
