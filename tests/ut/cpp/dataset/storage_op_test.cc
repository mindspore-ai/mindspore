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
#include "dataset/core/client.h"
#include "common/common.h"
#include "common/utils.h"
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

class MindDataTestStorageOp : public UT::DatasetOpTesting {

};

TEST_F(MindDataTestStorageOp, TestStorageBasic1) {

  // single storage op and nothing else
  //
  //    StorageOp

  MS_LOG(INFO) << "UT test TestStorageBasic1.";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 2 divides evenly into total rows.
  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testDataset1";
  std::shared_ptr<StorageOp> my_storage_op;
  StorageOp::Builder builder;
  builder.SetDatasetFilesDir(dataset_path)
      .SetRowsPerBuffer(2)
      .SetWorkerConnectorSize(16)
      .SetNumWorkers(1);
  rc = builder.Build(&my_storage_op);
  ASSERT_TRUE(rc.IsOk());
  my_tree->AssociateNode(my_storage_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_storage_op);

  MS_LOG(INFO) << "Launching tree and begin iteration.";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

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
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
  ASSERT_EQ(row_count, 10); // Should be 10 rows fetched

  // debugging temp.  what happens if we keep fetching..
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestStorageOp, TestStorageBasic2) {

  // single storage op and nothing else
  //
  //    StorageOp

  MS_LOG(INFO) << "UT test TestStorageBasic1.";

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  // RowsPerBuffer buffer setting of 3 yields 4 buffers with the last buffer having single row
  // only.  2 workers.
  // Test a column selection instead of all columns as well.
  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testDataset1";
  std::vector<std::string> column_list;
  std::string label_colname("label");
  column_list.push_back(label_colname);
  std::shared_ptr<StorageOp> my_storage_op;
  StorageOp::Builder builder;
  builder.SetDatasetFilesDir(dataset_path)
    .SetRowsPerBuffer(3)
    .SetWorkerConnectorSize(16)
    .SetNumWorkers(2)
    .SetColumnsToLoad(column_list);
  rc = builder.Build(&my_storage_op);
  ASSERT_TRUE(rc.IsOk());
  my_tree->AssociateNode(my_storage_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_storage_op);

  MS_LOG(INFO) << "Launching tree and begin iteration.";
  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

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
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
  ASSERT_EQ(row_count, 10); // Should be 10 rows fetched
}
