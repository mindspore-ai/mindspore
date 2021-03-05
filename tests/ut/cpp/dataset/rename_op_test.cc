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
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/engine/datasetops/rename_op.h"
#include "common/common.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "gtest/gtest.h"
#include "minddata/dataset/core/global_context.h"
#include "utils/log_adapter.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRenameOp : public UT::DatasetOpTesting {
 };

TEST_F(MindDataTestRenameOp, TestRenameOpDefault) {
// Tree:
//
//
//       OpId(2) RenameOp
//            |
//     OpId(0) TFReaderOp
// Start with an empty execution tree
  Status rc;
  MS_LOG(INFO) << "UT test TestRenameBasic.";
  auto my_tree = std::make_shared<ExecutionTree>();
  // Creating TFReaderOp

  std::string dataset_path = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
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

  // Creating DatasetOp
  std::vector<std::string> in_colnames = {"label"};
  std::vector<std::string> out_colnames = {"label1"};

  std::shared_ptr<RenameOp> rename_op;
  rc = RenameOp::Builder()
      .SetInColNames(in_colnames)
      .SetOutColNames(out_colnames)
      .Build(&rename_op);
  EXPECT_TRUE(rc.IsOk());

  rc = my_tree->AssociateNode(rename_op);
  EXPECT_TRUE(rc.IsOk());
  rc = rename_op->AddChild(std::move(my_tfreader_op));
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(rename_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->Prepare();
  EXPECT_TRUE(rc.IsOk());

  // Launch the tree execution to kick off threads and start running the pipeline
  MS_LOG(INFO) << "Launching my tree.";
  rc = my_tree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Simulate a parse of data from our pipeline.
  std::shared_ptr<DatasetOp> root_node = my_tree->root();

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
  ASSERT_EQ(row_count, 3); // Should be 3 rows fetched
}
