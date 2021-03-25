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
#include <iostream>
#include <memory>
#include <vector>

#include "common/common.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/client.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestConcatOp : public UT::DatasetOpTesting {};


TEST_F(MindDataTestConcatOp, TestConcatProject) {
/* Tree:
 *
 *            OpId(2) ConcatOp
 *            /               \
 *     OpId(0) TFReaderOp    OpId(1) TFReaderOp
 *
 * Start with an empty execution tree
*/
  MS_LOG(INFO) << "UT test TestConcatProject.";
  auto my_tree = std::make_shared<ExecutionTree>();

  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";

  // TFReaderOp1
  std::shared_ptr<TFReaderOp> my_tfreader_op1;
  TFReaderOp::Builder builder1;
  builder1.SetDatasetFilesList({dataset_path}).SetRowsPerBuffer(16).SetWorkerConnectorSize(16);
  std::unique_ptr<DataSchema> schema1 = std::make_unique<DataSchema>();
  schema1->LoadSchemaFile(datasets_root_path_ + "/testTFTestAllTypes/datasetSchema1Row.json", {});
  builder1.SetDataSchema(std::move(schema1));
  Status rc = builder1.Build(&my_tfreader_op1);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_tfreader_op1);
  ASSERT_TRUE(rc.IsOk());

  // TFReaderOp2
  std::shared_ptr<TFReaderOp> my_tfreader_op2;
  TFReaderOp::Builder builder2;
  builder2.SetDatasetFilesList({dataset_path}).SetRowsPerBuffer(16).SetWorkerConnectorSize(16);
  std::unique_ptr<DataSchema> schema2 = std::make_unique<DataSchema>();
  schema2->LoadSchemaFile(datasets_root_path_ + "/testTFTestAllTypes/datasetSchema1Row.json", {});
  builder2.SetDataSchema(std::move(schema2));
  rc = builder2.Build(&my_tfreader_op2);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_tfreader_op2);
  ASSERT_TRUE(rc.IsOk());

  // Creating ConcatOp
  std::shared_ptr<ConcatOp> concat_op;
  rc = ConcatOp::Builder().Build(&concat_op);
  EXPECT_TRUE(rc.IsOk());

  rc = my_tree->AssociateNode(concat_op);
  EXPECT_TRUE(rc.IsOk());
  rc = concat_op->AddChild(std::move(my_tfreader_op1));
  EXPECT_TRUE(rc.IsOk());
  rc = concat_op->AddChild(std::move(my_tfreader_op2));
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(concat_op);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree->Prepare();
  EXPECT_TRUE(rc.IsOk());

  // Launch the tree execution to kick off threads and start running the pipeline
  MS_LOG(INFO) << "Launching my tree.";
  rc = my_tree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Simulate a parse of data from our pipeline.
  std::shared_ptr<DatasetOp> rootNode = my_tree->root();

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
  ASSERT_EQ(row_count, 2); // Should be 2 rows fetched
}