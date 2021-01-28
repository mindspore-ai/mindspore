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
#include "gtest/gtest.h"
#include <memory>
#include <vector>
#include <iostream>
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/data_schema.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestRandomDataOp : public UT::DatasetOpTesting {

};

// Test info:
// - Simple test with a user-provided schema generated purely from DataSchema C API
// - has an interaction loop
//
// Tree:  single node tree with RandomDataOp
//
//    RandomDataOp
//
TEST_F(MindDataTestRandomDataOp, RandomDataOpBasic1) {
  Status rc;
  int32_t rank = 0; // not used
  MS_LOG(INFO) << "UT test RandomDataOpBasic1";

  // Start with an empty execution tree
  auto myTree = std::make_shared<ExecutionTree>();

  // Create a schema using the C api's
  std::unique_ptr<DataSchema> testSchema = std::make_unique<DataSchema>();

  // RandomDataOp can randomly fill in unknown dimension lengths of a shape.
  // Most other ops cannot do that as they are limited by the physical data itself. We're
  // more flexible with random data since it is just making stuff up on the fly.
  TensorShape c1Shape({TensorShape::kDimUnknown, TensorShape::kDimUnknown, 3});
  ColDescriptor c1("image",
                   DataType(DataType::DE_INT8),
                   TensorImpl::kFlexible,
                   rank,  // not used
                   &c1Shape);

  // Column 2 will just be a scalar label number
  TensorShape c2Shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor c2("label",
                   DataType(DataType::DE_UINT32),
                   TensorImpl::kFlexible,
                   rank,
                   &c2Shape);

  testSchema->AddColumn(c1);
  testSchema->AddColumn(c2);

  std::shared_ptr<RandomDataOp> myRandomDataOp;
  RandomDataOp::Builder builder;

  rc = builder.SetRowsPerBuffer(2)
    .SetNumWorkers(1)
    .SetDataSchema(std::move(testSchema))
    .SetTotalRows(25)
    .Build(&myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssignRoot(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  std::ostringstream ss;
  ss << *myRandomDataOp;
  MS_LOG(INFO) << "RandomDataOp print: %s" << ss.str();

  MS_LOG(INFO) << "Launching tree and begin iteration";
  rc = myTree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  EXPECT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    // Don't display these rows...too big to show
    MS_LOG(INFO) << "Row fetched #: " << rowCount;

    rc = dI.FetchNextTensorRow(&tensorList);
    EXPECT_TRUE(rc.IsOk());
    rowCount++;
  }
  ASSERT_EQ(rowCount, 25);
}

// Test info:
// - Simple test with a randomly generated schema
// - no iteration loop on this one, just create the op
//
// Tree:  single node tree with RandomDataOp
//
//    RandomDataOp
//
TEST_F(MindDataTestRandomDataOp, RandomDataOpBasic2) {
  Status rc;
  MS_LOG(INFO) << "UT test RandomDataOpBasic2";

  // Start with an empty execution tree
  auto myTree = std::make_shared<ExecutionTree>();

  std::shared_ptr<RandomDataOp> myRandomDataOp;
  RandomDataOp::Builder builder;

  rc = builder.SetRowsPerBuffer(2)
    .SetNumWorkers(1)
    .Build(&myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssignRoot(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  std::ostringstream ss;
  ss << *myRandomDataOp;
  MS_LOG(INFO) << "RandomDataOp print: " << ss.str();
}

// Test info:
// - json file test with iteration
//
// Tree:  single node tree with RandomDataOp
//
//    RandomDataOp
//
TEST_F(MindDataTestRandomDataOp, RandomDataOpBasic3) {
  Status rc;
  MS_LOG(INFO) << "UT test RandomDataOpBasic3";

  // Start with an empty execution tree
  auto myTree = std::make_shared<ExecutionTree>();

  std::unique_ptr<DataSchema> testSchema = std::make_unique<DataSchema>();
  rc = testSchema->LoadSchemaFile(datasets_root_path_ + "/testRandomData/datasetSchema.json", {});
  EXPECT_TRUE(rc.IsOk());

  std::shared_ptr<RandomDataOp> myRandomDataOp;
  RandomDataOp::Builder builder;

  rc = builder.SetRowsPerBuffer(2)
    .SetNumWorkers(1)
    .SetDataSchema(std::move(testSchema))
    .SetTotalRows(10)
    .Build(&myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssignRoot(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  std::ostringstream ss;
  ss << *myRandomDataOp;
  MS_LOG(INFO) << "RandomDataOp print: " << ss.str();

  MS_LOG(INFO) << "Launching tree and begin iteration";
  rc = myTree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  EXPECT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    // Don't display these rows...too big to show
    MS_LOG(INFO) << "Row fetched #: " << rowCount;

    rc = dI.FetchNextTensorRow(&tensorList);
    EXPECT_TRUE(rc.IsOk());
    rowCount++;
  }
  ASSERT_EQ(rowCount, 10);
}

// Test info:
// - json schema input it's a fairly simple one
// - has an interaction loop
//
// Tree:  RepeatOp over RandomDataOp
//
//     RepeatOp
//        |
//    RandomDataOp
//
TEST_F(MindDataTestRandomDataOp, RandomDataOpBasic4) {
  Status rc;
  MS_LOG(INFO) << "UT test RandomDataOpBasic4";

  // Start with an empty execution tree
  auto myTree = std::make_shared<ExecutionTree>();

  std::unique_ptr<DataSchema> testSchema = std::make_unique<DataSchema>();
  rc = testSchema->LoadSchemaFile(datasets_root_path_ + "/testRandomData/datasetSchema2.json", {});
  EXPECT_TRUE(rc.IsOk());

  std::shared_ptr<RandomDataOp> myRandomDataOp;
  RandomDataOp::Builder builder;

  rc = builder.SetRowsPerBuffer(2)
    .SetNumWorkers(1)
    .SetDataSchema(std::move(testSchema))
    .SetTotalRows(10)
    .Build(&myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  uint32_t numRepeats = 2;
  std::shared_ptr<RepeatOp> myRepeatOp;
  rc = RepeatOp::Builder(numRepeats)
    .Build(&myRepeatOp);
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myRepeatOp);
  EXPECT_TRUE(rc.IsOk());

  myRandomDataOp->set_total_repeats(numRepeats);
  myRandomDataOp->set_num_repeats_per_epoch(numRepeats);
  rc = myRepeatOp->AddChild(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssignRoot(myRepeatOp);
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration";
  rc = myTree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  EXPECT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << rowCount;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensorList.size(); i++) {
      std::ostringstream ss;
      ss << *tensorList[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: %s" << ss.str();
    }

    rc = dI.FetchNextTensorRow(&tensorList);
    EXPECT_TRUE(rc.IsOk());
    rowCount++;
  }
  ASSERT_EQ(rowCount, 20);
}

// Test info:
// - json schema input it's a fairly simple one
// - has an interaction loop
// - same as MindDataTestRandomDataOpBasic4 except that this one will have parallel workers
//
// Tree:  RepeatOp over RandomDataOp
//
//     RepeatOp
//        |
//    RandomDataOp
//
TEST_F(MindDataTestRandomDataOp, RandomDataOpBasic5) {
  Status rc;
  MS_LOG(INFO) << "UT test RandomDataOpBasic5";

  // Start with an empty execution tree
  auto myTree = std::make_shared<ExecutionTree>();

  std::unique_ptr<DataSchema> testSchema = std::make_unique<DataSchema>();
  rc = testSchema->LoadSchemaFile(datasets_root_path_ + "/testRandomData/datasetSchema2.json", {});
  EXPECT_TRUE(rc.IsOk());

  std::shared_ptr<RandomDataOp> myRandomDataOp;
  RandomDataOp::Builder builder;

  rc = builder.SetRowsPerBuffer(2)
    .SetNumWorkers(4)
    .SetDataSchema(std::move(testSchema))
    .SetTotalRows(10)
    .Build(&myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  uint32_t numRepeats = 3;
  std::shared_ptr<RepeatOp> myRepeatOp;
  rc = RepeatOp::Builder(numRepeats)
    .Build(&myRepeatOp);
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myRepeatOp);
  EXPECT_TRUE(rc.IsOk());

  myRandomDataOp->set_total_repeats(numRepeats);
  myRandomDataOp->set_num_repeats_per_epoch(numRepeats);
  rc = myRepeatOp->AddChild(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssignRoot(myRepeatOp);
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration";
  rc = myTree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  EXPECT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << rowCount;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensorList.size(); i++) {
      std::ostringstream ss;
      ss << *tensorList[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: ", ss.str();
    }

    rc = dI.FetchNextTensorRow(&tensorList);
    EXPECT_TRUE(rc.IsOk());
    rowCount++;
  }
  ASSERT_EQ(rowCount, 30);
}

// Test info:
// - repeat shuffle random
//
// Tree:  RepeatOp over RandomDataOp
//
//     RepeatOp
//        |
//     ShuffleOp
//        |
//    RandomDataOp
//
TEST_F(MindDataTestRandomDataOp, RandomDataOpTree1) {
  Status rc;
  MS_LOG(INFO) << "UT test RandomDataOpTree1";

  // Start with an empty execution tree
  auto myTree = std::make_shared<ExecutionTree>();

  std::unique_ptr<DataSchema> testSchema = std::make_unique<DataSchema>();
  rc = testSchema->LoadSchemaFile(datasets_root_path_ + "/testRandomData/datasetSchema2.json", {});
  EXPECT_TRUE(rc.IsOk());

  std::shared_ptr<RandomDataOp> myRandomDataOp;
  RandomDataOp::Builder builder;

  rc = builder.SetRowsPerBuffer(2)
    .SetNumWorkers(4)
    .SetDataSchema(std::move(testSchema))
    .SetTotalRows(10)
    .Build(&myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  std::shared_ptr<ShuffleOp> myShuffleOp;
  rc = ShuffleOp::Builder()
      .SetRowsPerBuffer(2)
      .SetShuffleSize(4)
      .Build(&myShuffleOp);
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myShuffleOp);
  EXPECT_TRUE(rc.IsOk());

  uint32_t numRepeats = 3;
  std::shared_ptr<RepeatOp> myRepeatOp;
  rc = RepeatOp::Builder(numRepeats)
    .Build(&myRepeatOp);
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myRepeatOp);
  EXPECT_TRUE(rc.IsOk());

  myShuffleOp->set_total_repeats(numRepeats);
  myShuffleOp->set_num_repeats_per_epoch(numRepeats);
  rc = myRepeatOp->AddChild(myShuffleOp);
  EXPECT_TRUE(rc.IsOk());

  myRandomDataOp->set_total_repeats(numRepeats);
  myRandomDataOp->set_num_repeats_per_epoch(numRepeats);
  rc = myShuffleOp->AddChild(myRandomDataOp);
  EXPECT_TRUE(rc.IsOk());

  rc = myTree->AssignRoot(myRepeatOp);
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration";
  rc = myTree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = myTree->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  EXPECT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << rowCount;

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensorList.size(); i++) {
      std::ostringstream ss;
      ss << *tensorList[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: " << ss.str();
    }

    rc = dI.FetchNextTensorRow(&tensorList);
    EXPECT_TRUE(rc.IsOk());
    rowCount++;
  }
  ASSERT_EQ(rowCount, 30);
}
