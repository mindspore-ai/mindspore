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
#include <iostream>
#include <memory>
#include <string>
#include "dataset/core/client.h"
#include "common/common.h"
#include "common/utils.h"
#include "gtest/gtest.h"
#include "dataset/core/global_context.h"
#include "dataset/util/de_error.h"
#include "utils/log_adapter.h"
#include "securec.h"
#include "dataset/util/status.h"

namespace common = mindspore::common;
namespace de = mindspore::dataset;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestBatchOp : public UT::DatasetOpTesting {
 protected:

};


std::shared_ptr<de::BatchOp> Batch(int32_t batch_size = 1, bool drop = false, int rows_per_buf = 2) {
  Status rc;
  std::shared_ptr<de::BatchOp> op;
  rc = de::BatchOp::Builder(batch_size).SetDrop(drop).Build(&op);
  EXPECT_TRUE(rc.IsOk());
  return op;
}

std::shared_ptr<de::RepeatOp> Repeat(int repeat_cnt = 1) {
  de::RepeatOp::Builder builder(repeat_cnt);
  std::shared_ptr<de::RepeatOp> op;
  Status rc = builder.Build(&op);
  EXPECT_TRUE(rc.IsOk());
  return op;
}

std::shared_ptr<de::StorageOp> Storage(std::string schema, int rows_per_buf = 2, int num_works = 8) {
  std::shared_ptr<de::StorageOp> so;
  de::StorageOp::Builder builder;
  builder.SetDatasetFilesDir(schema).SetRowsPerBuffer(rows_per_buf).SetNumWorkers(num_works);
  Status rc = builder.Build(&so);
  return so;
}

std::shared_ptr<de::ExecutionTree> Build(std::vector<std::shared_ptr<de::DatasetOp>> ops) {
  std::shared_ptr<de::ExecutionTree> tree = std::make_shared<de::ExecutionTree>();
  for (int i = 0; i < ops.size(); i++) {
    tree->AssociateNode(ops[i]);
    if (i > 0) {
      ops[i]->AddChild(ops[i - 1]);
    }
    if (i == ops.size() - 1) {
      tree->AssignRoot(ops[i]);
    }
  }
  return tree;
}

TEST_F(MindDataTestBatchOp, TestSimpleBatch) {
  std::string schema_file = datasets_root_path_ + "/testBatchDataset";
  bool success = false;
  auto tree = Build({Storage(schema_file), Batch(12)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
  } else {
    int64_t payload[] = {-9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807};
    de::DatasetIterator di(tree);
    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    std::shared_ptr<de::Tensor> t;
    rc = de::Tensor::CreateTensor(&t,
                                  TensorImpl::kFlexible, de::TensorShape({12, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) payload);
    EXPECT_TRUE(rc.IsOk());
    // verify the actual data in Tensor is correct
    EXPECT_EQ(*t == *tensor_map["col_sint64"], true);
    // change what's in Tensor and verify this time the data is incorrect1;
    EXPECT_EQ(*t == *tensor_map["col_sint16"], false);
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    if (tensor_map.size() == 0) {
      success = true;
    }
  }
  EXPECT_EQ(success, true);
}


TEST_F(MindDataTestBatchOp, TestRepeatBatchDropTrue) {
  std::string schema_file = datasets_root_path_ + "/testBatchDataset";
  bool success = false;
  auto tree = Build({Storage(schema_file), Repeat(2), Batch(7, true, 99)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
  } else {
    int64_t payload[] = {-9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807,
                         -9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807};
    de::DatasetIterator di(tree);
    std::shared_ptr<de::Tensor> t1, t2, t3;
    rc = de::Tensor::CreateTensor(&t1,
                                  TensorImpl::kFlexible, de::TensorShape({7, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) payload);
    EXPECT_TRUE(rc.IsOk());
    rc = de::Tensor::CreateTensor(&t2,
                                  TensorImpl::kFlexible, de::TensorShape({7, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) (payload + 7));
    EXPECT_TRUE(rc.IsOk());
    rc = de::Tensor::CreateTensor(&t3,
                                  TensorImpl::kFlexible, de::TensorShape({7, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) (payload + 2));
    EXPECT_TRUE(rc.IsOk());

    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t1 == *(tensor_map["col_sint64"]), true);  // first call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t2 == *(tensor_map["col_sint64"]), true);  // second call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t3 == *(tensor_map["col_sint64"]), true);  // third call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    if (tensor_map.size() == 0) {
      success = true;
    }
  }
  EXPECT_EQ(success, true);
}


TEST_F(MindDataTestBatchOp, TestRepeatBatchDropFalse) {
  std::string schema_file = datasets_root_path_ + "/testBatchDataset";
  bool success = false;
  auto tree = Build({Storage(schema_file), Repeat(2), Batch(7, false, 99)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
  } else {
    int64_t payload[] = {-9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807,
                         -9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807};
    de::DatasetIterator di(tree);
    std::shared_ptr<de::Tensor> t1, t2, t3, t4;
    rc = de::Tensor::CreateTensor(&t1,
                                  TensorImpl::kFlexible, de::TensorShape({7, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) payload);
    EXPECT_TRUE(rc.IsOk());
    rc = de::Tensor::CreateTensor(&t2,
                                  TensorImpl::kFlexible, de::TensorShape({7, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) (payload + 7));
    EXPECT_TRUE(rc.IsOk());
    rc = de::Tensor::CreateTensor(&t3,
                                  TensorImpl::kFlexible, de::TensorShape({7, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) (payload + 2));
    EXPECT_TRUE(rc.IsOk());
    rc = de::Tensor::CreateTensor(&t4,
                                  TensorImpl::kFlexible, de::TensorShape({3, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) (payload + 9));
    EXPECT_TRUE(rc.IsOk());

    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t1 == *(tensor_map["col_sint64"]), true);  // first call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t2 == *(tensor_map["col_sint64"]), true);  // second call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t3 == *(tensor_map["col_sint64"]), true);  // third call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t4 == *(tensor_map["col_sint64"]), true);  // last call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    if (tensor_map.size() == 0) {
      success = true;
    }
  }
  EXPECT_EQ(success, true);
}


TEST_F(MindDataTestBatchOp, TestBatchDropFalseRepeat) {
  std::string schema_file = datasets_root_path_ + "/testBatchDataset";
  bool success = false;
  auto tree = Build({Storage(schema_file), Batch(7, false, 99), Repeat(2)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
  } else {
    int64_t payload[] = {-9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807,
                         -9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807};
    de::DatasetIterator di(tree);
    std::shared_ptr<de::Tensor> t1, t2;
    rc = de::Tensor::CreateTensor(&t1,
                                  TensorImpl::kFlexible, de::TensorShape({7, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) payload);
    EXPECT_TRUE(rc.IsOk());
    rc = de::Tensor::CreateTensor(&t2,
                                  TensorImpl::kFlexible, de::TensorShape({5, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) (payload + 7));
    EXPECT_TRUE(rc.IsOk());

    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t1 == *(tensor_map["col_sint64"]), true);  // first call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t2 == *(tensor_map["col_sint64"]), true);  // second call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t1 == *(tensor_map["col_sint64"]), true);  // third call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t2 == *(tensor_map["col_sint64"]), true);  // last call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    if (tensor_map.size() == 0) {
      success = true;
    }
  }
  EXPECT_EQ(success, true);
}


TEST_F(MindDataTestBatchOp, TestBatchDropTrueRepeat) {
  std::string schema_file = datasets_root_path_ + "/testBatchDataset";
  bool success = false;
  auto tree = Build({Storage(schema_file), Batch(5, true, 99), Repeat(2)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
  } else {
    int64_t payload[] = {-9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807,
                         -9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807};
    de::DatasetIterator di(tree);
    std::shared_ptr<de::Tensor> t1, t2;
    rc = de::Tensor::CreateTensor(&t1,
                                  TensorImpl::kFlexible, de::TensorShape({5, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) payload);
    EXPECT_TRUE(rc.IsOk());
    rc = de::Tensor::CreateTensor(&t2,
                                  TensorImpl::kFlexible, de::TensorShape({5, 1}),
                                  de::DataType(DataType::DE_INT64),
                                  (unsigned char *) (payload + 5));
    EXPECT_TRUE(rc.IsOk());

    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t1 == *(tensor_map["col_sint64"]), true);  // first call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t2 == *(tensor_map["col_sint64"]), true);  // second call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t1 == *(tensor_map["col_sint64"]), true);  // third call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(*t2 == *(tensor_map["col_sint64"]), true);  // last call to getNext()

    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    if (tensor_map.size() == 0) {
      success = true;
    }
  }
  EXPECT_EQ(success, true);
}
