/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include "common/common.h"
#include "include/dataset/ms_tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/general/transform/transforms_ir.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

namespace mindspore::dataset::test {
class NoOp : public TensorOp {
 public:
  NoOp(){};

  ~NoOp(){};

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override {
    *output = std::move(input);
    return Status::OK();
  };

  void Print(std::ostream &out) const override { out << "NoOp"; };

  std::string Name() const override { return kNoOp; }
};

class ThreeToOneOp : public TensorOp {
 public:
  ThreeToOneOp(){};

  ~ThreeToOneOp(){};

  uint32_t NumInput() override {
    uint32_t numInput = 3;
    return numInput;
  }

  // Compute function that holds the actual implementation of the operation.
  Status Compute(const TensorRow &input, TensorRow *output) override {
    output->push_back(input[0]);
    return Status::OK();
  };

  void Print(std::ostream &out) const override { out << "ThreeToOneOp"; };

  std::string Name() const override { return "ThreeToOneOp"; }
};

class OneToThreeOp : public TensorOp {
 public:
  OneToThreeOp(){};

  ~OneToThreeOp(){};

  uint32_t NumOutput() override {
    uint32_t numOutput = 3;
    return numOutput;
  }

  // Compute function that holds the actual implementation of the operation.
  // Simply pushing the same shared pointer of the first element of input vector three times.
  Status Compute(const TensorRow &input, TensorRow *output) override {
    output->push_back(input[0]);
    output->push_back(input[0]);
    output->push_back(input[0]);
    return Status::OK();
  };

  void Print(std::ostream &out) const override { out << "OneToThreeOp"; };

  std::string Name() const override { return "OneToThreeOp"; };
};

class NoTransform final : public TensorTransform {
 public:
  explicit NoTransform() {}
  ~NoTransform() = default;

 protected:
  std::shared_ptr<TensorOperation> Parse() override {
    return std::make_shared<transforms::PreBuiltOperation>(std::make_shared<mindspore::dataset::test::NoOp>());
  }

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

class ThreeToOneTransform final : public TensorTransform {
 public:
  explicit ThreeToOneTransform() {}
  ~ThreeToOneTransform() = default;

 protected:
  std::shared_ptr<TensorOperation> Parse() override {
    return std::make_shared<transforms::PreBuiltOperation>(std::make_shared<mindspore::dataset::test::ThreeToOneOp>());
  }

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

class OneToThreeTransform final : public TensorTransform {
 public:
  explicit OneToThreeTransform() {}
  ~OneToThreeTransform() = default;

 protected:
  std::shared_ptr<TensorOperation> Parse() override {
    return std::make_shared<transforms::PreBuiltOperation>(std::make_shared<mindspore::dataset::test::OneToThreeOp>());
  }

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};
}  // namespace mindspore::dataset::test

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

MSTensorVec BucketBatchTestFunction(MSTensorVec input) {
  TensorRow output;
  std::shared_ptr<Tensor> out;
  (void)Tensor::CreateEmpty(TensorShape({1}), DataType(DataType::Type::DE_INT32), &out);
  constexpr int value = 2;
  (void)out->SetItemAt({0}, value);
  output.push_back(out);
  return RowToVec(output);
}

/// Feature: Batch and Repeat ops
/// Description: Test Batch and Repeat ops on MnistDataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestBatchAndRepeat) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBatchAndRepeat.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 2;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BucketBatchByLength op
/// Description: Test BucketBatchByLength op with default values
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthSuccess1) {
  // Calling with default values
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthSuccess1.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a BucketBatchByLength operation on ds
  ds = ds->BucketBatchByLength({"image"}, {1, 2, 3}, {4, 5, 6, 7});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  // 2 batches of size 5
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BucketBatchByLength op
/// Description: Test BucketBatchByLength op with non-default values
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthSuccess2) {
  // Calling with non-default values
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthSuccess2.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a BucketBatchByLength operation on ds
  std::map<std::string, std::pair<std::vector<int64_t>, mindspore::MSTensor>> pad_info = {};
  ds = ds->BucketBatchByLength({"image"}, {1, 2}, {1, 2, 3}, &BucketBatchTestFunction, pad_info, true, true);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  // With 2 boundaries, 3 buckets are created
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: BucketBatchByLength op
/// Description: Test BucketBatchByLength op with empty bucket_boundaries
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthFail1) {
  // Empty bucket_boundaries
  // Calling with function pointer
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthFail1.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a BucketBatchByLength operation on ds
  ds = ds->BucketBatchByLength({"image"}, {}, {1});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: BucketBatchByLength op
/// Description: Test BucketBatchByLength op with empty bucket_batch_sizes
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthFail2) {
  // Empty bucket_batch_sizes
  // Calling with function pointer
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthFail2.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a BucketBatchByLength operation on ds
  ds = ds->BucketBatchByLength({"image"}, {1}, {});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: BucketBatchByLength op
/// Description: Test BucketBatchByLength op with negative boundaries
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthFail3) {
  // Negative boundaries
  // Calling with function pointer
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthFail3.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a BucketBatchByLength operation on ds
  ds = ds->BucketBatchByLength({"image"}, {-1, 1}, {1, 2, 3});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: BucketBatchByLength op
/// Description: Test BucketBatchByLength op with boundaries not strictly increasing
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthFail4) {
  // Boundaries not strictly increasing
  // Calling with function pointer
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthFail4.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a BucketBatchByLength operation on ds
  ds = ds->BucketBatchByLength({"image"}, {2, 2}, {1, 2, 3});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: BucketBatchByLength op
/// Description: Test BucketBatchByLength op with incorrect size of bucket_batch_size
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthFail5) {
  // Incorrect size of bucket_batch_size
  // Calling with function pointer
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthFail5.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a BucketBatchByLength operation on ds
  ds = ds->BucketBatchByLength({"image"}, {1, 2}, {1, 2});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: BucketBatchByLength op
/// Description: Test BucketBatchByLength op with negative bucket_batch_size
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthFail6) {
  // Negative bucket_batch_size
  // Calling with function pointer
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthFail6.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);
  // Create a BucketBatchByLength operation on ds
  ds = ds->BucketBatchByLength({"image"}, {1, 2}, {1, -2, 3});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: BucketBatchByLength op
/// Description: Test with element_length_function not specified and column_names has more than 1 element
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBucketBatchByLengthFail7) {
  // This should fail because element_length_function is not specified and column_names has more than 1 element.
  // Calling with function pointer
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBucketBatchByLengthFail7.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a BucketBatchByLength operation on ds
  ds = ds->BucketBatchByLength({"image", "label"}, {1, 2}, {1, 2, 3});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Concat op
/// Description: Test Concat op where the input column names of concatenated datasets are not the same
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestConcatFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatFail1.";
  // This case is expected to fail because the input column names of concatenated datasets are not the same

  // Create an ImageFolder Dataset
  // Column names: {"image", "label"}
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Rename operation on ds
  ds2 = ds2->Rename({"image", "label"}, {"col1", "col2"});
  EXPECT_NE(ds, nullptr);

  // Create a Concat operation on the ds
  // Name of datasets to concat doesn't not match
  ds = ds->Concat({ds2});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Concat op
/// Description: Test Concat op where the input dataset is empty
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestConcatFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatFail2.";
  // This case is expected to fail because the input dataset is empty.

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Concat operation on the ds
  // Input dataset to concat is empty
  ds = ds->Concat({});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Concat op
/// Description: Test Concat op where the input dataset is nullptr
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestConcatFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatFail3.";
  // This case is expected to fail because the input dataset is nullptr.

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Concat operation on the ds
  // Input dataset to concat is null
  ds = ds->Concat({nullptr});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Concat op
/// Description: Test Concat op where the input dataset is nullptr
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestConcatFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatFail4.";
  // This case is expected to fail because the input dataset is nullptr.

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Concat operation on the ds
  // Input dataset to concat is null
  ds = ds + nullptr;
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Concat op
/// Description: Test Concat op where the dataset concat itself which causes ProjectNode with two parent nodes
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestConcatFail5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatFail5.";
  // This case is expected to fail because the dataset concat itself which causes ProjectNode has two parent nodes

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds2, nullptr);

  // Create a Project operation on ds
  ds1 = ds1->Project({"image"});
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project({"image"});
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  // Input dataset is the dataset itself
  ds1 = ds1 + ds1 + ds2;
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  // Expect failure: The data pipeline is not a tree
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Concat op
/// Description: Test Concat op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConcatSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatSuccess.";

  // Create an ImageFolder Dataset
  // Column names: {"image", "label"}
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Cifar10 Dataset
  // Column names: {"image", "label"}
  folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds2 = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 9));
  EXPECT_NE(ds2, nullptr);

  // Create a Project operation on ds
  ds = ds->Project({"image"});
  EXPECT_NE(ds, nullptr);
  ds2 = ds2->Project({"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Concat operation on the ds
  ds = ds->Concat({ds2});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 19);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Concat op
/// Description: Test Concat op followed by GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConcatGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatGetDatasetSize.";

  // Create an ImageFolder Dataset
  // Column names: {"image", "label"}
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Cifar10 Dataset
  // Column names: {"image", "label"}
  folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds2 = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 9));
  EXPECT_NE(ds2, nullptr);

  // Create a Project operation on ds
  ds = ds->Project({"image"});
  EXPECT_NE(ds, nullptr);
  ds2 = ds2->Project({"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Concat operation on the ds
  ds = ds->Concat({ds2});
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 19);
}

/// Feature: Concat op
/// Description: Test Concat op using "+" operator to concat two datasets
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConcatSuccess2) {
  // Test "+" operator to concat two datasets
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatSuccess2.";

  // Create an ImageFolder Dataset
  // Column names: {"image", "label"}
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Cifar10 Dataset
  // Column names: {"image", "label"}
  folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds2 = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 9));
  EXPECT_NE(ds2, nullptr);

  // Create a Project operation on ds
  ds = ds->Project({"image"});
  EXPECT_NE(ds, nullptr);
  ds2 = ds2->Project({"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Concat operation on the ds
  ds = ds + ds2;
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 19);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Filter op
/// Description: Test Filter op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFilterSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFilterSuccess1.";
  // Test basic filter api with specific predicate to judge if label is equal to 3

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
  std::string schema_file = datasets_root_path_ + "/test_tf_file_3_images/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({data_file}, schema_file, {"image", "label"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>(true);
  EXPECT_NE(decode_op, nullptr);

  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{64, 64});
  EXPECT_NE(resize_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({decode_op, resize_op});
  EXPECT_NE(ds, nullptr);

  // Create a Filter operation on ds
  ds = ds->Filter(Predicate1, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<uint64_t> label_list;
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];

    std::shared_ptr<Tensor> de_label;
    uint64_t label_value;
    ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
    ASSERT_OK(de_label->GetItemAt(&label_value, {0}));
    label_list.push_back(label_value);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Only 1 column whose label is equal to 3
  EXPECT_EQ(i, 1);
  EXPECT_EQ(label_list.at(0), 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Filter op
/// Description: Test Filter op without input_columns
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFilterSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFilterSuccess2.";
  // Test filter api without input_columns

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
  std::string schema_file = datasets_root_path_ + "/test_tf_file_3_images/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({data_file}, schema_file, {"image", "label"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Filter operation on ds
  ds = ds->Filter(Predicate2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<uint64_t> label_list;
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];

    std::shared_ptr<Tensor> de_label;
    uint64_t label_value;
    ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
    ASSERT_OK(de_label->GetItemAt(&label_value, {0}));
    label_list.push_back(label_value);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  // There are 2 columns whose label is more than 1
  EXPECT_EQ(i, 2);
  EXPECT_EQ(label_list.at(0), 2);
  EXPECT_EQ(label_list.at(1), 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Filter op
/// Description: Test Filter op with nullptr predicate
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFilterFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFilterFail1.";
  // Test filter api with nullptr predicate

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
  std::string schema_file = datasets_root_path_ + "/test_tf_file_3_images/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({data_file}, schema_file, {"image", "label"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  std::function<MSTensorVec(MSTensorVec)> predicate_null = nullptr;

  // Create a Filter operation on ds
  ds = ds->Filter(predicate_null);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Filter input with nullptr predicate
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Filter op
/// Description: Test Filter op with wrong input_columns
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFilterFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFilterFail2.";
  // Test filter api with wrong input_columns

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
  std::string schema_file = datasets_root_path_ + "/test_tf_file_3_images/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({data_file}, schema_file, {"image", "label"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Filter operation on ds
  ds = ds->Filter(Predicate1, {"not_exist"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  EXPECT_ERROR(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    EXPECT_ERROR(iter->GetNextRow(&row));
  }

  // Expect failure: column check fail and return nothing
  EXPECT_EQ(i, 0);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Filter op
/// Description: Test Filter op with empty string as column name
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFilterFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFilterFail3.";
  // Test filter api with empty input_columns

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
  std::string schema_file = datasets_root_path_ + "/test_tf_file_3_images/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({data_file}, schema_file, {"image", "label"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Filter operation on ds
  ds = ds->Filter(Predicate1, {""});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Filter input with empty string of column name
  EXPECT_EQ(iter, nullptr);
}

/// Test ImageFolder with Batch and Repeat operations
void ImageFolderBatchAndRepeat(int32_t repeat_count, int32_t batch_size, int64_t num_samples, bool replacement,
                               std::string datasets_root_path) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path + "/testPK/data/";
  std::shared_ptr<Dataset> ds =
    ImageFolder(folder_path, true, std::make_shared<RandomSampler>(replacement, num_samples));
  uint64_t ds_size = 44;
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(repeat_count);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  uint64_t expect = 0;
  if (batch_size != 0) {
    if (num_samples == 0) {
      expect = ds_size * repeat_count / batch_size;
    } else {
      expect = num_samples * repeat_count / batch_size;
    }
  } else {
    expect = 0;
  }

  EXPECT_EQ(i, expect);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test ImageFolder with Batch and Repeat operations
/// Description: Perform Repeat and Batch ops with varying parameters, iterate through dataset and count rows
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestImageFolderBatchAndRepeat) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderBatchAndRepeat.";
  ImageFolderBatchAndRepeat(2, 2, 10, false, datasets_root_path_);
  ImageFolderBatchAndRepeat(2, 11, 0, false, datasets_root_path_);
  ImageFolderBatchAndRepeat(3, 2, 12, true, datasets_root_path_);
}

/// Feature: Test ImageFolder with Batch and Repeat operations
/// Description: Test ImageFolder with Repeat and Batch operations, followed by GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPipelineGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPipelineGetDatasetSize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 2;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 10);
}

/// Feature: GetDatasetSize
/// Description: Test distributed ImageFolder where num_per_shard is more than num_samples, followed by GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDistributedGetDatasetSize1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedGetDatasetSize1.";
  // Test get dataset size in distributed scenario when num_per_shard is more than num_samples

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds =
    ImageFolder(folder_path, true, std::make_shared<DistributedSampler>(4, 0, ShuffleMode::kFalse, 10));
  EXPECT_NE(ds, nullptr);

  // num_per_shard is equal to 44/4 = 11 which is more than num_samples = 10, so the output is 10
  EXPECT_EQ(ds->GetDatasetSize(), 10);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // The value of i should be equal to the result of get dataset size
  EXPECT_EQ(i, 10);
}

/// Feature: GetDatasetSize
/// Description: Test distributed ImageFolder where num_per_shard is less than num_samples, followed by GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDistributedGetDatasetSize2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedGetDatasetSize2.";
  // Test get dataset size in distributed scenario when num_per_shard is less than num_samples

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds =
    ImageFolder(folder_path, true, std::make_shared<DistributedSampler>(4, 0, ShuffleMode::kFalse, 15));
  EXPECT_NE(ds, nullptr);

  // num_per_shard is equal to 44/4 = 11 which is less than num_samples = 15, so the output is 11
  EXPECT_EQ(ds->GetDatasetSize(), 11);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // The value of i should be equal to the result of get dataset size
  EXPECT_EQ(i, 11);
}

/// Feature: Project and Map ops
/// Description: Test Project op after a Map op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestProjectMap) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestProjectMap.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(0.5);
  EXPECT_NE(random_vertical_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_vertical_flip_op}, {}, {});
  EXPECT_NE(ds, nullptr);

  // Create a Project operation on ds
  std::vector<std::string> column_project = {"image"};
  ds = ds->Project(column_project);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Project op
/// Description: Test Project op with duplicate column name
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestProjectDuplicateColumnFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestProjectDuplicateColumnFail.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(0.5);
  EXPECT_NE(random_vertical_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_vertical_flip_op}, {}, {});
  EXPECT_NE(ds, nullptr);

  // Create a Project operation on ds
  std::vector<std::string> column_project = {"image", "image"};

  // Create a Project operation on ds
  ds = ds->Project(column_project);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: duplicate project op column name
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Map op
/// Description: Test Map op with duplicate column name
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMapDuplicateColumnFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMapDuplicateColumnFail.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(0.5);
  EXPECT_NE(random_vertical_flip_op, nullptr);

  // Create a Map operation on ds
  auto ds1 = ds->Map({random_vertical_flip_op}, {"image", "image"}, {});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: duplicate Map op input column name
  EXPECT_EQ(iter1, nullptr);

  // Create a Map operation on ds
  auto ds2 = ds->Map({random_vertical_flip_op}, {}, {"label", "label"});
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: duplicate Map op output column name
  EXPECT_EQ(iter2, nullptr);

  // Create a Map operation on ds
  auto ds3 = ds->Map({random_vertical_flip_op}, {}, {});
  EXPECT_NE(ds3, nullptr);
}

/// Feature: Map op
/// Description: Test Map op with nullptr as the operation
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMapNullOperation) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMapNullOperation.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Map operation on ds
  std::shared_ptr<TensorTransform> operation = nullptr;
  auto ds1 = ds->Map({operation}, {"image"}, {});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: Operation is nullptr
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: Project and Map ops
/// Description: Test auto injection of Project op after Map op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestProjectMapAutoInjection) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline.TestProjectMapAutoInjection";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{30, 30});
  EXPECT_NE(resize_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({resize_op}, {}, {});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<ProjectDataset> project_ds = ds->Project({"image"});
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // 'label' is dropped during the project op
  EXPECT_EQ(row.find("label"), row.end());
  // 'image' column should still exist
  EXPECT_NE(row.find("image"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0], 30);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Rename op
/// Description: Test Rename op where input and output in Rename are not the same size
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRenameFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRenameFail1.";
  // We expect this test to fail because input and output in Rename are not the same size

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Rename operation on ds
  ds = ds->Rename({"image", "label"}, {"col2"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Rename op
/// Description: Test Rename op where input or output column name is empty
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRenameFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRenameFail2.";
  // We expect this test to fail because input or output column name is empty

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Rename operation on ds
  ds = ds->Rename({"image", "label"}, {"col2", ""});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Rename op
/// Description: Test Rename op with duplicate column name
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRenameFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRenameFail3.";
  // We expect this test to fail because duplicate column name

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Rename operation on ds
  auto ds1 = ds->Rename({"image", "image"}, {"col1", "col2"});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter1, nullptr);

  // Create a Rename operation on ds
  auto ds2 = ds->Rename({"image", "label"}, {"col1", "col1"});
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter2, nullptr);
}

/// Feature: Rename op
/// Description: Test Rename op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRenameSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRenameSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Rename operation on ds
  ds = ds->Rename({"image", "label"}, {"col1", "col2"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  EXPECT_NE(row.find("col1"), row.end());
  EXPECT_NE(row.find("col2"), row.end());
  EXPECT_EQ(row.find("image"), row.end());
  EXPECT_EQ(row.find("label"), row.end());

  while (row.size() != 0) {
    i++;
    auto image = row["col1"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Repeat op
/// Description: Test Repeat op with default inputs (repeat count is -1)
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRepeatDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRepeatDefault.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  // Default value of repeat count is -1, expected to repeat infinitely
  ds = ds->Repeat();
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  uint64_t i = 0;
  while (row.size() != 0) {
    // manually stop
    if (i == 100) {
      break;
    }
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 100);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Repeat op
/// Description: Test Repeat op with repeat count to be 1
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRepeatOne) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRepeatOne.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 1;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Repeat op
/// Description: Test Repeat op with invalid repeat_num=0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRepeatFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRepeatFail1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 0;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Repeat op
/// Description: Test Repeat op with invalid repeat_num=-2
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRepeatFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRepeatFail2.";
  // This case is expected to fail because the repeat count is invalid (<-1 && !=0).

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = -2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Shuffle op
/// Description: Test Shuffle op on ImageFolderDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestShuffleDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestShuffleDataset.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Shuffle operation on ds
  int32_t shuffle_size = 10;
  ds = ds->Shuffle(shuffle_size);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 2;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Test shuffle operation on TFRecord dataset
void TestShuffleTFRecord(int32_t shuffle_size, std::string dataset_root_path) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestShuffleTFRecord.";

  // Create an TFRecord Dataset
  std::string folder_path = dataset_root_path + "/testDataset1/testDataset1.data";
  std::shared_ptr<Dataset> ds = TFRecord({folder_path}, "", {}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Shuffle operation on ds
  ds = ds->Shuffle(shuffle_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test shuffle operation on TFRecord dataset
/// Description: Iterate through dataset with a shuffle size of 4 and 100 and count the number of rows
/// Expectation: There should be 10 rows in the dataset
TEST_F(MindDataTestPipeline, TestShuffleTFRecord) {
  TestShuffleTFRecord(4, datasets_root_path_);
  TestShuffleTFRecord(100, datasets_root_path_);
}

/// Feature: Test skip operation on TFRecord dataset
/// Description: Iterate through dataset with skip op
/// Expectation: Success
TEST_F(MindDataTestPipeline, TestSkipDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSkipDataset.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds
  int32_t count = 3;
  ds = ds->Skip(count);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 10-3=7 rows
  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Skip, Take, Repeat ops
/// Description: Test Skip, Project, Take, then Repeat op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSkipTakeRepeat) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSkipTakeRepeat.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 6));

  // Create a Skip operation on ds
  int32_t count = 0;
  ds = ds->Skip(count);

  // Create a Project operation on ds
  std::vector<std::string> column_project = {"image"};
  ds = ds->Project(column_project);

  // Add a Take(-1)
  ds = ds->Take(-1);

  // Add a Repeat(1)
  ds = ds->Repeat(1);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 6 rows
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Skip op
/// Description: Test Skip op followed by GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSkipGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSkipGetDatasetSize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds
  int32_t count = 3;
  ds = ds->Skip(count);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 7);
}

/// Feature: Skip op
/// Description: Test Skip op with invalid count input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSkipDatasetError1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSkipDatasetError1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds with invalid count input
  int32_t count = -1;
  ds = ds->Skip(count);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Skip input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Take op
/// Description: Test Take op with default count=-1
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestTakeDatasetDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTakeDatasetDefault.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds, default count = -1
  ds = ds->Take();
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 7 rows
  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Take op
/// Description: Test Take op followed by GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestTakeGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTakeGetDatasetSize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds, default count = -1
  ds = ds->Take(2);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
}

/// Feature: Take op
/// Description: Test Take op with invalid count input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestTakeDatasetError1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTakeDatasetError1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds with invalid count input
  int32_t count = -5;
  auto ds1 = ds->Take(count);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);

  // Create a Take operation on ds with invalid count input
  count = 0;
  auto ds2 = ds->Take(count);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  iter = ds2->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Take op
/// Description: Test Take op with valid count input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestTakeDatasetNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTakeDatasetNormal.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 8));
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds
  ds = ds->Take(5);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 5 rows
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Tensor and Map ops
/// Description: Test Tensor and Map ops
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestTensorOpsAndMap) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTensorOpsAndMap.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{30, 30});
  EXPECT_NE(resize_op, nullptr);

  auto center_crop_op = std::make_shared<vision::CenterCrop>(std::vector<int32_t>{16, 16});
  EXPECT_NE(center_crop_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({resize_op, center_crop_op});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 40);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Zip op
/// Description: Test Zip op with datasets that have image and label columns (same column names)
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestZipFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestZipFail.";
  // We expect this test to fail because we are the both datasets we are zipping have "image" and "label" columns
  // and zip doesn't accept datasets with same column names

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an ImageFolder Dataset
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds1, nullptr);

  // Create a Zip operation on the datasets
  ds = Zip({ds, ds1});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Zip op
/// Description: Test Zip op with empty input dataset
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestZipFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestZipFail2.";
  // This case is expected to fail because the input dataset is empty.

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Zip operation on the datasets
  // Input dataset to zip is empty
  ds = Zip({});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Op input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Zip op
/// Description: Test Zip op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestZipSuccess) {
  // Testing the member zip() function
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestZipSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Project operation on ds
  std::vector<std::string> column_project = {"image"};
  ds = ds->Project(column_project);
  EXPECT_NE(ds, nullptr);

  // Create an ImageFolder Dataset
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds1, nullptr);

  // Create a Rename operation on ds (so that the 3 datasets we are going to zip have distinct column names)
  ds1 = ds1->Rename({"image", "label"}, {"col1", "col2"});
  EXPECT_NE(ds1, nullptr);

  folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds2 = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds2, nullptr);

  // Create a Project operation on ds
  column_project = {"label"};
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Zip operation on the datasets
  ds = ds->Zip({ds1, ds2});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Check zipped column names
  EXPECT_EQ(row.size(), 4);
  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  EXPECT_NE(row.find("col1"), row.end());
  EXPECT_NE(row.find("col2"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Zip op
/// Description: Test Zip op followed by GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestZipGetDatasetSize) {
  // Testing the member zip() function
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestZipGetDatasetSize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create a Project operation on ds
  std::vector<std::string> column_project = {"image"};
  ds = ds->Project(column_project);
  EXPECT_NE(ds, nullptr);

  // Create an ImageFolder Dataset
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds1, nullptr);

  // Create a Rename operation on ds (so that the 3 datasets we are going to zip have distinct column names)
  ds1 = ds1->Rename({"image", "label"}, {"col1", "col2"});
  EXPECT_NE(ds1, nullptr);

  folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds2 = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds2, nullptr);

  // Create a Project operation on ds
  column_project = {"label"};
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Zip operation on the datasets
  ds = ds->Zip({ds1, ds2});
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
}

/// Feature: Zip op
/// Description: Test Zip op using static zip() function
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestZipSuccess2) {
  // Testing the static zip() function
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestZipSuccess2.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 9));
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds2, nullptr);

  // Create a Rename operation on ds (so that the 2 datasets we are going to zip have distinct column names)
  ds = ds->Rename({"image", "label"}, {"col1", "col2"});
  EXPECT_NE(ds, nullptr);

  // Create a Zip operation on the datasets
  ds = Zip({ds, ds2});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Check zipped column names
  EXPECT_EQ(row.size(), 4);
  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  EXPECT_NE(row.find("col1"), row.end());
  EXPECT_NE(row.find("col2"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 9);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SetNumWorkers op
/// Description: Test SetNumWorkers with various inputs
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNumWorkersValidate) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNumWorkersValidate.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 1));

  // ds needs to be non nullptr otherwise, the subsequent logic will core dump
  ASSERT_NE(ds, nullptr);

  // test if set num_workers=-1
  EXPECT_EQ(ds->SetNumWorkers(-1)->CreateIterator(), nullptr);

  // test if set num_workers can be very large
  EXPECT_EQ(ds->SetNumWorkers(INT32_MAX)->CreateIterator(), nullptr);

  int32_t cpu_core_cnt = GlobalContext::config_manager()->num_cpu_threads();

  // only do this test if cpu_core_cnt can be successfully obtained
  if (cpu_core_cnt > 0) {
    EXPECT_EQ(ds->SetNumWorkers(cpu_core_cnt + 1)->CreateIterator(), nullptr);
    // verify setting num_worker to 1 or cpu_core_cnt is allowed
    ASSERT_OK(ds->SetNumWorkers(cpu_core_cnt)->IRNode()->ValidateParams());
    ASSERT_OK(ds->SetNumWorkers(1)->IRNode()->ValidateParams());
  }
}

/// Feature: Test Concat operation on TFRecord dataset
/// Description: Perform Concat on two identical datasets, iterate through the product and count rows
/// Expectation: There should be 2 rows in the concatenated dataset (2 times original size)
TEST_F(MindDataTestPipeline, TestConcatTFRecord) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatSuccess.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";
  std::string schema_path = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema1Row.json";
  std::shared_ptr<Dataset> ds1 = TFRecord({file_path}, schema_path);
  EXPECT_NE(ds1, nullptr);

  // Create a TFRecord Dataset
  std::shared_ptr<Dataset> ds2 = TFRecord({file_path}, schema_path);
  EXPECT_NE(ds2, nullptr);

  // Create a Project operation on ds
  ds1 = ds1->Project({"col_sint16"});
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project({"col_sint16"});
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test ImageFolder with Sequential Sampler and Decode
/// Description: Create ImageFolder dataset with decode=true, iterate through dataset and count rows
/// Expectation: There should be 20 rows in the dataset (# of samples taken)
TEST_F(MindDataTestPipeline, TestImageFolderDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderDecode.";

  std::shared_ptr<Sampler> sampler = std::make_shared<SequentialSampler>(0, 20);
  EXPECT_NE(sampler, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);
  iter->Stop();
}

/// Feature: Test TFRecord with Take operation
/// Description: Perform Take operation with count = 5, iterate through dataset and count rows
/// Expectation: There should be 5 rows in the dataset
TEST_F(MindDataTestPipeline, TestTFRecordTake) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordTake.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";
  std::string schema_path = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({file_path}, schema_path);
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds
  ds = ds->Take(5);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 5 rows
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Skip operation on TFRecord dataset
/// Description: Perform skip operation with count = 5, iterate through dataset and count rows
/// Expectation: There should be 7 rows, (12 rows initially and 5 are skipped)
TEST_F(MindDataTestPipeline, TestTFRecordSkip) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordSkip.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";
  std::string schema_path = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({file_path}, schema_path);
  EXPECT_NE(ds, nullptr);

  // Create a Take operation on ds
  ds = ds->Skip(5);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 7 rows
  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Rename operation on TFRecord
/// Description: Rename columns in dataset, iterate through dataset and count rows
/// Expectation: The columns should have a new name after the Rename op and there should be 3 rows in the dataset
TEST_F(MindDataTestPipeline, TestTFRecordRename) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordRename.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
  std::shared_ptr<Dataset> ds = TFRecord({file_path});
  EXPECT_NE(ds, nullptr);

  // Create a Rename operation on ds
  ds = ds->Rename({"label"}, {"label1"});
  ds = ds->Rename({"label1", "image"}, {"label2", "image1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  EXPECT_NE(row.find("label2"), row.end());
  EXPECT_NE(row.find("image1"), row.end());
  EXPECT_EQ(row.find("image"), row.end());
  EXPECT_EQ(row.find("label"), row.end());
  EXPECT_EQ(row.find("label1"), row.end());

  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test TFRecord with Zip and Repeat operation
/// Description: Create two datasets and apply Zip operation on them.
///     Apply Repeat operation on resulting dataset and count rows
/// Expectation: There should be 9 rows in the dataset
TEST_F(MindDataTestPipeline, TestTFRecordZip) {
  // Testing the member zip() function
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordZip.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
  std::shared_ptr<Dataset> ds = TFRecord({file_path});
  EXPECT_NE(ds, nullptr);

  // Create a TFRecord Dataset
  std::string file_path1 = datasets_root_path_ + "/testBatchDataset/test.data";
  std::shared_ptr<Dataset> ds1 = TFRecord({file_path1});
  EXPECT_NE(ds1, nullptr);

  // Create a Zip operation on the datasets
  ds = ds->Zip({ds1});
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 9);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Repeat and Map with decode and resize ops on TFRecord
/// Description: Iterate through dataset and count the number of rows and check the shape of the image data
/// Expectation: There should be 6 rows in the dataset and shape is {30,30}
TEST_F(MindDataTestPipeline, TestTFRecordDecodeRepeatResize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline.TestTFRecordDecodeRepeatResize";

  // Create an ImageFolder Dataset
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images/train-0000-of-0001.data";
  std::shared_ptr<Dataset> ds = TFRecord({file_path}, "", {"image", "label"});
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::vector<int32_t> size = {30, 30};
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> resize_op = std::make_shared<vision::Resize>(size);
  EXPECT_NE(decode_op, nullptr);
  EXPECT_NE(resize_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({decode_op, resize_op}, {}, {});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<ProjectDataset> project_ds = ds->Project({"image"});
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // 'label' is dropped during the project op
  EXPECT_EQ(row.find("label"), row.end());
  // 'image' column should still exist
  EXPECT_NE(row.find("image"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0], 30);
    EXPECT_EQ(image.Shape()[1], 30);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Batch on TFRecord
/// Description: Iterate through dataset, count the number of rows and verify the data in the row
/// Expectation: There should be 1 row in the dataset and the data should the expected data
TEST_F(MindDataTestPipeline, TestBatch) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBatch.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/testBatchDataset/test.data";
  std::vector<std::string> files = {file_path};
  std::shared_ptr<Dataset> ds = TFRecord(files, nullptr, {}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 12;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> data = {-9223372036854775807 - 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9223372036854775807};

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;

    auto this_row = row["col_sint64"];
    auto value = this_row.Data();
    int64_t *p = (int64_t *)value.get();
    for (size_t j = 0; j < data.size(); j++) {
      EXPECT_EQ(p[j], data[j]);
    }

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Test Repeat and Batch on TFRecord
void TestRepeatBatch(bool drop, uint64_t expected_rows, std::string datasets_root_path) {
  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path + "/testBatchDataset/test.data";
  std::shared_ptr<Dataset> ds = TFRecord({file_path});
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 7;
  ds = ds->Batch(batch_size, drop);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, expected_rows);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Repeat and Batch on TFRecord
/// Description: Apply repeat then batch with drop on and off, count rows in the dataset
/// Expectation: The number of rows should equal the expected number of rows
TEST_F(MindDataTestPipeline, TestRepeatBatchDrop) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRepeatBatchDrop.";
  TestRepeatBatch(true, 3, datasets_root_path_);
  TestRepeatBatch(false, 4, datasets_root_path_);
}

/// Test Batch and Repeat on TFRecord
void TestBatchRepeat(bool drop, uint64_t expected_rows, std::string datasets_root_path) {
  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path + "/testBatchDataset/test.data";
  std::shared_ptr<Dataset> ds = TFRecord({file_path});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 7;
  ds = ds->Batch(batch_size, drop);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, expected_rows);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Batch and Repeat on TFRecord
/// Description: Apply batch then repeat with drop on and off, count rows in the dataset
/// Expectation: The number of rows should equal the expected number of rows
TEST_F(MindDataTestPipeline, TestBatchDropRepeat) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBatchDropRepeat.";
  TestBatchRepeat(true, 2, datasets_root_path_);
  TestBatchRepeat(false, 4, datasets_root_path_);
}

/// Feature: Test Map on TFRecord
/// Description: Apply Map with a TensorOp that does noting but swaps input columns with output column
/// Expectation: "Image" column is replaced with "X"
TEST_F(MindDataTestPipeline, TestMap) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline.TestMap";

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/testDataset2/testDataset2.data";
  std::string schema_file = datasets_root_path_ + "/testDataset2/datasetSchema.json";
  std::shared_ptr<Dataset> ds =
    TFRecord({data_file}, schema_file, {"image", "label", "A", "B"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> no_op = std::make_shared<mindspore::dataset::test::NoTransform>();
  EXPECT_NE(no_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({no_op}, {"image"}, {"X"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  while (row.size() != 0) {
    EXPECT_EQ(row.find("image"), row.end());
    EXPECT_NE(row.find("label"), row.end());
    EXPECT_NE(row.find("X"), row.end());
    EXPECT_NE(row.find("A"), row.end());
    EXPECT_NE(row.find("B"), row.end());

    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Map on TFRecord
/// Description: Apply Map with a TensorOp that swaps 3 input columns with 1 output column
/// Expectation: "Image", "A", "B" are replaced with "X"
TEST_F(MindDataTestPipeline, Test3to1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline.Test3to1";

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/testDataset2/testDataset2.data";
  std::string schema_file = datasets_root_path_ + "/testDataset2/datasetSchema.json";
  std::shared_ptr<Dataset> ds =
    TFRecord({data_file}, schema_file, {"image", "label", "A", "B"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> three_to_one_op = std::make_shared<mindspore::dataset::test::ThreeToOneTransform>();
  EXPECT_NE(three_to_one_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({three_to_one_op}, {"image", "A", "B"}, {"X"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  while (row.size() != 0) {
    EXPECT_EQ(row.find("image"), row.end());
    EXPECT_NE(row.find("label"), row.end());
    EXPECT_NE(row.find("X"), row.end());
    EXPECT_EQ(row.find("A"), row.end());
    EXPECT_EQ(row.find("B"), row.end());

    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Map on TFRecord
/// Description: Apply Map with a TensorOp that swaps 1 input column with 3 output columns
/// Expectation: "Image" is replaced with "X", "Y", "Z"
TEST_F(MindDataTestPipeline, Test1to3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline.Test1to3";

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/testDataset2/testDataset2.data";
  std::string schema_file = datasets_root_path_ + "/testDataset2/datasetSchema.json";
  std::shared_ptr<Dataset> ds =
    TFRecord({data_file}, schema_file, {"image", "label", "A", "B"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_to_three_op = std::make_shared<mindspore::dataset::test::OneToThreeTransform>();
  EXPECT_NE(one_to_three_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_to_three_op}, {"image"}, {"X", "Y", "Z"});
  EXPECT_NE(ds, nullptr);

  ds = ds->Project({"X", "Y", "Z", "label", "A", "B"});

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  while (row.size() != 0) {
    EXPECT_EQ(row.find("image"), row.end());
    EXPECT_NE(row.find("label"), row.end());
    EXPECT_NE(row.find("A"), row.end());
    EXPECT_NE(row.find("B"), row.end());
    EXPECT_NE(row.find("X"), row.end());
    EXPECT_NE(row.find("Y"), row.end());
    EXPECT_NE(row.find("Z"), row.end());

    EXPECT_EQ(row["X"].Shape(), std::vector<int64_t>({3, 4, 2}));
    EXPECT_EQ(row["Y"].Shape(), std::vector<int64_t>({3, 4, 2}));
    EXPECT_EQ(row["Z"].Shape(), std::vector<int64_t>({3, 4, 2}));
    EXPECT_EQ(row["A"].Shape(), std::vector<int64_t>({1, 13, 14, 12}));
    EXPECT_EQ(row["B"].Shape(), std::vector<int64_t>({9}));

    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Map on TFRecord
/// Description: Apply 3to1 and then 1to3 to replace 3 input columns with 3 output columns
/// Expectation: "Image", "A", "B" are replaced with "X", "y", "Z"
TEST_F(MindDataTestPipeline, TestMultiTensorOp) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline.TestMultiTensorOp";

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/testDataset2/testDataset2.data";
  std::string schema_file = datasets_root_path_ + "/testDataset2/datasetSchema.json";
  std::shared_ptr<Dataset> ds =
    TFRecord({data_file}, schema_file, {"image", "label", "A", "B"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> three_to_one_op = std::make_shared<mindspore::dataset::test::ThreeToOneTransform>();
  std::shared_ptr<TensorTransform> one_to_three_op = std::make_shared<mindspore::dataset::test::OneToThreeTransform>();
  EXPECT_NE(one_to_three_op, nullptr);
  EXPECT_NE(three_to_one_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({three_to_one_op, one_to_three_op}, {"image", "A", "B"}, {"X", "Y", "Z"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  while (row.size() != 0) {
    EXPECT_EQ(row.find("image"), row.end());
    EXPECT_NE(row.find("label"), row.end());
    EXPECT_EQ(row.find("A"), row.end());
    EXPECT_EQ(row.find("B"), row.end());
    EXPECT_NE(row.find("X"), row.end());
    EXPECT_NE(row.find("Y"), row.end());
    EXPECT_NE(row.find("Z"), row.end());

    EXPECT_EQ(row["X"].Shape(), std::vector<int64_t>({3, 4, 2}));
    EXPECT_EQ(row["Y"].Shape(), std::vector<int64_t>({3, 4, 2}));
    EXPECT_EQ(row["Z"].Shape(), std::vector<int64_t>({3, 4, 2}));

    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Test Repeat and Map on TFRecord
/// Description: Apply Map with NoOp and Repeat with num_repeats=3, iterate through dataset and count rows
/// Expectation: There should  be 10 rows in the dataset
TEST_F(MindDataTestPipeline, TestTFReaderRepeatMap) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline.TestTFReaderRepeatMap";

  // Create a TFRecord Dataset
  std::string data_file = datasets_root_path_ + "/testDataset2/testDataset2.data";
  std::string schema_file = datasets_root_path_ + "/testDataset2/datasetSchema.json";
  std::shared_ptr<Dataset> ds =
    TFRecord({data_file}, schema_file, {"image", "label", "A", "B"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> no_op = std::make_shared<mindspore::dataset::test::NoTransform>();
  EXPECT_NE(no_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({no_op}, {"label"}, {});
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 30);

  // Manually terminate the pipeline
  iter->Stop();
}
