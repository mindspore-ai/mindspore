/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "include/api/types.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/vision.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TensorRow VecToRow(const MSTensorVec &v) {
  TensorRow row;
  for (const mindspore::MSTensor &t : v) {
    std::shared_ptr<Tensor> rt;
    Status rc =
      Tensor::CreateFromMemory(TensorShape(t.Shape()), MSTypeToDEType(static_cast<mindspore::TypeId>(t.DataType())),
                               (const uchar *)(t.Data().get()), t.DataSize(), &rt);
    row.emplace_back(rt);
  }
  return row;
}
MSTensorVec RowToVec(const TensorRow &v) {
  MSTensorVec rv;  // std::make_shared<DETensor>(de_tensor)
  std::transform(v.begin(), v.end(), std::back_inserter(rv), [](std::shared_ptr<Tensor> t) -> mindspore::MSTensor {
    return mindspore::MSTensor(std::make_shared<DETensor>(t));
  });
  return rv;
}

MSTensorVec BucketBatchTestFunction(MSTensorVec input) {
  mindspore::dataset::TensorRow output;
  std::shared_ptr<Tensor> out;
  Tensor::CreateEmpty(mindspore::dataset::TensorShape({1}),
                      mindspore::dataset::DataType(mindspore::dataset::DataType::Type::DE_INT32), &out);
  out->SetItemAt({0}, 2);
  output.push_back(out);
  return RowToVec(output);
}

MSTensorVec Predicate1(MSTensorVec in) {
  // Return true if input is equal to 3
  uint64_t input_value;
  TensorRow input = VecToRow(in);
  input.at(0)->GetItemAt(&input_value, {0});
  bool result = (input_value == 3);

  // Convert from boolean to TensorRow
  TensorRow output;
  std::shared_ptr<Tensor> out;
  Tensor::CreateEmpty(mindspore::dataset::TensorShape({1}),
                      mindspore::dataset::DataType(mindspore::dataset::DataType::Type::DE_BOOL), &out);
  out->SetItemAt({0}, result);
  output.push_back(out);

  return RowToVec(output);
}

MSTensorVec Predicate2(MSTensorVec in) {
  // Return true if label is more than 1
  // The index of label in input is 1
  uint64_t input_value;
  TensorRow input = VecToRow(in);
  input.at(1)->GetItemAt(&input_value, {0});
  bool result = (input_value > 1);

  // Convert from boolean to TensorRow
  TensorRow output;
  std::shared_ptr<Tensor> out;
  Tensor::CreateEmpty(mindspore::dataset::TensorShape({1}),
                      mindspore::dataset::DataType(mindspore::dataset::DataType::Type::DE_BOOL), &out);
  out->SetItemAt({0}, result);
  output.push_back(out);

  return RowToVec(output);
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }
  // 2 batches of size 5
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }
  // With 2 boundaries, 3 buckets are created
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 19);
  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 19);
  // Manually terminate the pipeline
  iter->Stop();
}

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

  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({64, 64}));
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
  iter->GetNextRow(&row);

  std::vector<uint64_t> label_list;
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];

    std::shared_ptr<Tensor> de_label;
    uint64_t label_value;
    ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
    de_label->GetItemAt(&label_value, {0});
    label_list.push_back(label_value);

    iter->GetNextRow(&row);
  }

  // Only 1 column whose label is equal to 3
  EXPECT_EQ(i, 1);
  EXPECT_EQ(label_list.at(0), 3);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  std::vector<uint64_t> label_list;
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];

    std::shared_ptr<Tensor> de_label;
    uint64_t label_value;
    ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
    de_label->GetItemAt(&label_value, {0});
    label_list.push_back(label_value);

    iter->GetNextRow(&row);
  }

  // There are 2 columns whose label is more than 1
  EXPECT_EQ(i, 2);
  EXPECT_EQ(label_list.at(0), 2);
  EXPECT_EQ(label_list.at(1), 3);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  // Expect failure: column check fail and return nothing
  EXPECT_EQ(i, 0);

  // Manually terminate the pipeline
  iter->Stop();
}

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

TEST_F(MindDataTestPipeline, TestImageFolderBatchAndRepeat) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderBatchAndRepeat.";

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

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

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

TEST_F(MindDataTestPipeline, TestDistributedGetDatasetSize1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedGetDatasetSize1.";
  // Test get dataset size in distributed scenario when num_per_shard is more than num_samples

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<DistributedSampler>(4, 0, false, 10));
  EXPECT_NE(ds, nullptr);

  // num_per_shard is equal to 44/4 = 11 which is more than num_samples = 10, so the output is 10
  EXPECT_EQ(ds->GetDatasetSize(), 10);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  // The value of i should be equal to the result of get dataset size
  EXPECT_EQ(i, 10);
}

TEST_F(MindDataTestPipeline, TestDistributedGetDatasetSize2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedGetDatasetSize2.";
  // Test get dataset size in distributed scenario when num_per_shard is less than num_samples

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<DistributedSampler>(4, 0, false, 15));
  EXPECT_NE(ds, nullptr);

  // num_per_shard is equal to 44/4 = 11 which is less than num_samples = 15, so the output is 11
  EXPECT_EQ(ds->GetDatasetSize(), 11);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  // The value of i should be equal to the result of get dataset size
  EXPECT_EQ(i, 11);
}

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
  ds = ds->Map({random_vertical_flip_op}, {}, {}, {"image", "label"});
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  ds = ds->Map({random_vertical_flip_op}, {}, {}, {"image", "label"});
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
  auto ds1 = ds->Map({random_vertical_flip_op}, {"image", "image"}, {}, {});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: duplicate Map op input column name
  EXPECT_EQ(iter1, nullptr);

  // Create a Map operation on ds
  auto ds2 = ds->Map({random_vertical_flip_op}, {}, {"label", "label"}, {});
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: duplicate Map op output column name
  EXPECT_EQ(iter2, nullptr);

  // Create a Map operation on ds
  auto ds3 = ds->Map({random_vertical_flip_op}, {}, {}, {"image", "image"});
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: duplicate Map op project column name
  EXPECT_EQ(iter3, nullptr);
}

TEST_F(MindDataTestPipeline, TestMapNullOperation) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMapNullOperation.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Map operation on ds
  std::shared_ptr<TensorTransform> operation = nullptr;
  auto ds1 = ds->Map({operation}, {"image"}, {}, {});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: Operation is nullptr
  EXPECT_EQ(iter1, nullptr);
}

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
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30, 30}));
  EXPECT_NE(resize_op, nullptr);

  // Create a Map operation on ds
  // {"image"} is the project columns. This will trigger auto injection of ProjectOp after MapOp.
  ds = ds->Map({resize_op}, {}, {}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

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
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  EXPECT_NE(row.find("col1"), row.end());
  EXPECT_NE(row.find("col2"), row.end());
  EXPECT_EQ(row.find("image"), row.end());
  EXPECT_EQ(row.find("label"), row.end());

  while (row.size() != 0) {
    i++;
    auto image = row["col1"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    // manually stop
    if (i == 100) {
      break;
    }
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 100);
  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);
  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 10-3=7 rows
  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 6 rows
  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 7 rows
  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }
  MS_LOG(INFO) << "Number of rows: " << i;

  // Expect 5 rows
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30, 30}));
  EXPECT_NE(resize_op, nullptr);

  std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 40);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

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
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

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
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 9);

  // Manually terminate the pipeline
  iter->Stop();
}

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
