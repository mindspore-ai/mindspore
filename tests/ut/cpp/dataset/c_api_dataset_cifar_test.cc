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

#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestCifar10Dataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10Dataset.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("label"), row.end());

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

TEST_F(MindDataTestPipeline, TestCifar10DatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10DatasetWithPipeline.";

  // Create two Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds1 = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  std::shared_ptr<Dataset> ds2 = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 1;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 1;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"image", "label"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("label"), row.end());

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

TEST_F(MindDataTestPipeline, TestCifar10GetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10GetDatasetSize.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 10000);
}

TEST_F(MindDataTestPipeline, TestCifar10Getters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10MixGetter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 10000);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint32");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].ToString(), "<32,32,3>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 10000);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), -1);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 10000);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetDatasetSize(), 10000);
}

TEST_F(MindDataTestPipeline, TestCifar100Dataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100Dataset.";

  // Create a Cifar100 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("coarse_label"), row.end());
  EXPECT_NE(row.find("fine_label"), row.end());

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

TEST_F(MindDataTestPipeline, TestCifar100Getters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100Getters.";

  // Create a Cifar100 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"image", "coarse_label", "fine_label"};
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  int64_t num_classes = ds->GetNumClasses();

  EXPECT_EQ(types.size(), 3);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint32");
  EXPECT_EQ(types[2].ToString(), "uint32");
  EXPECT_EQ(shapes.size(), 3);
  EXPECT_EQ(shapes[0].ToString(), "<32,32,3>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(shapes[2].ToString(), "<>");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 10);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

TEST_F(MindDataTestPipeline, TestCifar100DatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100DatasetFail.";

  // Create a Cifar100 Dataset
  std::shared_ptr<Dataset> ds = Cifar100("", "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Cifar100 input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar10DatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10DatasetFail.";

  // Create a Cifar10 Dataset
  std::shared_ptr<Dataset> ds = Cifar10("", "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Cifar10 input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar10DatasetWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10DatasetWithNullSamplerFail.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "validation");
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Cifar10 input, validation is not a valid usage
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar10DatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10DatasetWithNullSamplerFail.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Cifar10 input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar100DatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100DatasetWithNullSamplerFail.";

  // Create a Cifar100 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Cifar100 input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar100DatasetWithWrongSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100DatasetWithWrongSamplerFail.";

  // Create a Cifar100 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", std::make_shared<RandomSampler>(false, -10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Cifar100 input, sampler is not constructed correctly
  EXPECT_EQ(iter, nullptr);
}
