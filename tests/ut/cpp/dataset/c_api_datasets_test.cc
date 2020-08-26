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
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset::api;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestCelebADataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebADataset.";

  // Create a CelebA Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path, "all", SequentialSampler(0, 2), false, {});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if CelebAOp read correct images/attr
  std::string expect_file[] = {"1.JPEG", "2.jpg"};
  std::vector<std::vector<uint32_t>> expect_attr_vector =
    {{0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0,
      1, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 1}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto attr = row["attr"];

    std::shared_ptr<Tensor> expect_image;
    Tensor::CreateFromFile(folder_path + expect_file[i], &expect_image);
    EXPECT_EQ(*image, *expect_image);

    std::shared_ptr<Tensor> expect_attr;
    Tensor::CreateFromVector(expect_attr_vector[i], TensorShape({40}), &expect_attr);
    EXPECT_EQ(*attr, *expect_attr);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCelebADefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebADefault.";

  // Create a CelebA Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  // Check if CelebAOp read correct images/attr
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto attr = row["attr"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    MS_LOG(INFO) << "Tensor attr shape: " << attr->shape();

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCelebAException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebAException.";

  // Create a CelebA Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::string invalid_folder_path = "./testNotExist";
  std::string invalid_dataset_type = "invalid_type";
  std::shared_ptr<Dataset> ds = CelebA(invalid_folder_path);
  EXPECT_EQ(ds, nullptr);
  std::shared_ptr<Dataset> ds1 = CelebA(folder_path, invalid_dataset_type);
  EXPECT_EQ(ds1, nullptr);
}

TEST_F(MindDataTestPipeline, TestCelebADatasetWithNullSampler) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebADataset.";

  // Create a CelebA Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path, "all", nullptr, false, {});
  // Expect failure: sampler can not be nullptr
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestMnistFailWithWrongDatasetDir) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMnistFailWithWrongDatasetDir.";

  // Create a Mnist Dataset
  std::shared_ptr<Dataset> ds = Mnist("", RandomSampler(false, 10));
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestMnistFailWithNullSampler) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMnistFailWithNullSampler.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, nullptr);
  // Expect failure: sampler can not be nullptr
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestImageFolderWithWrongDatasetDir) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderWithWrongDatasetDir.";

  // Create an ImageFolder Dataset
  std::shared_ptr<Dataset> ds = ImageFolder("", true, nullptr);
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestImageFolderFailWithWrongExtension) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithWrongExtension.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 2), {".JGP"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);
  // Expect no data: can not find files with specified extension
  EXPECT_EQ(row.size(), 0);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestImageFolderFailWithNullSampler) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithNullSampler.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, nullptr);
  // Expect failure: sampler can not be nullptr
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestImageFolderFailWithWrongSampler) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithWrongSampler.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, SequentialSampler(-2, 5));
  // Expect failure: sampler is not construnced correctly
  EXPECT_EQ(ds, nullptr);
}
