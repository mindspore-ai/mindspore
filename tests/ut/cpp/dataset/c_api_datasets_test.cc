/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/core/tensor.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for datasets (in alphabetical order)

/// Feature: CelebADataset
/// Description: Test CelebADataset with all usage and SequentialSampler
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCelebADataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebADataset.";

  // Create a CelebA Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path, "all", std::make_shared<SequentialSampler>(0, 2), false, {});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Check if CelebA() read correct images/attr
  std::string expect_file[] = {"1_apple.JPEG", "2_banana.jpg"};
  std::vector<std::vector<uint32_t>> expect_attr_vector = {
     {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
      0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1},
     {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto attr = row["attr"];

    mindspore::MSTensor expect_image = ReadFileToTensor(folder_path + expect_file[i]);
    EXPECT_MSTENSOR_EQ(image, expect_image);

    std::shared_ptr<Tensor> de_expect_attr;
    ASSERT_OK(Tensor::CreateFromVector(expect_attr_vector[i], TensorShape({40}), &de_expect_attr));
    mindspore::MSTensor expect_attr =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_attr));
    EXPECT_MSTENSOR_EQ(attr, expect_attr);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CelebADataset
/// Description: Test CelebADataset with default inputs
/// Expectation: The data is processed successfully
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Check if CelebA() read correct images/attr
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto attr = row["attr"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor attr shape: " << attr.Shape();

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

// Feature: Test Repeat operation on CelebA dataset
// Description: Perform repeat operation with count = 2, and count rows in the dataset
// Expectation: Dataset should have 8 rows (2 times original size since it is repeated)
TEST_F(MindDataTestPipeline, TestCelebARepeat) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebARepeat.";

  std::string file_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(file_path);
  ds = ds->Repeat(2);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect a valid iterator
  ASSERT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Verify correct number of rows fetched
  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

// Feature: Test SubsetRandomSampler on CelebA dataset
// Description: Create dataset with SubsetRandomSampler given a single index and count rows in the dataset
// Expectation: Dataset should have 1 row
TEST_F(MindDataTestPipeline, TestCelebASubsetRandomSampler) {
  std::vector<int64_t> indices = {1};

  std::shared_ptr<Sampler> sampl = std::make_shared<SubsetRandomSampler>(indices);
  EXPECT_NE(sampl, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path, "all", sampl);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: GetRepeatCount
/// Description: Test GetRepeatCount on ImageFolderDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetRepeatCount) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetRepeatCount.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  ds = ds->Repeat(4);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetRepeatCount(), 4);
  ds = ds->Repeat(3);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetRepeatCount(), 3);
}

/// Feature: GetBatchSize
/// Description: Test GetBatchSize on ImageFolderDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetBatchSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetRepeatCount.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true)->Project({"label"});
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  ds = ds->Batch(2);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetBatchSize(), 2);
  ds = ds->Batch(3);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetBatchSize(), 3);
}
TEST_F(MindDataTestPipeline, TestCelebAGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebAGetDatasetSize.";

  // Create a CelebA Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path, "valid");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 1);
}

/// Feature: CelebADataset
/// Description: Test CelebADataset with invalid dataset path and invalid dataset type
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCelebAError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebAError.";

  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::string invalid_folder_path = "./testNotExist";
  std::string invalid_dataset_type = "invalid_type";

  // Create a CelebA Dataset
  std::shared_ptr<Dataset> ds1 = CelebA(invalid_folder_path);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid CelebA input, invalid dataset path
  EXPECT_EQ(iter1, nullptr);

  // Create a CelebA Dataset
  std::shared_ptr<Dataset> ds2 = CelebA(folder_path, invalid_dataset_type);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid CelebA input, invalid dataset type
  EXPECT_EQ(iter2, nullptr);
}

/// Feature: CelebADataset
/// Description: Test CelebADataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCelebADatasetWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCelebADataset.";

  // Create a CelebA Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path, "all", nullptr, false, {});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid CelebA input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with wrong dataset directory
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestImageFolderWithWrongDatasetDirFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderWithWrongDatasetDirFail.";

  // Create an ImageFolder Dataset
  std::shared_ptr<Dataset> ds = ImageFolder("", true, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid ImageFolder input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with wrong extension
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestImageFolderFailWithWrongExtensionFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithWrongExtensionFail.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2), {".JGP"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect no data: cannot find files with specified extension
  EXPECT_ERROR(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 0);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestImageFolderGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderGetDatasetSize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 44);
  EXPECT_EQ(ds->GetNumClasses(), 4);
  EXPECT_EQ(ds->GetNumClasses(), 4);
  EXPECT_EQ(ds->GetDatasetSize(), 44);
  EXPECT_EQ(ds->GetDatasetSize(), 44);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestImageFolderFailWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithNullSamplerFail.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid ImageFolder input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with wrong sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestImageFolderFailWithWrongSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithWrongSamplerFail.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(-2, 5));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid ImageFolder input, sampler is not constructed correctly
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MnistDataset
/// Description: Test MnistDataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMnistGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMnistGetDatasetSize.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetDatasetSize(), 20);
}

/// Feature: MnistDataset
/// Description: Test MnistDataset with wrong dataset directory
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMnistFailWithWrongDatasetDirFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMnistFailWithWrongDatasetDirFail.";

  // Create a Mnist Dataset
  std::shared_ptr<Dataset> ds = Mnist("", "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Mnist input, incorrect dataset directory input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MnistDataset
/// Description: Test MnistDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMnistFailWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMnistFailWithNullSamplerFail.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Mnist input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with class_index
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestImageFolderClassIndexDatasetSize) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  std::map<std::string, int32_t> class_index;
  class_index["class1"] = 111;
  class_index["class2"] = 333;
  auto ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, class_index);
  EXPECT_EQ(ds->GetNumClasses(), 2);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with wrong class in class_index
/// Expectation: GetNumClasses should return -1
TEST_F(MindDataTestPipeline, TestImageFolderClassIndexDatasetSizeFail) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  std::map<std::string, int32_t> class_index;
  class_index["class1"] = 111;
  class_index["wrong class"] = 333;
  auto ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, class_index);
  EXPECT_EQ(ds->GetNumClasses(), -1);
}

/// Feature: GetClassIndexing
/// Description: Test GetClassIndexing on ImageFolderDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestImageFolderClassIndexing) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderClassIndexing.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path);
  EXPECT_NE(ds, nullptr);
  std::vector<std::pair<std::string, std::vector<int32_t>>> indexing = ds->GetClassIndexing();

  EXPECT_EQ(indexing.size(), 4);
  EXPECT_EQ(indexing[0].first, "class1");
  EXPECT_EQ(indexing[0].second[0], 0);
  EXPECT_EQ(indexing[1].first, "class2");
  EXPECT_EQ(indexing[1].second[0], 1);
  EXPECT_EQ(indexing[2].first, "class3");
  EXPECT_EQ(indexing[2].second[0], 2);
  EXPECT_EQ(indexing[3].first, "class4");
  EXPECT_EQ(indexing[3].second[0], 3);

  std::map<std::string, int32_t> class_index;
  class_index["class1"] = 0;
  class_index["class2"] = 1;
  class_index["class3"] = 9;
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, class_index);
  std::vector<std::pair<std::string, std::vector<int32_t>>> indexing2 = ds2->GetClassIndexing();

  EXPECT_EQ(indexing2.size(), 3);
  EXPECT_EQ(indexing2[0].first, "class1");
  EXPECT_EQ(indexing2[0].second[0], 0);
  EXPECT_EQ(indexing2[1].first, "class2");
  EXPECT_EQ(indexing2[1].second[0], 1);
  EXPECT_EQ(indexing2[2].first, "class3");
  EXPECT_EQ(indexing2[2].second[0], 9);
}
