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
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include <functional>

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Sampler
/// Description: Test ImageFolderDataset with various samplers
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestImageFolderWithSamplers) {
  std::shared_ptr<Sampler> sampl = std::make_shared<DistributedSampler>(2, 1);
  EXPECT_NE(sampl, nullptr);

  sampl = std::make_shared<PKSampler>(3);
  EXPECT_NE(sampl, nullptr);

  sampl = std::make_shared<RandomSampler>(false, 12);
  EXPECT_NE(sampl, nullptr);

  sampl = std::make_shared<SequentialSampler>(0, 12);
  EXPECT_NE(sampl, nullptr);

  std::vector<double> weights = {0.9, 0.8, 0.68, 0.7, 0.71, 0.6, 0.5, 0.4, 0.3, 0.5, 0.2, 0.1};
  sampl = std::make_shared<WeightedRandomSampler>(weights, 12);
  EXPECT_NE(sampl, nullptr);

  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
  sampl = std::make_shared<SubsetSampler>(indices);
  EXPECT_NE(sampl, nullptr);

  sampl = std::make_shared<SubsetRandomSampler>(indices);
  EXPECT_NE(sampl, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampl);
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

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

// Feature: Test ImageFolder with WeightedRandomSampler
// Description: Create ImageFolder dataset with WeightedRandomRampler given num_samples=12,
// iterate through dataset and count rows
// Expectation: There should be 12 rows
TEST_F(MindDataTestPipeline, TestWeightedRandomSamplerImageFolder) {
  std::vector<double> weights = {0.9, 0.8, 0.68, 0.7, 0.71, 0.6, 0.5, 0.4, 0.3, 0.5, 0.2, 0.1};
  std::shared_ptr<Sampler> sampl = std::make_shared<WeightedRandomSampler>(weights, 12);
  EXPECT_NE(sampl, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampl);
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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test ImageFolderDataset with no sampler provided (defaults to RandomSampler)
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNoSamplerSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNoSamplerSuccess1.";
  // Test building a dataset with no sampler provided (defaults to random sampler

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, ds->GetDatasetSize());
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test basic setting of DistributedSampler through shared pointer
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDistributedSamplerSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerSuccess1.";
  // Test basic setting of distributed_sampler

  // num_shards=4, shard_id=0, shuffle=false, num_samplers=0, seed=0, offset=-1, even_dist=true
  std::shared_ptr<Sampler> sampler = std::make_shared<DistributedSampler>(4, 0, false, 0, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 11);
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test basic setting of DistributedSampler through new definition
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDistributedSamplerSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerSuccess2.";
  // Test basic setting of distributed_sampler

  // num_shards=4, shard_id=0, shuffle=false, num_samplers=0, seed=0, offset=-1, even_dist=true
  auto sampler(new DistributedSampler(4, 0, false, 0, 0, -1, true));
  // Note that with new, we have to explicitly delete the allocated object as shown below.
  // Note: No need to check for output after calling API class constructor

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 11);
  iter->Stop();

  // Delete allocated objects with raw pointers
  delete sampler;
}

/// Feature: Sampler
/// Description: Test basic setting of DistributedSampler through object definition
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDistributedSamplerSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerSuccess3.";
  // Test basic setting of distributed_sampler

  // num_shards=4, shard_id=0, shuffle=false, num_samplers=0, seed=0, offset=-1, even_dist=true
  DistributedSampler sampler = DistributedSampler(4, 0, false, 0, 0, -1, true);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 11);
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test pointer of DistributedSampler
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDistributedSamplerSuccess4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerSuccess4.";
  // Test pointer of distributed_sampler
  SequentialSampler sampler = SequentialSampler(0, 4);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, false, &sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);
  iter->Stop();
}

// Feature: Test ImageFolder with DistributedSampler
// Description: Create ImageFolder dataset with DistributedSampler given num_shards=11 and shard_id=10,
// count rows in dataset
// Expectation: There should be 4 rows (44 rows in original data/11 = 4)
TEST_F(MindDataTestPipeline, TestDistributedSamplerSuccess5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerSuccess5.";
  // Test basic setting of distributed_sampler

  // num_shards=11, shard_id=10, shuffle=false, num_samplers=0, seed=0, offset=-1, even_dist=true
  std::shared_ptr<Sampler> sampler = std::make_shared<DistributedSampler>(11, 10, false, 0, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);
  iter->Stop();
}

// Feature: Test ImageFolder with DistributedSampler
// Description: Create ImageFolder dataset with DistributedSampler given num_shards=4 and shard_id=3,
// count rows in dataset
// Expectation: There should be 11 rows (44 rows in original data/4 = 11)
TEST_F(MindDataTestPipeline, TestDistributedSamplerSuccess6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerSuccess6.";
  // Test basic setting of distributed_sampler

  // num_shards=4, shard_id=3, shuffle=false, num_samplers=12, seed=0, offset=-1, even_dist=true
  std::shared_ptr<Sampler> sampler = std::make_shared<DistributedSampler>(4, 3, false, 12, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 11);
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test DistributedSampler with num_shards < offset through shared pointer
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDistributedSamplerFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerFail1.";
  // Test basic setting of distributed_sampler

  // num_shards=4, shard_id=0, shuffle=false, num_samplers=0, seed=0, offset=5, even_dist=true
  // offset=5 which is greater than num_shards=4 --> will fail later
  std::shared_ptr<Sampler> sampler = std::make_shared<DistributedSampler>(4, 0, false, 0, 0, 5, false);
  EXPECT_NE(sampler, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate will fail because sampler is not initiated successfully.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Sampler
/// Description: Test DistributedSampler with num_shards < offset through new definition
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDistributedSamplerFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerFail2.";
  // Test basic setting of distributed_sampler

  // num_shards=4, shard_id=0, shuffle=false, num_samplers=0, seed=0, offset=5, even_dist=true
  // offset=5 which is greater than num_shards=4 --> will fail later
  auto sampler(new DistributedSampler(4, 0, false, 0, 0, 5, false));
  // Note that with new, we have to explicitly delete the allocated object as shown below.
  // Note: No need to check for output after calling API class constructor

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate will fail because sampler is not initiated successfully.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);

  // Delete allocated objects with raw pointers
  delete sampler;
}

/// Feature: Sampler
/// Description: Test DistributedSampler with num_shards < offset through object definition
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDistributedSamplerFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerFail3.";
  // Test basic setting of distributed_sampler

  // num_shards=4, shard_id=0, shuffle=false, num_samplers=0, seed=0, offset=5, even_dist=true
  // offset=5 which is greater than num_shards=4 --> will fail later
  DistributedSampler sampler = DistributedSampler(4, 0, false, 0, 0, 5, false);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate will fail because sampler is not initiated successfully.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Sampler
/// Description: Test DistributedSampler with SequentialSampler as child sampler
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSamplerAddChild) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSamplerAddChild.";

  auto sampler = std::make_shared<DistributedSampler>(1, 0, false, 5, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  auto child_sampler = std::make_shared<SequentialSampler>();
  EXPECT_NE(child_sampler, nullptr);

  sampler->AddChild(child_sampler);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
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

  EXPECT_EQ(ds->GetDatasetSize(), 5);
  iter->Stop();
}

/// Feature: MindData Sampler Support
/// Description: Test MindData Sampler AddChild with nested children
/// Expectation: Result dataset has expected number of samples.
TEST_F(MindDataTestPipeline, TestSamplerAddChild2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSamplerAddChild2.";

  // num_samples of parent sampler > num_sampler of child sampler, namely 5 > 2, num_shards is 2 to output dataset with
  // 1 sampler
  auto sampler = std::make_shared<DistributedSampler>(2, 0, false, 5, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  // num_samples of parent sampler > num_samples of child sampler, namely 4 > 2
  auto child_sampler = std::make_shared<RandomSampler>(true, 4);
  EXPECT_NE(child_sampler, nullptr);
  auto child_sampler2 = std::make_shared<SequentialSampler>(0, 2);
  EXPECT_NE(child_sampler2, nullptr);

  child_sampler->AddChild(child_sampler2);
  sampler->AddChild(child_sampler);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
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
  EXPECT_EQ(i, 1);

  EXPECT_EQ(ds->GetDatasetSize(), 1);
  iter->Stop();
}

/// Feature: MindData Sampler Support
/// Description: Test MindData Sampler AddChild with num_samples of parent sampler > num_samples of child sampler
/// Expectation: Result dataset has expected number of samples.
TEST_F(MindDataTestPipeline, TestSamplerAddChild3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSamplerAddChild3.";

  // num_samples of parent sampler > num_samples of child sampler, namely 5 > 4
  std::vector<double> weights = {1.0, 0.1, 0.02, 0.3};
  auto sampler = std::make_shared<WeightedRandomSampler>(weights, 5);
  EXPECT_NE(sampler, nullptr);

  auto child_sampler = std::make_shared<SequentialSampler>(0, 4);
  EXPECT_NE(child_sampler, nullptr);

  sampler->AddChild(child_sampler);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
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
  EXPECT_EQ(i, 4);

  EXPECT_EQ(ds->GetDatasetSize(), 4);
  iter->Stop();
}

/// Feature: MindData Sampler Support
/// Description: Test MindData Sampler AddChild with num_samples of parent sampler < num_samples of child sampler
/// Expectation: Result dataset has expected number of samples.
TEST_F(MindDataTestPipeline, TestSamplerAddChild4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSamplerAddChild4.";

  // num_samples of parent sampler < num_samples of child sampler, namely 5 < 7
  auto sampler = std::make_shared<DistributedSampler>(1, 0, false, 5, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  auto child_sampler = std::make_shared<PKSampler>(3, true, 7);
  EXPECT_NE(child_sampler, nullptr);

  sampler->AddChild(child_sampler);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
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
  EXPECT_EQ(i, 5);

  EXPECT_EQ(ds->GetDatasetSize(), 5);
  iter->Stop();
}

/// Feature: MindData Sampler Support
/// Description: Test MindData Sampler AddChild with several children
/// Expectation: Result dataset has expected number of samples, and output error messages for more than 1 child.
TEST_F(MindDataTestPipeline, TestSamplerAddChild5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSamplerAddChild5.";

  // Use all samples (num_sampler=0) for parent DistributedSampler
  auto sampler = std::make_shared<DistributedSampler>(1, 0, false, 0, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  auto child_sampler1 = std::make_shared<SequentialSampler>(0, 10);
  EXPECT_NE(child_sampler1, nullptr);
  sampler->AddChild(child_sampler1);

  // Attempt to add more than one child_sampler is expected to fail
  auto child_sampler2 = std::make_shared<SequentialSampler>(0, 6);
  EXPECT_NE(child_sampler2, nullptr);
  sampler->AddChild(child_sampler2);

  auto child_sampler3 = std::make_shared<SequentialSampler>(0, 7);
  EXPECT_NE(child_sampler3, nullptr);
  sampler->AddChild(child_sampler3);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
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
  EXPECT_EQ(i, 10);

  EXPECT_EQ(ds->GetDatasetSize(), 10);
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test basic setting of SubsetSampler with default num_samples
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSubsetSamplerSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSubsetSamplerSuccess1.";
  // Test basic setting of subset_sampler with default num_samples

  std::vector<int64_t> indices = {2, 4, 6, 8, 10, 12};
  std::shared_ptr<Sampler> sampl = std::make_shared<SubsetSampler>(indices);
  EXPECT_NE(sampl, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampl);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 6);
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test SubsetSampler with num_samples provided
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSubsetSamplerSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSubsetSamplerSuccess2.";
  // Test subset_sampler with num_samples

  std::vector<int64_t> indices = {2, 4, 6, 8, 10, 12};
  std::shared_ptr<Sampler> sampl = std::make_shared<SubsetSampler>(indices, 3);
  EXPECT_NE(sampl, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampl);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test SubsetSampler with num_samples larger than the indices size
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSubsetSamplerSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSubsetSamplerSuccess3.";
  // Test subset_sampler with num_samples larger than the indices size.

  std::vector<int64_t> indices = {2, 4, 6, 8, 10, 12};
  std::shared_ptr<Sampler> sampl = std::make_shared<SubsetSampler>(indices, 8);
  EXPECT_NE(sampl, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampl);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 6);
  iter->Stop();
}

/// Feature: Sampler
/// Description: Test SubsetSampler with index out of bounds
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSubsetSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSubsetSamplerFail.";
  // Test subset_sampler with index out of bounds.

  std::vector<int64_t> indices = {2, 4, 6, 8, 10, 100};  // Sample ID (100) is out of bound
  std::shared_ptr<Sampler> sampl = std::make_shared<SubsetSampler>(indices);
  EXPECT_NE(sampl, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampl);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect failure: index 100 is out of dataset bounds
  EXPECT_ERROR(iter->GetNextRow(&row));

  iter->Stop();
}

// Feature: Test ImageFolder with PKSampler
// Description: Create ImageFolder dataset with DistributedSampler given num_val=3 and count rows
// Expectation: There should be 12 rows
TEST_F(MindDataTestPipeline, TestPKSamplerImageFolder) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPKSamplerImageFolder.";

  std::shared_ptr<Sampler> sampler = std::make_shared<PKSampler>(3, false);
  EXPECT_NE(sampler, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
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

  EXPECT_EQ(i, 12);
  iter->Stop();
}