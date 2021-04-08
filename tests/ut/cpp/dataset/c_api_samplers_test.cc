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
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "minddata/dataset/include/datasets.h"
#include <functional>

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, ds->GetDatasetSize());
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 11);
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 11);
  iter->Stop();

  // Delete allocated objects with raw pointers
  delete sampler;
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 11);
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);
  iter->Stop();
}

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(ds->GetDatasetSize(), 5);
  iter->Stop();
}
