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
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestImageFolderWithSamplers) {
  std::shared_ptr<SamplerObj> sampl = DistributedSampler(2, 1);
  EXPECT_NE(sampl, nullptr);

  sampl = PKSampler(3);
  EXPECT_NE(sampl, nullptr);

  sampl = RandomSampler(false, 12);
  EXPECT_NE(sampl, nullptr);

  sampl = SequentialSampler(0, 12);
  EXPECT_NE(sampl, nullptr);

  std::vector<double> weights = {0.9, 0.8, 0.68, 0.7, 0.71, 0.6, 0.5, 0.4, 0.3, 0.5, 0.2, 0.1};
  sampl = WeightedRandomSampler(weights, 12);
  EXPECT_NE(sampl, nullptr);

  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
  sampl = SubsetSampler(indices);
  EXPECT_NE(sampl, nullptr);

  sampl = SubsetRandomSampler(indices);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCalculateNumSamples) {
  int64_t num_rows = 30;  // dummy variable for number of rows in the dataset
  std::shared_ptr<SamplerObj> sampl = DistributedSampler(2, 1, false, 6);
  EXPECT_NE(sampl, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt = sampl->SamplerBuild();
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 6);

  sampl = PKSampler(3, false);
  EXPECT_NE(sampl, nullptr);
  sampler_rt = sampl->SamplerBuild();
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 30);

  sampl = RandomSampler(false, 12);
  EXPECT_NE(sampl, nullptr);
  sampler_rt = sampl->SamplerBuild();
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 12);

  sampl = SequentialSampler(0, 10);
  EXPECT_NE(sampl, nullptr);
  sampler_rt = sampl->SamplerBuild();
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 10);

  std::vector<double> weights = {0.9, 0.8, 0.68, 0.7, 0.71, 0.6, 0.5, 0.4, 0.3, 0.5, 0.2, 0.1};
  sampl = WeightedRandomSampler(weights, 12);
  EXPECT_NE(sampl, nullptr);
  sampler_rt = sampl->SamplerBuild();
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 12);

  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21};
  sampl = SubsetRandomSampler(indices, 11);
  EXPECT_NE(sampl, nullptr);
  sampler_rt = sampl->SamplerBuild();
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 11);

  // Testing chains
  // Parent and child have num_samples
  std::shared_ptr<SamplerObj> sampl1 = WeightedRandomSampler(weights, 12);
  EXPECT_NE(sampl1, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt1 = sampl1->SamplerBuild();

  std::shared_ptr<SamplerObj> sampl2 = SequentialSampler(0, 10);
  EXPECT_NE(sampl2, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt2 = sampl2->SamplerBuild();
  sampler_rt2->AddChild(sampler_rt1);
  EXPECT_EQ(sampler_rt2->CalculateNumSamples(num_rows), 10);

  // Parent doesn't have num_samples
  std::shared_ptr<SamplerObj> sampl3 = WeightedRandomSampler(weights, 12);
  EXPECT_NE(sampl3, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt3 = sampl3->SamplerBuild();

  std::shared_ptr<SamplerObj> sampl4 = SubsetRandomSampler(indices);
  EXPECT_NE(sampl4, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt4 = sampl4->SamplerBuild();
  sampler_rt4->AddChild(sampler_rt3);
  EXPECT_EQ(sampler_rt4->CalculateNumSamples(num_rows), 11);

  // Child doesn't have num_samples
  std::shared_ptr<SamplerObj> sampl5 = RandomSampler(false);
  EXPECT_NE(sampl5, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt5 = sampl5->SamplerBuild();

  std::shared_ptr<SamplerObj> sampl6 = PKSampler(3, false, 7);
  EXPECT_NE(sampl6, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt6 = sampl6->SamplerBuild();
  sampler_rt6->AddChild(sampler_rt5);
  EXPECT_EQ(sampler_rt6->CalculateNumSamples(num_rows), 7);
}

TEST_F(MindDataTestPipeline, TestSamplersMoveParameters) {
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
  std::shared_ptr<SamplerObj> sampl1 = SubsetRandomSampler(indices);
  EXPECT_FALSE(indices.empty());
  EXPECT_NE(sampl1->SamplerBuild(), nullptr);
  std::shared_ptr<SamplerObj> sampl2 = SubsetRandomSampler(std::move(indices));
  EXPECT_TRUE(indices.empty());
  EXPECT_NE(sampl2->SamplerBuild(), nullptr);
}

TEST_F(MindDataTestPipeline, TestWeightedRandomSamplerFail) {
  // weights is empty
  std::vector<double> weights1 = {};
  std::shared_ptr<SamplerObj> sampl1 = WeightedRandomSampler(weights1);
  EXPECT_EQ(sampl1, nullptr);

  // weights has negative number
  std::vector<double> weights2 = {0.5, 0.2, -0.4};
  std::shared_ptr<SamplerObj> sampl2 = WeightedRandomSampler(weights2);
  EXPECT_EQ(sampl2, nullptr);

  // weights elements are all zero
  std::vector<double> weights3 = {0, 0, 0};
  std::shared_ptr<SamplerObj> sampl3 = WeightedRandomSampler(weights3);
  EXPECT_EQ(sampl3, nullptr);
}

TEST_F(MindDataTestPipeline, TestDistributedSamplerSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerSuccess.";
  // Test basic setting of distributed_sampler

  // num_shards=4, shard_id=0, shuffle=false, num_samplers=0, seed=0, offset=-1, even_dist=true
  std::shared_ptr<SamplerObj> sampler = DistributedSampler(4, 0, false, 0, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
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

TEST_F(MindDataTestPipeline, TestSamplerAddChild) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSamplerAddChild.";

  auto sampler = DistributedSampler(1, 0, false, 5, 0, -1, true);
  EXPECT_NE(sampler, nullptr);

  auto child_sampler = SequentialSampler();
  sampler->AddChildSampler(child_sampler);
  EXPECT_NE(child_sampler, nullptr);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, sampler);
  EXPECT_NE(ds, nullptr);

  // Iterate the dataset and get each row
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(ds->GetDatasetSize(), 5);
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestDistributedSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDistributedSamplerFail.";
  // Test invalid offset setting of distributed_sampler

  // offset=5 which is greater than num_shards=4
  std::shared_ptr<SamplerObj> sampler = DistributedSampler(4, 0, false, 0, 0, 5, false);
  EXPECT_EQ(sampler, nullptr);
}
