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

#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#include "minddata/dataset/engine/ir/datasetops/bucket_batch_by_length_node.h"
#include "minddata/dataset/engine/ir/datasetops/concat_node.h"
#include "minddata/dataset/engine/ir/datasetops/project_node.h"
#include "minddata/dataset/engine/ir/datasetops/rename_node.h"
#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"

using namespace mindspore::dataset::api;
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

TEST_F(MindDataTestPipeline, TestSamplersMoveParameters) {
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
  std::shared_ptr<SamplerObj> sampl1 = SubsetRandomSampler(indices);
  EXPECT_FALSE(indices.empty());
  EXPECT_NE(sampl1->Build(), nullptr);
  std::shared_ptr<SamplerObj> sampl2 = SubsetRandomSampler(std::move(indices));
  EXPECT_TRUE(indices.empty());
  EXPECT_NE(sampl2->Build(), nullptr);
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
