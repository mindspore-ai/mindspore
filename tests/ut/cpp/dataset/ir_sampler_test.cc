/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/distributed_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/pk_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/prebuilt_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/random_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/sequential_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/skip_first_epoch_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/subset_random_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/subset_sampler_ir.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/weighted_random_sampler_ir.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestIrSampler : public UT::DatasetOpTesting {
 protected:
};

/// Feature: MindData IR Sampler Support
/// Description: Test CalculateNumSamples with various SamplerObj
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestIrSampler, TestCalculateNumSamples) {
  int64_t num_rows = 30;  // dummy variable for number of rows in the dataset
  std::shared_ptr<SamplerObj> sampl = std::make_shared<DistributedSamplerObj>(2, 1, false, 6, 1, -1, true);
  EXPECT_NE(sampl, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt;
  sampl->SamplerBuild(&sampler_rt);
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 6);

  sampl = std::make_shared<PKSamplerObj>(3, false, 0);
  EXPECT_NE(sampl, nullptr);
  sampl->SamplerBuild(&sampler_rt);
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), -1);

  sampl = std::make_shared<RandomSamplerObj>(false, 12);
  EXPECT_NE(sampl, nullptr);
  sampl->SamplerBuild(&sampler_rt);
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 12);

  sampl = std::make_shared<SequentialSamplerObj>(0, 10);
  EXPECT_NE(sampl, nullptr);
  sampl->SamplerBuild(&sampler_rt);
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 10);

  std::vector<double> weights = {0.9, 0.8, 0.68, 0.7, 0.71, 0.6, 0.5, 0.4, 0.3, 0.5, 0.2, 0.1};
  sampl = std::make_shared<WeightedRandomSamplerObj>(weights, 12);
  EXPECT_NE(sampl, nullptr);
  sampl->SamplerBuild(&sampler_rt);
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 12);

  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21};
  sampl = std::make_shared<SubsetRandomSamplerObj>(indices, 11);
  EXPECT_NE(sampl, nullptr);
  sampl->SamplerBuild(&sampler_rt);
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), 11);

  sampl = std::make_shared<SkipFirstEpochSamplerObj>(0);
  EXPECT_NE(sampl, nullptr);
  sampl->SamplerBuild(&sampler_rt);
  EXPECT_EQ(sampler_rt->CalculateNumSamples(num_rows), -1);

  // Testing chains
  // Parent and child have num_samples
  std::shared_ptr<SamplerObj> sampl1 = std::make_shared<WeightedRandomSamplerObj>(weights, 12);
  EXPECT_NE(sampl1, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt1;
  sampl1->SamplerBuild(&sampler_rt1);

  std::shared_ptr<SamplerObj> sampl2 = std::make_shared<SequentialSamplerObj>(0, 10);
  EXPECT_NE(sampl2, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt2;
  sampl2->SamplerBuild(&sampler_rt2);
  sampler_rt2->AddChild(sampler_rt1);
  EXPECT_EQ(sampler_rt2->CalculateNumSamples(num_rows), 10);

  // Parent doesn't have num_samples
  std::shared_ptr<SamplerObj> sampl3 = std::make_shared<WeightedRandomSamplerObj>(weights, 12);
  EXPECT_NE(sampl3, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt3;
  sampl3->SamplerBuild(&sampler_rt3);

  std::shared_ptr<SamplerObj> sampl4 = std::make_shared<SubsetRandomSamplerObj>(indices, 0);
  EXPECT_NE(sampl4, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt4;
  sampl4->SamplerBuild(&sampler_rt4);
  sampler_rt4->AddChild(sampler_rt3);
  EXPECT_EQ(sampler_rt4->CalculateNumSamples(num_rows), 11);

  // Child doesn't have num_samples
  std::shared_ptr<SamplerObj> sampl5 = std::make_shared<RandomSamplerObj>(false, 0);
  EXPECT_NE(sampl5, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt5;
  sampl5->SamplerBuild(&sampler_rt5);

  std::shared_ptr<SamplerObj> sampl6 = std::make_shared<PKSamplerObj>(3, false, 7);
  EXPECT_NE(sampl6, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt6;
  sampl6->SamplerBuild(&sampler_rt6);
  sampler_rt6->AddChild(sampler_rt5);
  EXPECT_EQ(sampler_rt6->CalculateNumSamples(num_rows), -1);

  std::shared_ptr<SamplerObj> sampl7 = std::make_shared<SkipFirstEpochSamplerObj>(0);
  EXPECT_NE(sampl7, nullptr);
  std::shared_ptr<SamplerRT> sampler_rt7;
  sampl7->SamplerBuild(&sampler_rt7);
  sampler_rt7->AddChild(sampler_rt5);
  EXPECT_EQ(sampler_rt7->CalculateNumSamples(num_rows), -1);
}

/// Feature: MindData IR Sampler Support
/// Description: Test samplers move parameter with indices (array of int64) and std::move(indices)
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestIrSampler, TestSamplersMoveParameters) {
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23};
  std::shared_ptr<SamplerObj> sampl1 = std::make_shared<SubsetRandomSamplerObj>(indices, 0);
  EXPECT_FALSE(indices.empty());
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  sampl1->SamplerBuild(&sampler_rt);
  EXPECT_NE(sampler_rt, nullptr);
  std::shared_ptr<SamplerObj> sampl2 = std::make_shared<SubsetRandomSamplerObj>(std::move(indices), 0);
  EXPECT_TRUE(indices.empty());
  std::shared_ptr<SamplerRT> sampler_rt2 = nullptr;
  sampl2->SamplerBuild(&sampler_rt2);
  EXPECT_NE(sampler_rt, nullptr);
}

/// Feature: MindData IR Sampler Support
/// Description: Test MindData IR Sampler by Compile more than one epoch
/// Expectation: Results are successfully outputted, first epoch has fewer rows.
TEST_F(MindDataTestIrSampler, TestSkipFirstEpochSampler) {
  MS_LOG(INFO) << "Doing MindDataTestIrSampler-TestSkipFirstEpochSampler.";
  std::string dataset_dir = "./data/dataset/testPK/data";
  std::set<std::string> extensions = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::map<std::string, int32_t> class_indexing = {};
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SkipFirstEpochSamplerObj>(1);
  std::shared_ptr<DatasetNode> ds =
    std::make_shared<ImageFolderNode>(dataset_dir, false, sampler, false, extensions, class_indexing, cache);
  auto ir_tree = std::make_shared<TreeAdapter>();
  // Compile with more than one epoch
  int32_t num_epoch = 3;
  EXPECT_OK(ir_tree->Compile(ds, num_epoch, 0));

  for (int i = 0; i < num_epoch; i++) {
    TensorRow row;
    ir_tree->GetNext(&row);
    int count = 0;
    while (row.size() != 0) {
      ir_tree->GetNext(&row);
      count++;
    }
    if (i == 0) {
      EXPECT_EQ(count, 43);
    } else {
      EXPECT_EQ(count, 44);
    }
  }
}

/// Feature: MindData IR Sampler Support
/// Description: Compare SequentialSampler and SkipFirstEpochSampler with More Than One Epoch
/// Expectation: SequentialSampler and SkipFirstEpochSampler have similar output
TEST_F(MindDataTestIrSampler, CompareSequentialSamplerAndSkipFirstEpochSampler) {
  MS_LOG(INFO) << "Doing MindDataTestIrSampler-CompareSequentialSamplerAndSkipFirstEpochSampler.";
  std::string dataset_dir = "./data/dataset/testPK/data";
  std::set<std::string> extensions = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::map<std::string, int32_t> class_indexing = {};
  int32_t skip_num = 2;
  std::shared_ptr<SamplerObj> sampler1 = std::make_shared<SequentialSamplerObj>(skip_num, 0);
  std::shared_ptr<SamplerObj> sampler2 = std::make_shared<SkipFirstEpochSamplerObj>(skip_num);
  std::shared_ptr<DatasetNode> ds1 =
    std::make_shared<ImageFolderNode>(dataset_dir, false, sampler1, false, extensions, class_indexing, cache);
  std::shared_ptr<DatasetNode> ds2 =
    std::make_shared<ImageFolderNode>(dataset_dir, false, sampler2, false, extensions, class_indexing, cache);
  auto ir_tree1 = std::make_shared<TreeAdapter>();
  auto ir_tree2 = std::make_shared<TreeAdapter>();
  // Compile with more than one epoch
  int32_t num_epoch = 3;
  EXPECT_OK(ir_tree1->Compile(ds1, num_epoch, 0));
  EXPECT_OK(ir_tree2->Compile(ds2, num_epoch, 0));

  for (int i = 0; i < num_epoch; i++) {
    TensorRow row1;
    TensorRow row2;
    // only the first epoch has the same output
    if (i != 0) {
      // SkipFirstEpochSampler doesn't skip after the first epoch
      for (int j = 0; j < skip_num; j++) {
        EXPECT_OK(ir_tree2->GetNext(&row2));
      }
    }
    EXPECT_OK(ir_tree1->GetNext(&row1));
    EXPECT_OK(ir_tree2->GetNext(&row2));
    EXPECT_EQ(row1.size(), row2.size());
    while (row1.size() != 0 && row2.size() != 0) {
      std::vector<std::shared_ptr<Tensor>> r1 = row1.getRow();
      std::vector<std::shared_ptr<Tensor>> r2 = row2.getRow();
      ASSERT_EQ(r1.size(), r2.size());
      for (int i = 0; i < r1.size(); i++) {
        nlohmann::json out_json1;
        EXPECT_OK(r1[i]->to_json(&out_json1));
        std::stringstream json_ss1;
        json_ss1 << out_json1;

        nlohmann::json out_json2;
        EXPECT_OK(r2[i]->to_json(&out_json2));
        std::stringstream json_ss2;
        json_ss2 << out_json2;
        EXPECT_EQ(json_ss1.str(), json_ss2.str());
      }
      EXPECT_OK(ir_tree1->GetNext(&row1));
      EXPECT_OK(ir_tree2->GetNext(&row2));
    }
  }
}