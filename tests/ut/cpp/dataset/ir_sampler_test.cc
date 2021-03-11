/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/tensor.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestIrSampler : public UT::DatasetOpTesting {
 protected:
};

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
}

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
