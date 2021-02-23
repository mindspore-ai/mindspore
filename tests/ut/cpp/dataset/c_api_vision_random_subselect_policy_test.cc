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
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API RandomSelectSubpolicy TensorTransform Operations

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicySuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicySuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 7));
  EXPECT_NE(ds, nullptr);

  /* FIXME - Resolve RandomSelectSubpolicy to properly handle TensorTransform input
  // Create objects for the tensor ops
  // Valid case: TensorTransform is not null and probability is between (0,1)
  std::shared_ptr<TensorTransform> random_select_subpolicy(new vision::RandomSelectSubpolicy(
    {{{vision::Invert(), 0.5}, {vision::Equalize(), 0.5}}, {{vision::Resize({15, 15}), 1}}}));
  EXPECT_NE(random_select_subpolicy, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy});
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
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
  */
}

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicyFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicyFail.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  /* FIXME - Resolve RandomSelectSubpolicy to properly handle TensorTransform input
  // RandomSelectSubpolicy : probability of transform must be between 0.0 and 1.0
  std::shared_ptr<TensorTransform> random_select_subpolicy1(new vision::RandomSelectSubpolicy(
    {{{vision::Invert(), 1.5}, {vision::Equalize(), 0.5}}, {{vision::Resize({15, 15}), 1}}}));
  EXPECT_NE(random_select_subpolicy1, nullptr);

  // RandomSelectSubpolicy: policy must not be empty
  std::shared_ptr<TensorTransform> random_select_subpolicy2(new vision::RandomSelectSubpolicy({{{vision::Invert(), 0.5}, {vision::Equalize(), 0.5}}, {{nullptr, 1}}}));
  EXPECT_NE(random_select_subpolicy2, nullptr);

  // RandomSelectSubpolicy: policy must not be empty
  std::shared_ptr<TensorTransform> random_select_subpolicy3(new vision::RandomSelectSubpolicy({}));
  EXPECT_NE(random_select_subpolicy3, nullptr);

  // RandomSelectSubpolicy: policy must not be empty
  std::shared_ptr<TensorTransform> random_select_subpolicy4(new vision::RandomSelectSubpolicy({{{vision::Invert(), 0.5}, {vision::Equalize(), 0.5}}, {}}));
  EXPECT_NE(random_select_subpolicy4, nullptr);

  // RandomSelectSubpolicy: policy must not be empty
  std::shared_ptr<TensorTransform> random_select_subpolicy5(new vision::RandomSelectSubpolicy({{{}, {vision::Equalize(), 0.5}}, {{vision::Resize({15, 15}), 1}}}));
  EXPECT_NE(random_select_subpolicy5, nullptr);
  */
}
