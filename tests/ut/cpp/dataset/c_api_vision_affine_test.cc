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
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"

using namespace mindspore::dataset;
using mindspore::dataset::InterpolationMode;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestAffineAPI) {
  MS_LOG(INFO) << "Doing MindDataTestAffine-TestAffineAPI.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);

  // Create auto contrast object with default values

  std::shared_ptr<TensorTransform> crop(new vision::RandomCrop({256, 256}));
  std::shared_ptr<TensorTransform> affine(
    new vision::Affine(0.0, {0.0, 0.0}, 0.0, {0.0, 0.0}, InterpolationMode::kLinear));

  // Create a Map operation on ds
  ds = ds->Map({crop, affine});

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(row["image"].Shape().at(0), 256);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestAffineAPIFail) {
  MS_LOG(INFO) << "Doing MindDataTestAffine-TestAffineAPI.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);

  // Create auto contrast object with default values

  std::shared_ptr<TensorTransform> crop(new vision::RandomCrop({256, 256}));
  std::shared_ptr<TensorTransform> affine(
    new vision::Affine(0.0, {2.0, -1.0}, 0.0, {0.0, 0.0}, InterpolationMode::kLinear));

  // Create a Map operation on ds
  ds = ds->Map({crop, affine});

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}
