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
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "mindspore/ccsrc/minddata/dataset/core/tensor.h"
#include "mindspore/ccsrc/minddata/dataset/core/data_type.h"

using namespace mindspore::dataset;
using mindspore::dataset::BorderType;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for data transforms ops (in alphabetical order)

/// Feature: Compose op
/// Description: Test Compose op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestComposeSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto decode_op = std::make_shared<vision::Decode>();
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{777, 777});
  transforms::Compose compose({decode_op, resize_op});

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
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
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    EXPECT_EQ(image.Shape()[0], 777);
    EXPECT_EQ(image.Shape()[1], 777);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Compose op
/// Description: Test Compose op with invalid transform op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestComposeFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail1 with invalid transform.";

  // Create a synthetic image/label dataset
  std::shared_ptr<Dataset> ds = test::ImageSource(10, {32, 32, 3});
  EXPECT_NE(ds, nullptr);

  // Resize: Non-positive size value: -1 at element: 0
  // Compose: transform ops must not be null
  auto decode_op = vision::Decode();
  auto resize_op = vision::Resize({-1});
  auto compose = transforms::Compose({decode_op, resize_op});

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Compose parameter(invalid transform op)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Compose op
/// Description: Test Compose op with null transform op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestComposeFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail2 with invalid transform.";

  // Create a synthetic image/label dataset
  std::shared_ptr<Dataset> ds = test::ImageSource(10, {32, 32, 3});
  EXPECT_NE(ds, nullptr);

  // Compose: transform ops must not be null
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  auto compose =
    std::make_shared<transforms::Compose>(std::vector<std::shared_ptr<TensorTransform>>{decode_op, nullptr});

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Compose parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Compose op
/// Description: Test Compose op with empty transform list
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestComposeFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail3 with invalid transform.";

  // Create a synthetic image/label dataset
  std::shared_ptr<Dataset> ds = test::ImageSource(10, {32, 32, 3});
  EXPECT_NE(ds, nullptr);

  // Compose: transform list must not be empty
  std::vector<std::shared_ptr<TensorTransform>> list = {};
  auto compose = transforms::Compose(list);

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Compose parameter (transform list must not be empty)
  EXPECT_EQ(iter, nullptr);
}
