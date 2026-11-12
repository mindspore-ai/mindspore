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
#include <opencv2/opencv.hpp>

#include "common/common.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/vision/kernels/image_utils.h"

using namespace mindspore::dataset;
using mindspore::dataset::BorderType;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API A to Q TensorTransform Operations (in alphabetical order)

/// Feature: AdjustGamma op
/// Description: Test AdjustGamma op on 3 channels dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAdjustGamma3Channel) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustGamma3Channel.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds2, nullptr);

  auto adjustgamma_op = vision::AdjustGamma(10.0);

  ds1 = ds1->Map({adjustgamma_op});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  iter1->GetNextRow(&row1);

  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row2;
  iter2->GetNextRow(&row2);

  uint64_t i = 0;
  while (row1.size() != 0) {
    i++;
    auto image = row1["image"];
    iter1->GetNextRow(&row1);
    iter2->GetNextRow(&row2);
  }
  EXPECT_EQ(i, 2);

  iter1->Stop();
  iter2->Stop();
}

/// Feature: AdjustGamma op
/// Description: Test AdjustGamma op on 3 channels dataset transformed to 1 channel dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAdjustGamma1Channel) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustGamma1Channel.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds2, nullptr);

  auto adjustgamma_op = vision::AdjustGamma(10.0);
  auto rgb2gray_op = vision::RGB2GRAY();

  ds1 = ds1->Map({rgb2gray_op, adjustgamma_op});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  iter1->GetNextRow(&row1);

  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row2;
  iter2->GetNextRow(&row2);

  uint64_t i = 0;
  while (row1.size() != 0) {
    i++;
    auto image = row1["image"];
    iter1->GetNextRow(&row1);
    iter2->GetNextRow(&row2);
  }
  EXPECT_EQ(i, 2);

  iter1->Stop();
  iter2->Stop();
}

/// Feature: AdjustGamma op
/// Description: Test AdjustGamma op with negative gamma
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestAdjustGammaParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustGammaParamCheck.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Negative gamma
  // Create objects for the tensor ops
  auto adjust_gamma = std::make_shared<vision::AdjustGamma>(-1, 1.0);
  auto ds1 = ds->Map({adjust_gamma});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid value of AdjustGamma
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: CenterCrop op
/// Description: Test CenterCrop op with single integer input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCenterCrop) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCenterCrop with single integer input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create centre crop object with square crop
  auto centre_out1 = std::make_shared<vision::CenterCrop>(std::vector<int32_t>{30});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({centre_out1});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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

  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CenterCrop
/// Description: Use batched dataset as video inputs
/// Expectation: The log will print correct shape
TEST_F(MindDataTestPipeline, TestCenterCropBatch) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCenterCropBatch.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);
  // Create a Batch operation on ds, choose batch size 5 to test high dimension input
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create centre crop object with square crop
  const std::vector<int32_t> crop_size{30};
  std::shared_ptr<TensorTransform> centre_out1 = std::make_shared<vision::CenterCrop>(crop_size);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({centre_out1});
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

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CenterCrop op
/// Description: Test CenterCrop op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCropSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCropSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a crop object
  int height = 20;
  int width = 25;
  auto crop = std::make_shared<vision::Crop>(std::vector<int32_t>{0, 0}, std::vector<int32_t>{height, width});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({crop});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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
    EXPECT_EQ(image.Shape()[1], height);
    EXPECT_EQ(image.Shape()[2], width);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CenterCrop op
/// Description: Test CenterCrop op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCropParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCropParamCheck with invalid parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Case 1: Value of coordinates is negative
  // Create objects for the tensor ops
  auto crop1 = std::make_shared<vision::Crop>(std::vector<int32_t>{-1, -1}, std::vector<int32_t>{20});
  auto ds1 = ds->Map({crop1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid coordinates for Crop
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Size of coordinates is not 2
  // Create objects for the tensor ops
  auto crop2 = std::make_shared<vision::Crop>(std::vector<int32_t>{5}, std::vector<int32_t>{10});
  auto ds2 = ds->Map({crop2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid coordinates for Crop
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Value of size is negative
  // Create objects for the tensor ops
  auto crop3 = std::make_shared<vision::Crop>(std::vector<int32_t>{0, 0}, std::vector<int32_t>{-10, -5});
  auto ds3 = ds->Map({crop3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid size for Crop
  EXPECT_EQ(iter3, nullptr);

  // Case 4: Size is neither a single number nor a vector of size 2
  // Create objects for the tensor ops
  auto crop4 = std::make_shared<vision::Crop>(std::vector<int32_t>{0, 0}, std::vector<int32_t>{10, 10, 10});
  auto ds4 = ds->Map({crop4});
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid size for Crop
  EXPECT_EQ(iter4, nullptr);
}













/// Feature: Decode op
/// Description: Test Decode op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDecode.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create Decode object
  vision::Decode decode = vision::Decode(true);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({decode});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: HWC2CHW op
/// Description: Test HWC2CHW basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestHwcToChw) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestHwcToChw.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> channel_swap = std::make_shared<vision::HWC2CHW>();
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({channel_swap});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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
    // Check if the image is in NCHW
    EXPECT_EQ(
      batch_size == image.Shape()[0] && 3 == image.Shape()[1] && 2268 == image.Shape()[2] && 4032 == image.Shape()[3],
      true);
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}









/// Feature: Normalize op
/// Description: Test Normalize op basic usage with default HWC parameter
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNormalizeDefaultHWC) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeDefaultHWC.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto normalize =
    std::make_shared<vision::Normalize>(std::vector<float>{121.0, 115.0, 0.0}, std::vector<float>{70.0, 68.0, 71.0});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({normalize});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Normalize op
/// Description: Test Normalize HWC true and HWC input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNormalizeHWCTrueValid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeHWCTrueValid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto normalize = std::make_shared<vision::Normalize>(std::vector<float>{121.0, 115.0, 0.0},
                                                       std::vector<float>{70.0, 68.0, 71.0}, true);

  // Create a Map operation on ds
  ds = ds->Map({normalize});
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Normalize op
/// Description: Test Normalize op with HWC false and HWC input
/// Expectation: Error is caught for mismatched input image format (HWC)
TEST_F(MindDataTestPipeline, TestNormalizeHWCFalseInvalid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto normalize = std::make_shared<vision::Normalize>(std::vector<float>{121.0, 115.0, 0.0},
                                                       std::vector<float>{70.0, 68.0, 71.0}, false);

  // Create a Map operation on ds
  ds = ds->Map({normalize});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_ERROR(iter->GetNextRow(&row));
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Normalize op
/// Description: Test Normalize op with HWC true and CHW input
/// Expectation: Error is caught for mismatched input image format (CHW)
TEST_F(MindDataTestPipeline, TestNormalizeHWCTrueInvalid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeHWCTrueInvalid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto normalize = std::make_shared<vision::Normalize>(std::vector<float>{121.0, 115.0, 0.0},
                                                       std::vector<float>{70.0, 68.0, 71.0}, true);

  auto HWC2CHW = std::make_shared<vision::HWC2CHW>();

  // Create a Map operation on ds
  ds = ds->Map({HWC2CHW, normalize});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_ERROR(iter->GetNextRow(&row));
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Normalize op
/// Description: Test Normalize op with CHW true and CHW input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNormalizeHWCFalseValid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeHWCFalseValid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto normalize = std::make_shared<vision::Normalize>(std::vector<float>{121.0, 115.0, 0.0},
                                                       std::vector<float>{70.0, 68.0, 71.0}, false);

  auto HWC2CHW = std::make_shared<vision::HWC2CHW>();

  // Create a Map operation on ds
  ds = ds->Map({HWC2CHW, normalize});
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizePad op
/// Description: Test NormalizePad op basic usage with default HWC parameter
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNormalizePadDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadDefault.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto normalizepad = std::make_shared<vision::NormalizePad>(std::vector<float>{121.0, 115.0, 100.0},
                                                             std::vector<float>{70.0, 68.0, 71.0});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({normalizepad});
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
    EXPECT_EQ(image.Shape()[2], 4);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizePad op
/// Description: Test NormalizePad op basic usage with default HWC parameter and float16 type
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNormalizePadDefaultFloat16AndHWCDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadDefaultFloat16AndHWCDefault.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto normalizepad = std::make_shared<vision::NormalizePad>(std::vector<float>{121.0, 115.0, 100.0},
                                                             std::vector<float>{70.0, 68.0, 71.0}, "float16");
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({normalizepad});
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
    EXPECT_EQ(image.Shape()[2], 4);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizePad op
/// Description: Test NormalizePad op with HWC true and HWC input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNormalizePadHWCTrueValid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadHWCTrueValid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto normalizepad = std::make_shared<vision::NormalizePad>(std::vector<float>{121.0, 115.0, 100.0},
                                                             std::vector<float>{70.0, 68.0, 71.0}, "float32", true);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({normalizepad});
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
    EXPECT_EQ(image.Shape()[2], 4);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizePad op
/// Description: Test NormalizePad op with HWC false and HWC input
/// Expectation: Error is caught for mismatched input image format (HWC)
TEST_F(MindDataTestPipeline, TestNormalizePadHWCFalseInvalid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadHWCFalseInvalid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto normalizepad = std::make_shared<vision::NormalizePad>(std::vector<float>{121.0, 115.0, 100.0},
                                                             std::vector<float>{70.0, 68.0, 71.0}, "float16", false);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({normalizepad});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_ERROR(iter->GetNextRow(&row));

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizePad op
/// Description: Test NormalizePad op with HWC true and CHW input
/// Expectation: Error is caught for mismatched input image format (CHW)
TEST_F(MindDataTestPipeline, TestNormalizePadHWCTrueInvalid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadHWCTrueInvalid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto normalizepad = std::make_shared<vision::NormalizePad>(std::vector<float>{121.0, 115.0, 100.0},
                                                             std::vector<float>{70.0, 68.0, 71.0}, "float32", true);
  // Note: No need to check for output after calling API class constructor

  auto HWC2CHW = std::make_shared<vision::HWC2CHW>();

  // Create a Map operation on ds
  ds = ds->Map({HWC2CHW, normalizepad});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_ERROR(iter->GetNextRow(&row));

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizePad op
/// Description: Test NormalizePad op with HWC false and CHW input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestNormalizePadHWCFalseValid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadHWCFalseValid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto normalizepad = std::make_shared<vision::NormalizePad>(std::vector<float>{121.0, 115.0, 100.0},
                                                             std::vector<float>{70.0, 68.0, 71.0}, "float32", false);
  // Note: No need to check for output after calling API class constructor

  auto HWC2CHW = std::make_shared<vision::HWC2CHW>();

  // Create a Map operation on ds
  ds = ds->Map({HWC2CHW, normalizepad});
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
    EXPECT_EQ(image.Shape()[2], 4032);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: NormalizePad op
/// Description: Test NormalizePad op with invalid float64 dtype
/// Expectation: Error is caught for invalid dtype provided
TEST_F(MindDataTestPipeline, TestNormalizePadInvalidDTypeFloat64) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadInvalidDTypeFloat64.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Pass invalid dtype ("float64") to NormalizePad.
  auto normalizepad = std::make_shared<vision::NormalizePad>(std::vector<float>{121.0, 115.0, 100.0},
                                                             std::vector<float>{70.0, 68.0, 71.0}, "float64", false);
  EXPECT_NE(normalizepad, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({normalizepad});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // Catch a nullptr from the iterator object.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: NormalizePad op
/// Description: Test NormalizePad op with invalid int32 dtype
/// Expectation: Error is caught for invalid dtype provided
TEST_F(MindDataTestPipeline, TestNormalizePadInvalidDTypeInt32) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadInvalidDTypeInt32.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Pass invalid dtype ("int32") to NormalizePad.
  auto normalizepad = std::make_shared<vision::NormalizePad>(std::vector<float>{121.0, 115.0, 100.0},
                                                             std::vector<float>{70.0, 68.0, 71.0}, "int32", false);
  EXPECT_NE(normalizepad, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({normalizepad});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // Catch a nullptr from the iterator object.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Pad op
/// Description: Test Pad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPad) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPad.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto pad_op1 =
    std::make_shared<vision::Pad>(std::vector<int32_t>{1, 2, 3, 4}, std::vector<uint8_t>{0}, BorderType::kSymmetric);
  auto pad_op2 =
    std::make_shared<vision::Pad>(std::vector<int32_t>{1}, std::vector<uint8_t>{1, 1, 1}, BorderType::kEdge);
  auto pad_op3 = std::make_shared<vision::Pad>(std::vector<int32_t>{1, 4});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({pad_op1, pad_op2, pad_op3});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: AutoAugment op
/// Description: Test AutoAugment op pipeline
/// Expectation: Create an ImageFolder dataset then do auto augmentation on it with the policy
TEST_F(MindDataTestPipeline, TestAutoAugment) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAutoAugment.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  auto auto_augment_op = vision::AutoAugment(AutoAugmentPolicy::kImageNet, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({auto_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: AutoAugment op
/// Description: Test AutoAugment op with invalid fill_value
/// Expectation: Pipeline iteration failed with wrong argument fill_value
TEST_F(MindDataTestPipeline, TestAutoAugmentInvalidFillValue) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAutoAugmentInvalidFillValue.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto auto_augment_op =
    vision::AutoAugment(AutoAugmentPolicy::kImageNet, InterpolationMode::kNearestNeighbour, {20, 20});

  ds = ds->Map({auto_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: GetImageNumChannels
/// Description: Test GetImageNumChannels with pipeline mode
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestGetImageNumChannelsPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetImageNumChannelsPipeline.";

  std::shared_ptr<Tensor> input_tensor;
  std::vector<int> input_vector = {3, 4, 2, 5, 1, 3, 4, 5, 2, 5, 7, 3};
  ASSERT_OK(Tensor::CreateFromVector(input_vector, TensorShape({2, 2, 3}), &input_tensor));
  auto input_tensor_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  dsize_t channels = 0;
  ASSERT_OK(vision::GetImageNumChannels(input_tensor_ms, &channels));
  int expected = 3;

  ASSERT_EQ(channels, expected);
}

/// Feature: GetImageNumChannels
/// Description: Test GetImageNumChannels with invalid input
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestGetImageNumChannelsInValidInput) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetImageNumChannelsInValidInput.";

  std::shared_ptr<Tensor> input_tensor;
  std::vector<int> input_vector = {3, 4, 2, 5, 1, 3, 4, 5, 2, 5, 7, 3};
  ASSERT_OK(Tensor::CreateFromVector(input_vector, TensorShape({12}), &input_tensor));
  auto input_tensor_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  dsize_t channels = 0;
  ASSERT_FALSE(vision::GetImageNumChannels(input_tensor_ms, &channels));
}

/// Feature: GetImageSize
/// Description: Test GetImageSize with pipeline mode
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestGetImageSizePipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetImageSizePipeline.";

  std::shared_ptr<Tensor> input_tensor;
  std::vector<float> input_vector = {3, 4, 2, 5, 1, 34, 4, 5, 2, 5, 7, 3, 12, 1, 5, 6, 3, 2};
  ASSERT_OK(Tensor::CreateFromVector(input_vector, TensorShape({3, 2, 3}), &input_tensor));
  auto size = std::vector<dsize_t>(2);
  ASSERT_OK(ImageSize(input_tensor, &size));
  std::vector<dsize_t> expected = {3, 2};

  ASSERT_EQ(size, expected);
}

/// Feature: GetImageSize
/// Description: Test GetImageSize with invalid input
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestGetImageSizeInValidInput) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetImageSizeInValidInput.";

  std::shared_ptr<Tensor> input_tensor;
  std::vector<int> input_vector = {3, 4, 2, 5, 1, 3, 4, 5, 2, 5, 7, 3};
  ASSERT_OK(Tensor::CreateFromVector(input_vector, TensorShape({12}), &input_tensor));
  auto size = std::vector<dsize_t>(2);
  ASSERT_FALSE(ImageSize(input_tensor, &size));
}

/// Feature: PadToSize op
/// Description: Test default usage
/// Expectation: Samples processed successfully

/// Feature: PadToSize op
/// Description: Test parameter check
/// Expectation: Error logs are as expected

/// Feature: Solarize
/// Description: Test default usage
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestSolarize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSolarize.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  std::vector<float> threshold = {1.0, 255.0};
  auto solarize_op = vision::Solarize(threshold);

  ds = ds->Map({solarize_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: Solarize
/// Description: Test parameter check
/// Expectation: Error logs are as expected
TEST_F(MindDataTestPipeline, TestSolarizeInvalidFillValue) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSolarizeInvalidFillValue.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  std::vector<float> threshold = {150, 100};
  auto solarize_op = vision::Solarize(threshold);

  ds = ds->Map({solarize_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Erase op
/// Description: Test Erase pipeline
/// Expectation: Create an ImageFolder dataset then do erase on it with the policy
TEST_F(MindDataTestPipeline, TestErase) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestErase.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> erase = std::make_shared<vision::Erase>(10, 10, 10, 10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({erase});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Erase op
/// Description: test Erase with invalid fill_value
/// Expectation: pipeline iteration failed with wrong fill_value
TEST_F(MindDataTestPipeline, TestEraseWideInvalidFillValue) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEraseWideInvalidFillValue.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto erase_wide_op = vision::Erase(10, 10, 10, 10, {20, 20});

  ds = ds->Map({erase_wide_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: AdjustBrightness op
/// Description: Test AdjustBrightness C implementation Pipeline
/// Expectation: Output is equal to the expected output

/// Feature: AdjustBrightness op
/// Description: Test improper parameters for AdjustBrightness C implementation
/// Expectation: Throw ValueError exception and TypeError exception

/// Feature: TrivialAugmentWide op
/// Description: test TrivialAugmentWide pipeline
/// Expectation: create an ImageFolder dataset then do auto augmentation on it with the policy
TEST_F(MindDataTestPipeline, TestTrivialAugmentWide) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTrivialAugmentWide.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  auto trivial_augment_wide_op = vision::TrivialAugmentWide(31, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({trivial_augment_wide_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: TrivialAugmentWide op
/// Description: test TrivialAugmentWide with invalid fill_value
/// Expectation: pipeline iteration failed with wrong argument fill_value
TEST_F(MindDataTestPipeline, TestTrivialAugmentWideInvalidFillValue) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTrivialAugmentWideInvalidFillValue.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto trivial_augment_wide_op = vision::TrivialAugmentWide(31, InterpolationMode::kNearestNeighbour, {20, 20});

  ds = ds->Map({trivial_augment_wide_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: AdjustSharpness op
/// Description: Test AdjustSharpness pipeline
/// Expectation: Create an ImageFolder dataset then do auto AjustSharpness on it

/// Feature: AdjustSharpness op
/// Description: Test parameter check
/// Expectation: Error logs are as expected

/// Feature: AdjustSaturation op
/// Description: Test AdjustSaturation pipeline
/// Expectation: Create an ImageFolder dataset then do auto AjustSaturation on it

/// Feature: AdjustSaturation op
/// Description: Test parameter check
/// Expectation: Error logs are as expected

/// Feature: Posterize op
/// Description: Test posterize pipeline with non-default parameters
/// Expectation: Create an ImageFolder dataset then do posterize on it
TEST_F(MindDataTestPipeline, TestPosterizeSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPosterizeSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> posterize = std::make_shared<vision::Posterize>(0);
  // Create a Map operation on ds
  ds = ds->Map({posterize});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Posterize op
/// Description: Test posterize parameter check
/// Expectation: Error logs are as expected
TEST_F(MindDataTestPipeline, TestPosterizeParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPosterizeParamCheck.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Negative alpha
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> posterize = std::make_shared<vision::Posterize>(-1);
  auto ds1 = ds->Map({posterize});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid value of Posterize
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Over range parameter
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> posterize1 = std::make_shared<vision::Posterize>(9);
  auto ds2 = ds->Map({posterize1});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid value of Posterize
  EXPECT_EQ(iter2, nullptr);
}

/// Feature: AdjustHue op
/// Description: Test function of operation when hue_factor is 0.2
/// Expectation: Create an ImageFolder dataset then do auto AjustHue on it

/// Feature: AdjustHue op
/// Description: Test improper parameters for AdjustHue C implementation
/// Expectation: Throw ValueError exception and TypeError exception

/// Feature: AdjustContrast op
/// Description: Test AdjustContrast C implementation Pipeline
/// Expectation: Output is equal to the expected output

/// Feature: AdjustContrast op
/// Description: Test improper parameters for AdjustContrast C implementation
/// Expectation: Throw ValueError exception

/// Feature: Perspective
/// Description: Test Perspective pipeline
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestPerspective) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPerspective.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::vector<std::vector<int32_t>> src = {{0, 200}, {400, 200}, {400, 0}, {0, 0}};
  std::vector<std::vector<int32_t>> dst = {{0, 180}, {400, 180}, {400, 0}, {0, 0}};

  auto perspective_op = vision::Perspective(src, dst, InterpolationMode::kLinear);

  // Create a Map operation on ds
  ds = ds->Map({perspective_op});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
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
  EXPECT_EQ(i, 20);

  iter->Stop();
}

/// Feature: Perspective
/// Description: test Perspective with invalid input
/// Expectation: the returned result is as expected
TEST_F(MindDataTestPipeline, TestPerspectiveParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPosterizeParamCheck.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: invalid start point lists
  // Create objects for the tensor ops
  std::vector<std::vector<int32_t>> src = {{0, 200}, {400, 200}, {400, 0}, {0}};
  std::vector<std::vector<int32_t>> dst = {{0, 200}, {400, 200}, {400, 0}, {0, 0}};

  auto perspective = vision::Perspective(src, dst, InterpolationMode::kLinear);
  ds = ds->Map({perspective});
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid value of Perspective
  EXPECT_EQ(iter, nullptr);

  // Case 2: invalid end point lists
  // Create objects for the tensor ops
  std::vector<std::vector<int32_t>> src1 = {{0, 200}, {400, 200}, {400, 0}, {0, 0}};
  std::vector<std::vector<int32_t>> dst1 = {{0, 200}, {400, 180}, {400}, {0, 0}};
  auto perspective1 = vision::Perspective(src1, dst1, InterpolationMode::kLinear);
  ds = ds->Map({perspective1});
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  // Expect failure: invalid value of Perspective
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: EncodeJpeg
/// Description: Test EncodeJpeg by encoding the image as JPEG data according to the quality
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestEncodeJpegNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesEncodeJpegNormal.";
  mindspore::MSTensor output;
  std::string folder_path = "./data/dataset/";
  std::string filename;
  const UINT8 *data;

  filename = folder_path + "apple.jpg";
  cv::Mat image_bgr = cv::imread(filename, cv::ImreadModes::IMREAD_UNCHANGED);
  cv::Mat image;
  cv::cvtColor(image_bgr, image, cv::COLOR_BGRA2RGB);

  TensorShape img_tensor_shape = TensorShape({image.size[0], image.size[1], image.channels()});
  DataType pixel_type = DataType(DataType::DE_UINT8);

  std::shared_ptr<Tensor> input;
  Tensor::CreateFromMemory(img_tensor_shape, pixel_type, image.data, &input);
  auto input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));

  ASSERT_OK(mindspore::dataset::vision::EncodeJpeg(input_ms_tensor, &output));
  data = (const UINT8 *)(output.Data().get());
  EXPECT_EQ(data[0], 255);
  EXPECT_EQ(data[1], 216);
  EXPECT_EQ(data[2], 255);

  int quality;
  for (quality = 20; quality <= 100; quality += 40) {
    ASSERT_OK(mindspore::dataset::vision::EncodeJpeg(input_ms_tensor, &output, quality));
    data = (const UINT8 *)(output.Data().get());
    EXPECT_EQ(data[1], 216);
  }
}

/// Feature: EncodeJpeg
/// Description: Test EncodeJpeg with invalid parameter
/// Expectation: Error is caught when the parameter is invalid
TEST_F(MindDataTestPipeline, TestEncodeJpegException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesEncodeJpegException.";
  mindspore::MSTensor output;
  std::string folder_path = "./data/dataset/";
  std::string filename;

  filename = folder_path + "apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_UNCHANGED);

  TensorShape img_tensor_shape = TensorShape({image.size[0], image.size[1], image.channels()});
  DataType pixel_type = DataType(DataType::DE_UINT8);

  std::shared_ptr<Tensor> input;
  Tensor::CreateFromMemory(img_tensor_shape, pixel_type, image.data, &input);
  auto input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));

  // Test with an invalid integer for the quality
  ASSERT_ERROR(mindspore::dataset::vision::EncodeJpeg(input_ms_tensor, &output, 0));
  ASSERT_ERROR(mindspore::dataset::vision::EncodeJpeg(input_ms_tensor, &output, 101));

  // Test with an invalid image containing float32 elements
  std::shared_ptr<Tensor> float32_de_tensor;
  Tensor::CreateEmpty(TensorShape({5, 4, 3}), DataType(DataType::DE_FLOAT32), &float32_de_tensor);
  input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(float32_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::EncodeJpeg(input_ms_tensor, &output));

  // Test with an invalid image with only one dimension
  input->Reshape(TensorShape({image.size[0] * image.size[1] * image.channels()}));
  input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  ASSERT_ERROR(mindspore::dataset::vision::EncodeJpeg(input_ms_tensor, &output));

  // Test with an invalid image with four dimensions
  input->Reshape(TensorShape({image.size[0] / 2, image.size[1], image.channels(), 2}));
  input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  ASSERT_ERROR(mindspore::dataset::vision::EncodeJpeg(input_ms_tensor, &output));

  // Test with an invalid image with two channels
  input->Reshape(TensorShape({image.size[0] * image.channels() / 2, image.size[1], 2}));
  input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  ASSERT_ERROR(mindspore::dataset::vision::EncodeJpeg(input_ms_tensor, &output));
}

/// Feature: EncodePng
/// Description: Test EncodePng by encoding the image as PNG data according to the compression_level
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestEncodePngNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesEncodePngNormal.";
  mindspore::MSTensor output;
  std::string folder_path = "./data/dataset/";
  std::string filename;
  const UINT8 *data;

  filename = folder_path + "apple.jpg";
  cv::Mat image_bgr = cv::imread(filename, cv::ImreadModes::IMREAD_UNCHANGED);
  cv::Mat image;
  cv::cvtColor(image_bgr, image, cv::COLOR_BGRA2RGB);

  TensorShape tensor_shape = TensorShape({image.size[0], image.size[1], image.channels()});
  DataType pixel_type = DataType(DataType::DE_UINT8);

  std::shared_ptr<Tensor> input;
  Tensor::CreateFromMemory(tensor_shape, pixel_type, image.data, &input);
  auto input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));

  ASSERT_OK(mindspore::dataset::vision::EncodePng(input_ms_tensor, &output));
  data = (const UINT8 *)(output.Data().get());
  EXPECT_EQ(data[0], 137);
  EXPECT_EQ(data[1], 80);
  EXPECT_EQ(data[2], 78);
  EXPECT_EQ(data[3], 71);

  int compression_level;
  for (compression_level = 0; compression_level <= 9; compression_level++) {
    ASSERT_OK(mindspore::dataset::vision::EncodePng(input_ms_tensor, &output, compression_level));
    data = (const UINT8 *)(output.Data().get());
    EXPECT_EQ(data[1], 80);
  }
}

/// Feature: EncodePng
/// Description: Test EncodePng with invalid parameter
/// Expectation: Error is caught when the parameter is invalid
TEST_F(MindDataTestPipeline, TestEncodePngException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesEncodePngException.";
  mindspore::MSTensor output;
  std::string folder_path = "./data/dataset/";
  std::string filename;

  filename = folder_path + "apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_UNCHANGED);

  TensorShape tensor_shape = TensorShape({image.size[0], image.size[1], image.channels()});
  DataType pixel_type = DataType(DataType::DE_UINT8);

  std::shared_ptr<Tensor> input;
  Tensor::CreateFromMemory(tensor_shape, pixel_type, image.data, &input);
  auto input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));

  // Test with an invalid compression_level integer
  ASSERT_ERROR(mindspore::dataset::vision::EncodePng(input_ms_tensor, &output, -1));
  ASSERT_ERROR(mindspore::dataset::vision::EncodePng(input_ms_tensor, &output, 10));

  // Test with an invalid image with the type of float
  std::shared_ptr<Tensor> float32_de_tensor;
  Tensor::CreateEmpty(TensorShape({5, 4, 3}), DataType(DataType::DE_FLOAT32), &float32_de_tensor);
  input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(float32_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::EncodePng(input_ms_tensor, &output));

  // Test with an invalid image with only one dimension
  input->Reshape(TensorShape({image.size[0] * image.size[1] * image.channels()}));
  input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  ASSERT_ERROR(mindspore::dataset::vision::EncodePng(input_ms_tensor, &output));

  // Test with an invalid image with four dimensions
  input->Reshape(TensorShape({image.size[0] / 2, image.size[1], image.channels(), 2}));
  input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  ASSERT_ERROR(mindspore::dataset::vision::EncodePng(input_ms_tensor, &output));

  // Test with an invalid image with two channels
  input->Reshape(TensorShape({image.size[0] * image.channels() / 2, image.size[1], 2}));
  input_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  ASSERT_ERROR(mindspore::dataset::vision::EncodePng(input_ms_tensor, &output));
}
