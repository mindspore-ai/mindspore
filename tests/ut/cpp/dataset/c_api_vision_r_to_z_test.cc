/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API R to Z TensorTransform Operations (in alphabetical order)

/// Feature: RandomLighting op
/// Description: Test RandomLighting Op on pipeline when alpha=0.1
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestRandomLightingPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomLightingPipeline.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  auto image = row["image"];

  // Create objects for the tensor ops
  auto randomlighting = std::make_shared<mindspore::dataset::vision::RandomLighting>(0.1);
  // Note: No need to check for output after calling API class constructor

  // Convert to the same type
  auto type_cast = std::make_shared<transforms::TypeCast>(mindspore::DataType::kNumberTypeUInt8);
  // Note: No need to check for output after calling API class constructor

  ds = ds->Map({randomlighting, type_cast}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_NE(iter1, nullptr);

  // Iterate the dataset and get each row1
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  ASSERT_OK(iter1->GetNextRow(&row1));

  auto image1 = row1["image"];

  // Manually terminate the pipeline
  iter1->Stop();
}

/// Feature: RandomLighting op
/// Description: Test param check for RandomLighting Op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline
///     returns nullptr when params are invalid
TEST_F(MindDataTestPipeline, TestRandomLightingParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomLightingParamCheck.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: negative alpha
  // Create objects for the tensor ops
  auto random_lighting_op = std::make_shared<mindspore::dataset::vision::RandomLighting>(-0.1);
  auto ds2 = ds->Map({random_lighting_op});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid value of alpha
  EXPECT_EQ(iter2, nullptr);
}

/// Feature: Rescale op
/// Description: Test Rescale op with 1.0 rescale factor and 0.0 shift factor
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRescaleSucess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleSucess1.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  auto image = row["image"];

  // Create objects for the tensor ops
  auto rescale = std::make_shared<mindspore::dataset::vision::Rescale>(1.0, 0.0);
  // Note: No need to check for output after calling API class constructor

  // Convert to the same type
  auto type_cast = std::make_shared<transforms::TypeCast>(mindspore::DataType::kNumberTypeUInt8);
  // Note: No need to check for output after calling API class constructor

  ds = ds->Map({rescale, type_cast}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_NE(iter1, nullptr);

  // Iterate the dataset and get each row1
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  ASSERT_OK(iter1->GetNextRow(&row1));

  auto image1 = row1["image"];

  EXPECT_MSTENSOR_EQ(image, image1);

  // Manually terminate the pipeline
  iter1->Stop();
}

/// Feature: Rescale op
/// Description: Test Rescale op with 1.0 / 255 rescale factor and 1.0 shift factor
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRescaleSucess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleSucess2 with different params.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto rescale = std::make_shared<mindspore::dataset::vision::Rescale>(1.0 / 255, 1.0);
  // Note: No need to check for output after calling API class constructor

  ds = ds->Map({rescale}, {"image"});
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Resize op
/// Description: Test Resize op with single integer input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestResize1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResize1 with single integer input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 6));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 4;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create resize object with single integer input
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{30});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({resize_op});
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

  EXPECT_EQ(i, 24);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ResizeWithBBox op
/// Description: Test ResizeWithBBox op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestResizeWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizeWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize_with_bbox_op = std::make_shared<vision::ResizeWithBBox>(std::vector<int32_t>{30});
  auto resize_with_bbox_op1 = std::make_shared<vision::ResizeWithBBox>(std::vector<int32_t>{30, 30});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({resize_with_bbox_op, resize_with_bbox_op1}, {"image", "bbox"}, {"image", "bbox"});
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

/// Feature: RGB2GRAY op
/// Description: Test RGB2GRAY op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRGB2GRAYSucess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRGB2GRAYSucess.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto convert = std::make_shared<mindspore::dataset::vision::RGB2GRAY>();

  ds = ds->Map({convert});
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Rotate op
/// Description: Test Rotate op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRotateParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRotateParamCheck with invalid parameters.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: Size of center is not 2
  // Create objects for the tensor ops
  auto rotate1 =
    std::make_shared<vision::Rotate>(90.0, InterpolationMode::kNearestNeighbour, false, std::vector<float>{0.});
  auto ds2 = ds->Map({rotate1});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid center for Rotate
  EXPECT_EQ(iter2, nullptr);

  // Case 2: Size of fill_value is not 1 or 3
  // Create objects for the tensor ops
  auto rotate2 = std::make_shared<vision::Rotate>(-30, InterpolationMode::kNearestNeighbour, false,
                                                  std::vector<float>{1.0, 1.0}, std::vector<uint8_t>{2, 2});
  auto ds3 = ds->Map({rotate2});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid fill_value for Rotate
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: Rotate op
/// Description: Test Rotate op by passing it to a Map op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRotatePass) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRotatePass.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize = std::make_shared<vision::Resize>(std::vector<int32_t>{50, 25});

  auto rotate = std::make_shared<vision::Rotate>(90, InterpolationMode::kLinear, true, std::vector<float>{-1, -1},
                                                 std::vector<uint8_t>{255, 255, 255});

  // Resize the image to 50 * 25
  ds = ds->Map({resize});
  EXPECT_NE(ds, nullptr);

  // Rotate the image 90 degrees
  ds = ds->Map({rotate});
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
    // After rotation with expanding, the image size comes to 25 * 50
    EXPECT_EQ(image.Shape()[1], 25);
    EXPECT_EQ(image.Shape()[2], 50);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Rotate op
/// Description: Test Rotate op by processing tensor with dim more than 3
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRotateBatch) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRotateBatch.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds, choose batch size 3 to test high dimension input
  int32_t batch_size = 3;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto rotate = std::make_shared<vision::Rotate>(90, InterpolationMode::kLinear, false, std::vector<float>{-1, -1},
                                                 std::vector<uint8_t>{255, 255, 255});

  // Rotate the image 90 degrees
  ds = ds->Map({rotate});
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

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RGB2BGR op
/// Description: Test RGB2BGR op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRGB2BGR) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRGB2BGR.";
  // create two imagenet dataset
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds2, nullptr);

  auto rgb2bgr_op = vision::RGB2BGR();

  ds1 = ds1->Map({rgb2bgr_op});
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

/// Feature: RandomCrop
/// Description: Use batched dataset as video inputs
/// Expectation: The log will print correct shape
TEST_F(MindDataTestPipeline, TestRandomCropHighDimensions) {
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

  // Create random crop object with square crop
  const std::vector<int32_t> crop_size{30};
  std::shared_ptr<TensorTransform> centre_out1 = std::make_shared<vision::RandomCrop>(crop_size);
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

/// Feature: RandomEqualize op
/// Description: Test RandomEqualize op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomEqualize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomEqualize.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_equalize_op = vision::RandomEqualize(0.5);

  ds = ds->Map({random_equalize_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: RandomEqualize op
/// Description: Test RandomEqualize op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomEqualizeInvalidProb) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomEqualizeInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_equalize_op = vision::RandomEqualize(1.5);

  ds = ds->Map({random_equalize_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomInvert op
/// Description: Test RandomInvert op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomInvert) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomInvert.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_invert_op = vision::RandomInvert(0.5);

  ds = ds->Map({random_invert_op});
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

/// Feature: RandomInvert op
/// Description: Test RandomInvert op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomInvertInvalidProb) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomInvertInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_invert_op = vision::RandomInvert(1.5);

  ds = ds->Map({random_invert_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomAutoContrast) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAutoContrast.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_auto_contrast_op = vision::RandomAutoContrast(1.0, {0, 255}, 0.5);

  ds = ds->Map({random_auto_contrast_op});
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

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAutoContrastInvalidProb) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAutoContrastInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_auto_contrast_op = vision::RandomAutoContrast(0.0, {}, 1.5);

  ds = ds->Map({random_auto_contrast_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid cutoff
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAutoContrastInvalidCutoff) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAutoContrastInvalidCutoff.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_auto_contrast_op = vision::RandomAutoContrast(-2.0, {}, 0.5);

  ds = ds->Map({random_auto_contrast_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid ignore
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAutoContrastInvalidIgnore) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAutoContrastInvalidCutoff.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_auto_contrast_op = vision::RandomAutoContrast(1.0, {10, 256}, 0.5);

  ds = ds->Map({random_auto_contrast_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomAdjustSharpness) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAdjustSharpness.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(2.0, 0.5);

  ds = ds->Map({random_adjust_sharpness_op});
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

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAdjustSharpnessInvalidProb) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAdjustSharpnessInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(2.0, 1.5);

  ds = ds->Map({random_adjust_sharpness_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op with invalid degree
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAdjustSharpnessInvalidDegree) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAdjustSharpnessInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(-2.0, 0.3);

  ds = ds->Map({random_adjust_sharpness_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ToTensor op
/// Description: Test ToTensor op with default float32 type
/// Expectation: Tensor type is changed to float32 and all rows iterated correctly
TEST_F(MindDataTestPipeline, TestToTensorOpDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToTensorOpDefault.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto to_tensor_op = vision::ToTensor();
  ds = ds->Map({to_tensor_op}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image type: " << image.DataType();
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  iter->Stop();
}

/// Feature: ToTensor op
/// Description: Test ToTensor op with float64 type passed as string
/// Expectation: Tensor type is changed to float64 and all rows iterated correctly
TEST_F(MindDataTestPipeline, TestToTensorOpFloat64String) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToTensorOpFloat64String.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto to_tensor_op = vision::ToTensor("float64");
  ds = ds->Map({to_tensor_op}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image type: " << image.DataType();
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  iter->Stop();
}

/// Feature: ToTensor op
/// Description: Test ToTensor op with float64 type passed as DataType
/// Expectation: Tensor type is changed to float64 and all rows iterated correctly
TEST_F(MindDataTestPipeline, TestToTensorOpFloat64DataType) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToTensorOpFloat64DataType.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto to_tensor_op = vision::ToTensor(mindspore::DataType::kNumberTypeFloat64);
  ds = ds->Map({to_tensor_op}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image type: " << image.DataType();
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  iter->Stop();
}

/// Feature: ToTensor op
/// Description: Test ToTensor op with float64 type and invalid uint32 type input data
/// Expectation: Error is caught as given invalid input data type
TEST_F(MindDataTestPipeline, TestToTensorOpInvalidInput) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToTensorOpInvalidInput.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto type_cast = transforms::TypeCast(mindspore::DataType::kNumberTypeUInt32);
  auto to_tensor_op = vision::ToTensor("float64");
  ds = ds->Map({type_cast, to_tensor_op}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_ERROR(iter->GetNextRow(&row));
}

/// Feature: ResizedCrop op
/// Description: Test ResizedCrop pipeline
/// Expectation: Input is processed as expected and all rows iterated correctly
TEST_F(MindDataTestPipeline, TestResizedCrop) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizedCrop.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  unsigned int left = 256;
  unsigned int top = 256;
  unsigned int height = 256;
  unsigned int width = 256;
  std::vector<int32_t> size{128, 128};
  auto resized_crop_op = vision::ResizedCrop(top, left, height, width, size);

  ds = ds->Map({resized_crop_op});
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

/// Feature: ResizedCrop op
/// Description: Test ResizedCrop with invalid fill_value
/// Expectation: Pipeline iteration failed with wrong argument fill_value
TEST_F(MindDataTestPipeline, TestResizedCropParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizedCropParamCheck.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Value of top/left/height/width out of boundary
  // Create objects for the tensor ops
  auto resized_crop_op1 = vision::ResizedCrop(-1, -1, -1, -1, {128, 128});
  auto ds1 = ds->Map({resized_crop_op1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid coordinates for Crop
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Value of size is negative
  // Create objects for the tensor ops
  auto resized_crop_op2 = vision::ResizedCrop(256, 256, 256, 256, {-128, -128});
  auto ds2 = ds->Map({resized_crop_op2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid coordinates for Crop
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Size is neither a single number nor a vector of size 2
  // Create objects for the tensor ops
  auto resized_crop_op3 = vision::ResizedCrop(256, 256, 256, 256, {128, 128, 128});
  auto ds3 = ds->Map({resized_crop_op3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid size for Crop
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment pipeline
/// Expectation: create an ImageFolder dataset then do rand augmentation on it
TEST_F(MindDataTestPipeline, TestRandAugment) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugment.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 4, 5, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
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
  EXPECT_EQ(i, 5);

  iter->Stop();
}

/// Feature: RandAugment
/// Description: test RandAugment with invalid fill_value
/// Expectation: pipeline iteration failed with wrong argument fill_value
TEST_F(MindDataTestPipeline, TestRandAugmentInvalidFillValue) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentInvalidFillValue.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 4, 5, InterpolationMode::kNearestNeighbour, {20, 20});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment with invalid num_ops
/// Expectation: pipeline iteration failed with wrong argument num_ops
TEST_F(MindDataTestPipeline, TestRandAugmentInvalidNumOps) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentInvalidNumOps.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(-1, 4, 5);

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment with num_ops equal to 0
/// Expectation: pipeline iteration success with num_ops equal to 0
TEST_F(MindDataTestPipeline, TestRandAugmentNumOpsZero) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentNumOpsZero.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(0, 4, 5, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
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

/// Feature: RandAugment
/// Description: test RandAugment with invalid magnitude
/// Expectation: pipeline iteration failed with wrong argument magnitude
TEST_F(MindDataTestPipeline, TestRandAugmentInvalidMagnitude) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentInvalidMagnitude.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(2, -1, 2, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment with magnitude equal to 0
/// Expectation: pipeline iteration success with magnitude equal to 0
TEST_F(MindDataTestPipeline, TestRandAugmentMagnitudeZero) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentMagnitudeZero.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 0, 2, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
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
  EXPECT_EQ(i, 5);

  iter->Stop();
}

/// Feature: RandAugment
/// Description: test RandAugment with invalid num_magnitude_bins
/// Expectation: pipeline iteration failed with wrong argument num_magnitude_bins
TEST_F(MindDataTestPipeline, TestRandAugmentInvalidNumMagBins) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentInvalidNumMagBins.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 2, 1, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment with num_magnitude_bins equal to 2
/// Expectation: pipeline iteration success with num_magnitude_bins equal to 2
TEST_F(MindDataTestPipeline, TestRandAugmentNumMagBins) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentNumMagBins.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 0, 2, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
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
  EXPECT_EQ(i, 5);

  iter->Stop();
}

/// Feature: RandAugment
/// Description: test RandAugment with magnitude greater than num_magnitude_bins
/// Expectation: pipeline iteration failed with invalid magnitude value
TEST_F(MindDataTestPipeline, TestRandAugmentMagGreNMBError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentMagGreNMBError.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 5, 4);
  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ReadFile
/// Description: Test ReadFile with the an example file
/// Expectation: Output is equal to the expected data
TEST_F(MindDataTestPipeline, TestReadFileNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadFileNormal.";
  mindspore::MSTensor output;
  const UINT8 *data;
  ASSERT_OK(mindspore::dataset::vision::ReadFile("./data/dataset/apple.jpg", &output));
  EXPECT_EQ(output.Shape()[0], 159109);
  EXPECT_EQ(output.DataType(), mindspore::DataType::kNumberTypeUInt8);
  data = (const UINT8 *)(output.Data().get());
  EXPECT_EQ(data[0], 255);
  EXPECT_EQ(data[1], 216);
  EXPECT_EQ(data[2], 255);
  EXPECT_EQ(data[50000], 0);
  EXPECT_EQ(data[100000], 132);
  EXPECT_EQ(data[150000], 64);
  EXPECT_EQ(data[159106], 63);
  EXPECT_EQ(data[159107], 255);
  EXPECT_EQ(data[159108], 217);
}

/// Feature: ReadFile
/// Description: Test ReadFile with invalid filename
/// Expectation: Error is caught when the filename is invalid
TEST_F(MindDataTestPipeline, TestReadFileException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadFileException.";
  mindspore::MSTensor output;

  // Test with a not exist filename
  ASSERT_ERROR(mindspore::dataset::vision::ReadFile("this_file_is_not_exist", &output));

  // Test with a directory name
  ASSERT_ERROR(mindspore::dataset::vision::ReadFile("./data/dataset/", &output));
}

/// Feature: ReadImage
/// Description: Test ReadImage with JPEG, PNG, BMP, TIFF file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadImageNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadImage.";
  mindspore::MSTensor output;
  std::string folder_path = "./data/dataset/testFormats/";
  std::string filename;
  const UINT8 *data;

  filename = folder_path + "apple.jpg";
  ASSERT_OK(mindspore::dataset::vision::ReadImage(filename, &output));
  EXPECT_EQ(output.Shape()[0], 226);
  EXPECT_EQ(output.Shape()[1], 403);
  EXPECT_EQ(output.Shape()[2], 3);
  data = (const UINT8 *)(output.Data().get());
  EXPECT_EQ(data[0], 221);
  EXPECT_EQ(data[1], 221);
  EXPECT_EQ(data[2], 221);
  EXPECT_EQ(data[100 * 403 * 3 + 200 * 3 + 0], 195);
  EXPECT_EQ(data[100 * 403 * 3 + 200 * 3 + 1], 60);
  EXPECT_EQ(data[100 * 403 * 3 + 200 * 3 + 2], 31);
  EXPECT_EQ(data[225 * 403 * 3 + 402 * 3 + 0], 181);
  EXPECT_EQ(data[225 * 403 * 3 + 402 * 3 + 1], 181);
  EXPECT_EQ(data[225 * 403 * 3 + 402 * 3 + 2], 173);
  ASSERT_OK(mindspore::dataset::vision::ReadImage(filename, &output, ImageReadMode::kUNCHANGED));
  EXPECT_EQ(output.Shape()[0], 226);
  EXPECT_EQ(output.Shape()[1], 403);
  EXPECT_EQ(output.Shape()[2], 3);
  ASSERT_OK(mindspore::dataset::vision::ReadImage(filename, &output, ImageReadMode::kGRAYSCALE));
  EXPECT_EQ(output.Shape()[0], 226);
  EXPECT_EQ(output.Shape()[1], 403);
  EXPECT_EQ(output.Shape()[2], 1);
  ASSERT_OK(mindspore::dataset::vision::ReadImage(filename, &output, ImageReadMode::kCOLOR));
  EXPECT_EQ(output.Shape()[0], 226);
  EXPECT_EQ(output.Shape()[1], 403);
  EXPECT_EQ(output.Shape()[2], 3);

  filename = folder_path + "apple.png";
  ASSERT_OK(mindspore::dataset::vision::ReadImage(filename, &output));
  EXPECT_EQ(output.Shape()[0], 226);
  EXPECT_EQ(output.Shape()[1], 403);
  EXPECT_EQ(output.Shape()[2], 3);

  filename = folder_path + "apple_4_channels.png";
  ASSERT_OK(mindspore::dataset::vision::ReadImage(filename, &output));
  EXPECT_EQ(output.Shape()[0], 226);
  EXPECT_EQ(output.Shape()[1], 403);
  EXPECT_EQ(output.Shape()[2], 3);

  filename = folder_path + "apple.bmp";
  ASSERT_OK(mindspore::dataset::vision::ReadImage(filename, &output));
  EXPECT_EQ(output.Shape()[0], 226);
  EXPECT_EQ(output.Shape()[1], 403);
  EXPECT_EQ(output.Shape()[2], 3);

  filename = folder_path + "apple.tiff";
  ASSERT_OK(mindspore::dataset::vision::ReadImage(filename, &output));
  EXPECT_EQ(output.Shape()[0], 226);
  EXPECT_EQ(output.Shape()[1], 403);
  EXPECT_EQ(output.Shape()[2], 3);
}

/// Feature: ReadImage
/// Description: Test ReadImage with invalid filename or not supported image file
/// Expectation: Error is caught when the filename is invalid or it is a not supported image file
TEST_F(MindDataTestPipeline, TestReadImageException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadImageException.";
  std::string folder_path = "./data/dataset/testFormats/";
  mindspore::MSTensor output;

  // Test with a not exist filename
  ASSERT_ERROR(mindspore::dataset::vision::ReadImage("this_file_is_not_exist", &output));

  // Test with a directory name
  ASSERT_ERROR(mindspore::dataset::vision::ReadImage("./data/dataset/", &output));

  // Test with a not supported gif file
  ASSERT_ERROR(mindspore::dataset::vision::ReadImage(folder_path + "apple.gif", &output));
}

/// Feature: WriteFile
/// Description: Test WriteFile by writing the data into a file using binary mode
/// Expectation: The file should be writeen and removed successfully
TEST_F(MindDataTestPipeline, TestWriteFileNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesWriteFileNormal.";
  std::string folder_path = "./data/dataset/";
  std::string filename_1, filename_2;
  filename_1 = folder_path + "apple.jpg";
  filename_2 = filename_1 + ".test_write_file";

  std::shared_ptr<Tensor> de_tensor_1, de_tensor_2;
  Tensor::CreateFromFile(filename_1, &de_tensor_1);
  auto data_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor_1));

  ASSERT_OK(mindspore::dataset::vision::WriteFile(filename_2, data_tensor));
  Tensor::CreateFromFile(filename_2, &de_tensor_2);
  EXPECT_EQ(de_tensor_1->shape(), de_tensor_2->shape());
  remove(filename_2.c_str());
}

/// Feature: WriteFile
/// Description: Test WriFile with invalid parameter
/// Expectation: Error is caught when the parameter is invalid
TEST_F(MindDataTestPipeline, TestWriteFileException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesWriteFileException.";
  std::string folder_path = "./data/dataset/";
  std::string filename_1, filename_2;
  filename_1 = folder_path + "apple.jpg";
  filename_2 = filename_1 + ".test_write_file";

  std::shared_ptr<Tensor> de_tensor_1;
  Tensor::CreateFromFile(filename_1, &de_tensor_1);
  auto data_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor_1));

  // Test with a directory name
  ASSERT_ERROR(mindspore::dataset::vision::WriteFile(folder_path, data_tensor));

  // Test with an invalid filename
  ASSERT_ERROR(mindspore::dataset::vision::WriteFile("/dev/cdrom/0", data_tensor));

  // Test with invalid float elements
  std::shared_ptr<Tensor> input;
  std::vector<float> float_vector = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
  Tensor::CreateFromVector(float_vector, TensorShape({12}), &input);
  data_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  ASSERT_ERROR(mindspore::dataset::vision::WriteFile(filename_2, data_tensor));
}

/// Feature: WriteJpeg
/// Description: Test WriteJpeg by writing the image into a JPEG file
/// Expectation: The file should be written and removed
TEST_F(MindDataTestPipeline, TestWriteJpegNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesWriteJpegNormal.";
  std::string folder_path = "./data/dataset/testFormats/";
  std::string filename_1;
  std::string filename_2;
  cv::Mat image_1;
  cv::Mat image_2;

  filename_1 = folder_path + "apple.jpg";
  filename_2 = filename_1 + ".test_write_jpeg.jpg";

  cv::Mat image_bgr = cv::imread(filename_1, cv::ImreadModes::IMREAD_UNCHANGED);
  cv::cvtColor(image_bgr, image_1, cv::COLOR_BGRA2RGB);

  TensorShape img_tensor_shape = TensorShape({image_1.size[0], image_1.size[1], image_1.channels()});
  DataType pixel_type = DataType(DataType::DE_UINT8);

  std::shared_ptr<Tensor> image_de_tensor;
  Tensor::CreateFromMemory(img_tensor_shape, pixel_type, image_1.data, &image_de_tensor);
  auto image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));

  int quality;
  for (quality = 20; quality <= 100; quality += 40) {
    ASSERT_OK(mindspore::dataset::vision::WriteJpeg(filename_2, image_ms_tensor, quality));
    image_2 = cv::imread(filename_1, cv::ImreadModes::IMREAD_UNCHANGED);
    remove(filename_2.c_str());
    EXPECT_EQ(image_1.total(), image_2.total());
  }
}

/// Feature: WriteJpeg
/// Description: Test WriteJpeg with invalid parameter
/// Expectation: Error is caught when the parameter is invalid
TEST_F(MindDataTestPipeline, TestWriteJpegException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesWriteJpegException.";
  std::string folder_path = "./data/dataset/testFormats/";
  std::string filename_1;
  std::string filename_2;
  cv::Mat image_1;

  filename_1 = folder_path + "apple.jpg";
  filename_2 = filename_1 + ".test_write_jpeg.jpg";
  image_1 = cv::imread(filename_1, cv::ImreadModes::IMREAD_UNCHANGED);

  TensorShape img_tensor_shape = TensorShape({image_1.size[0], image_1.size[1], image_1.channels()});
  DataType pixel_type = DataType(DataType::DE_UINT8);

  std::shared_ptr<Tensor> image_de_tensor;
  Tensor::CreateFromMemory(img_tensor_shape, pixel_type, image_1.data, &image_de_tensor);
  auto image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));

  // Test with invalid quality 0, 101
  ASSERT_ERROR(mindspore::dataset::vision::WriteJpeg(filename_2, image_ms_tensor, 0));
  ASSERT_ERROR(mindspore::dataset::vision::WriteJpeg(filename_2, image_ms_tensor, 101));

  // Test with an invalid filename
  ASSERT_ERROR(mindspore::dataset::vision::WriteJpeg("/dev/cdrom/0", image_ms_tensor));

  // Test with a directory name
  ASSERT_ERROR(mindspore::dataset::vision::WriteJpeg("./data/dataset/", image_ms_tensor));

  // Test with an invalid image containing float elements
  std::shared_ptr<Tensor> float32_cde_tensor;
  Tensor::CreateEmpty(TensorShape({5, 4, 3}), DataType(DataType::DE_FLOAT32), &float32_cde_tensor);
  image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(float32_cde_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::WriteJpeg(filename_2, image_ms_tensor));

  // Test with an invalid image with only one dimension
  image_de_tensor->Reshape(TensorShape({image_1.size[0] * image_1.size[1] * image_1.channels()}));
  image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::WriteJpeg(filename_2, image_ms_tensor));

  // Test with an invalid image with four dimensions
  image_de_tensor->Reshape(TensorShape({image_1.size[0] / 2, image_1.size[1], image_1.channels(), 2}));
  image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::WriteJpeg(filename_2, image_ms_tensor));

  // Test with an invalid image with two channels
  image_de_tensor->Reshape(TensorShape({image_1.size[0] * image_1.channels() / 2, image_1.size[1], 2}));
  image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::WriteJpeg(filename_2, image_ms_tensor));
}

/// Feature: WritePng
/// Description: Test WritePng by writing the image into a PNG file
/// Expectation: The file should be written and removed
TEST_F(MindDataTestPipeline, TestWritePngNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesWritePngNormal.";
  std::string folder_path = "./data/dataset/testFormats/";
  std::string filename_1;
  std::string filename_2;
  cv::Mat image_1;
  cv::Mat image_2;

  filename_1 = folder_path + "apple.png";
  filename_2 = filename_1 + ".test_write_png.png";

  cv::Mat image_bgr = cv::imread(filename_1, cv::ImreadModes::IMREAD_UNCHANGED);
  cv::cvtColor(image_bgr, image_1, cv::COLOR_BGRA2RGB);

  TensorShape img_tensor_shape = TensorShape({image_1.size[0], image_1.size[1], image_1.channels()});
  DataType pixel_type = DataType(DataType::DE_UINT8);

  std::shared_ptr<Tensor> image_de_tensor;
  Tensor::CreateFromMemory(img_tensor_shape, pixel_type, image_1.data, &image_de_tensor);
  auto image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));

  int compression_level;
  for (compression_level = 0; compression_level <= 9; compression_level += 9) {
    ASSERT_OK(mindspore::dataset::vision::WritePng(filename_2, image_ms_tensor, compression_level));
    image_2 = cv::imread(filename_1, cv::ImreadModes::IMREAD_UNCHANGED);
    remove(filename_2.c_str());
    EXPECT_EQ(image_1.total(), image_2.total());
  }
}

/// Feature: WritePng
/// Description: Test WritePng with invalid parameter
/// Expectation: Error is caught when the parameter is invalid
TEST_F(MindDataTestPipeline, TestWritePngException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesWritePngException.";
  std::string folder_path = "./data/dataset/testFormats/";
  std::string filename_1;
  std::string filename_2;
  cv::Mat image_1;

  filename_1 = folder_path + "apple.png";
  filename_2 = filename_1 + ".test_write_png.png";
  image_1 = cv::imread(filename_1, cv::ImreadModes::IMREAD_UNCHANGED);

  TensorShape img_tensor_shape = TensorShape({image_1.size[0], image_1.size[1], image_1.channels()});
  DataType pixel_type = DataType(DataType::DE_UINT8);

  std::shared_ptr<Tensor> image_de_tensor;
  Tensor::CreateFromMemory(img_tensor_shape, pixel_type, image_1.data, &image_de_tensor);
  auto image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));

  // Test with invalid compression_level -1, 10
  ASSERT_ERROR(mindspore::dataset::vision::WritePng(filename_2, image_ms_tensor, -1));
  ASSERT_ERROR(mindspore::dataset::vision::WritePng(filename_2, image_ms_tensor, 10));

  // Test with an invalid filename
  ASSERT_ERROR(mindspore::dataset::vision::WritePng("/dev/cdrom/0", image_ms_tensor));

  // Test with a directory name
  ASSERT_ERROR(mindspore::dataset::vision::WritePng("./data/dataset/", image_ms_tensor));

  // Test with an invalid image containing floating-point elements
  std::shared_ptr<Tensor> float32_de_tensor;
  Tensor::CreateEmpty(TensorShape({5, 4, 3}), DataType(DataType::DE_FLOAT32), &float32_de_tensor);
  image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(float32_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::WritePng(filename_2, image_ms_tensor));

  // Test with an invalid image in only one dimension
  image_de_tensor->Reshape(TensorShape({image_1.size[0] * image_1.size[1] * image_1.channels()}));
  image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::WritePng(filename_2, image_ms_tensor));

  // Test with an invalid image in four dimensions
  image_de_tensor->Reshape(TensorShape({image_1.size[0] / 2, image_1.size[1], image_1.channels(), 2}));
  image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::WritePng(filename_2, image_ms_tensor));

  // Test with an invalid image including two channels
  image_de_tensor->Reshape(TensorShape({image_1.size[0] * image_1.channels() / 2, image_1.size[1], 2}));
  image_ms_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(image_de_tensor));
  ASSERT_ERROR(mindspore::dataset::vision::WritePng(filename_2, image_ms_tensor));
}

#ifdef ENABLE_FFMPEG
/// Description: a function used by the test cases for ReadVideoTimestamps
/// Expectation: create the timestamps according to the vector_length, video_fps, start_index
void create_expected_video_timestamps(std::tuple<std::vector<float>, float> *output, int vector_length, float video_fps,
                                      int start_index=0) {
  std::vector<float> pts_float_vector;
  int index;
  for (index=start_index; index<vector_length + start_index; index++) {
    pts_float_vector.push_back(static_cast<float>(index));
  }
  *output = std::make_tuple(pts_float_vector, video_fps);
}

/// Description: a function used by the test cases for ReadVideoTimestamps
/// Expectation: the difference between the mindspore_data and expected_data should be less than error_rate_limit
void check_video_timestamps(const std::tuple<std::vector<float>, float> &mindspore_data,
                            const std::tuple<std::vector<float>, float> &expected_data,
                            float timestamp_unit, float error_rate_limit=0.0005) {
  std::vector<float> mindspore_vector;
  std::vector<float> expected_vector;
  mindspore_vector = std::get<0>(mindspore_data);
  expected_vector = std::get<0>(expected_data);
  EXPECT_EQ(mindspore_vector.size(), expected_vector.size());

  float difference;

  float mindspore_timestamp_sum=0.0;
  float expected_timestamp_sum=0.0;

  for (float timestamp : mindspore_vector) {
    mindspore_timestamp_sum += timestamp;
  }
  for (float timestamp : expected_vector) {
    expected_timestamp_sum += timestamp;
  }
  expected_timestamp_sum *= timestamp_unit;

  int pts_ok = 0;
  float pts_error_rate;
  difference = std::abs(mindspore_timestamp_sum - expected_timestamp_sum);
  pts_error_rate = difference;
  if (expected_timestamp_sum > 1.0e-5) {
    pts_error_rate = difference / expected_timestamp_sum;
  }
  if (pts_error_rate < error_rate_limit) {
    pts_ok = 1;
  }
  EXPECT_EQ(pts_ok, 1);

  float mindspore_video_fps;
  float expected_video_fps;
  int fps_ok = 0;
  mindspore_video_fps = std::get<1>(mindspore_data);
  expected_video_fps = std::get<1>(expected_data);
  difference = std::abs(mindspore_video_fps - expected_video_fps);
  float fps_error_rate = difference;
  if (expected_video_fps > 1.0e-5) {
      fps_error_rate = difference / expected_video_fps;
  }
  if (fps_error_rate < error_rate_limit) {
    fps_ok = 1;
  }
  EXPECT_EQ(fps_ok, 1);
}

void check_meta_data(std::map<std::string, std::string> &meta_data, const std::string &fps_name, float expected_fps,
                     float error_rate_limit = 0.0005) {
  float error_rate;
  int data_ok = 0;
  float difference = std::abs(std::stof(meta_data[fps_name]) - expected_fps);
  error_rate = difference;
  if (expected_fps > 1.0e-5) {
    error_rate = difference / expected_fps;
  }
  if (error_rate <= error_rate_limit) {
    data_ok = 1;
  }
  EXPECT_EQ(data_ok, 1);
}

void check_mindspore_output(const std::string &filename, float video_fps, float audio_fps) {
  mindspore::MSTensor video_output;
  mindspore::MSTensor audio_output;
  std::map<std::string, std::string> metadata_output;
  float start_pts = 0.0;
  float end_pts = std::numeric_limits<float>::max();

  ASSERT_OK(mindspore::dataset::vision::ReadVideo(filename, &video_output, &audio_output, &metadata_output, start_pts,
                                                  end_pts));
  check_meta_data(metadata_output, "video_fps", video_fps);
  check_meta_data(metadata_output, "audio_fps", audio_fps);

  ASSERT_OK(mindspore::dataset::vision::ReadVideo(filename, &video_output, &audio_output, &metadata_output, start_pts,
                                                  end_pts, "pts"));
  check_meta_data(metadata_output, "video_fps", video_fps);
  check_meta_data(metadata_output, "audio_fps", audio_fps);

  ASSERT_OK(mindspore::dataset::vision::ReadVideo(filename, &video_output, &audio_output, &metadata_output, start_pts,
                                                  end_pts, "sec"));
  check_meta_data(metadata_output, "video_fps", video_fps);
  check_meta_data(metadata_output, "audio_fps", audio_fps);
}

/// Feature: ReadVideo
/// Description: Test ReadVideo with AVI file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoAVINormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoAVINormal.";

  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  filename = folder_path + "campus.avi";
  check_mindspore_output(filename, 29.97003, 48000.0);
}

/// Feature: ReadVideo
/// Description: Test ReadVideo with H264 file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoH264Normal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoH264Normal.";

  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  filename = folder_path + "campus.h264";
  check_mindspore_output(filename, 30.0, 44100.0);
}

/// Feature: ReadVideo
/// Description: Test ReadVideo with H265 file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoH265Normal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoH265Normal.";

  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  filename = folder_path + "campus.h265";
  check_mindspore_output(filename, 25.0, 44100.0);
}

/// Feature: ReadVideo
/// Description: Test ReadVideo with MOV file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoMOVNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoMOVNormal.";

  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  filename = folder_path + "campus.mov";
  check_mindspore_output(filename, 25.0, 44100.0);
}

/// Feature: ReadVideo
/// Description: Test ReadVideo with MP4 file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoMP4Normal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoMP4Normal.";

  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  filename = folder_path + "campus.mp4";
  check_mindspore_output(filename, 25.0, 44100.0);
}

/// Feature: ReadVideo
/// Description: Test ReadVideo with WMV file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoWMVNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoWMVNormal.";

  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  filename = folder_path + "campus.wmv";
  check_mindspore_output(filename, 25.0, 48000.0);
}

/// Feature: ReadVideo
/// Description: Test ReadVideo with invalid parameter
/// Expectation: Error is caught when the the parameter is invalid
TEST_F(MindDataTestPipeline, TestReadVideoException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoException.";
  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  filename = folder_path + "campus.mp4";

  mindspore::MSTensor video_output;
  mindspore::MSTensor audio_output;
  std::map<std::string, std::string> metadata_output;
  float start_pts = 0.0;
  float end_pts = std::numeric_limits<float>::max();

  // Test with a not exist filename
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideo("./this_file_is_not_exist", &video_output, &audio_output,
                                                     &metadata_output, start_pts, end_pts));
  // Test with a directory name
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideo(folder_path, &video_output, &audio_output, &metadata_output,
                                                     start_pts, end_pts));

  // Test with a not supported type of file
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideo("./data/dataset/declient.cfg", &video_output, &audio_output,
                                                     &metadata_output, start_pts, end_pts));

  // Test with an invalid start_pts
  ASSERT_ERROR(
    mindspore::dataset::vision::ReadVideo(folder_path, &video_output, &audio_output, &metadata_output, -1.0, end_pts));

  // Test with an invalid end_pts
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideo(folder_path, &video_output, &audio_output, &metadata_output,
                                                     start_pts, -1.0));

  // Test with a not supported pts_unit
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideo(filename, &video_output, &audio_output, &metadata_output,
                                                     start_pts, end_pts, "min"));
}

/// Feature: ReadVideoTimestamps
/// Description: Test ReadVideoTimestamps with AVI file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoTimestampsAVINormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoTimestampsAVINormal.";

  std::tuple<std::vector<float>, float> mindspore_output;
  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  int expected_vector_length;
  float timestamp_increment;
  float expected_video_fps;
  std::tuple<std::vector<float>, float> expected_output;

  filename = folder_path + "campus.avi";
  expected_vector_length = 5;
  timestamp_increment = 1;
  expected_video_fps = 29.97003;

  create_expected_video_timestamps(&expected_output, expected_vector_length, expected_video_fps);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "pts"));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "sec"));
  check_video_timestamps(mindspore_output, expected_output, 1.0 / expected_video_fps);
}

/// Feature: ReadVideoTimestamps
/// Description: Test ReadVideoTimestamps with H264 file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoTimestampsH264Normal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoTimestampsH264Normal.";

  std::tuple<std::vector<float>, float> mindspore_output;
  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  int expected_vector_length;
  float timestamp_increment;
  float expected_video_fps;
  std::tuple<std::vector<float>, float> expected_output;

  filename = folder_path + "campus.h264";
  expected_vector_length = 19;
  timestamp_increment= 512;
  expected_video_fps = 30.0;

  create_expected_video_timestamps(&expected_output, expected_vector_length, expected_video_fps);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "pts"));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "sec"));
  check_video_timestamps(mindspore_output, expected_output, 1.0 / expected_video_fps);
}

/// Feature: ReadVideoTimestamps
/// Description: Test ReadVideoTimestamps with H265 file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoTimestampsH265Normal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoTimestampsH265Normal.";

  std::tuple<std::vector<float>, float> mindspore_output;
  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  int expected_vector_length;
  float timestamp_increment;
  float expected_video_fps;
  std::tuple<std::vector<float>, float> expected_output;

  filename = folder_path + "campus.h265";
  expected_vector_length = 1;
  timestamp_increment= 5110;
  expected_video_fps = 25.0;

  create_expected_video_timestamps(&expected_output, expected_vector_length, expected_video_fps, 1);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "pts"));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "sec"));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment / (3600.0 * expected_video_fps));
}

/// Feature: ReadVideoTimestamps
/// Description: Test ReadVideoTimestamps with MOV file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoTimestampsMOVNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoTimestampsMOVNormal.";

  std::tuple<std::vector<float>, float> mindspore_output;
  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  int expected_vector_length;
  float timestamp_increment;
  float expected_video_fps;
  std::tuple<std::vector<float>, float> expected_output;

  filename = folder_path + "campus.mov";
  expected_vector_length = 5;
  timestamp_increment= 512;
  expected_video_fps = 25.0;

  create_expected_video_timestamps(&expected_output, expected_vector_length, expected_video_fps);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "pts"));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "sec"));
  check_video_timestamps(mindspore_output, expected_output, 1.0 / expected_video_fps);
}

/// Feature: ReadVideoTimestamps
/// Description: Test ReadVideoTimestamps with MP4 file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoTimestampsMP4Normal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoTimestampsMP4Normal.";

  std::tuple<std::vector<float>, float> mindspore_output;
  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  int expected_vector_length;
  float timestamp_increment;
  float expected_video_fps;
  std::tuple<std::vector<float>, float> expected_output;

  filename = folder_path + "campus.mp4";
  expected_vector_length = 5;
  timestamp_increment= 512;
  expected_video_fps = 25.0;

  create_expected_video_timestamps(&expected_output, expected_vector_length, expected_video_fps);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "pts"));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "sec"));
  check_video_timestamps(mindspore_output, expected_output, 1.0 / expected_video_fps);
}

/// Feature: ReadVideoTimestamps
/// Description: Test ReadVideoTimestamps with WMV file
/// Expectation: The Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestReadVideoTimestampsWMVNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoTimestampsWMVNormal.";

  std::tuple<std::vector<float>, float> mindspore_output;
  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  int expected_vector_length;
  float timestamp_increment;
  float expected_video_fps;
  std::tuple<std::vector<float>, float> expected_output;

  filename = folder_path + "campus.wmv";
  expected_vector_length = 4;
  timestamp_increment= 40.0;
  expected_video_fps = 25.0;

  create_expected_video_timestamps(&expected_output, expected_vector_length, expected_video_fps);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "pts"));
  check_video_timestamps(mindspore_output, expected_output, timestamp_increment);

  ASSERT_OK(mindspore::dataset::vision::ReadVideoTimestamps(filename, &mindspore_output, "sec"));
  check_video_timestamps(mindspore_output, expected_output, 1.0 / expected_video_fps);
}

/// Feature: ReadVideoTimestamps
/// Description: Test ReadVideoTimestamps with invalid parameter
/// Expectation: Error is caught when the the parameter is invalid
TEST_F(MindDataTestPipeline, TestReadVideoTimestampsException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestReadVideoTimestampsException.";
  std::tuple<std::vector<float>, float> output;
  std::string folder_path = "./data/dataset/video/";
  std::string filename;

  filename = folder_path + "campus.mp4";

  // Test with a not exist filename
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideoTimestamps("./this_file_is_not_exist", &output));

  // Test with a directory name
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideoTimestamps(folder_path, &output));

  // Test with a not supported type of file
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideoTimestamps("./data/dataset/declient.cfg", &output));

  // Test with a not supported pts_unit
  ASSERT_ERROR(mindspore::dataset::vision::ReadVideoTimestamps(filename, &output, "min"));
}
#endif
