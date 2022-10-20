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
#include "minddata/dataset/kernels/image/image_utils.h"

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

/// Feature: AutoContrast op
/// Description: Test AutoContrast op with default values
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAutoContrastSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAutoContrastSuccess1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create auto contrast object with default values
  auto auto_contrast = std::make_shared<vision::AutoContrast>();
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({auto_contrast});
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

/// Feature: AutoContrast op
/// Description: Test AutoContrast op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAutoContrastSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAutoContrastSuccess2.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create auto contrast object
  auto auto_contrast = std::make_shared<vision::AutoContrast>(10, std::vector<uint32_t>{10, 20});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({auto_contrast});
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

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op on a batch of CHW images
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCutMixBatchSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchSuccess1.";
  // Testing CutMixBatch on a batch of CHW images

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  int number_of_classes = 10;
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> hwc_to_chw = std::make_shared<vision::HWC2CHW>();
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({hwc_to_chw}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(number_of_classes);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNCHW, 1.0, 1.0);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op}, {"image", "label"});
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
    EXPECT_EQ(image.Shape().size() == 4 && batch_size == image.Shape()[0] && 3 == image.Shape()[1] &&
                32 == image.Shape()[2] && 32 == image.Shape()[3],
              true);
    EXPECT_EQ(label.Shape().size() == 2 && batch_size == label.Shape()[0] && number_of_classes == label.Shape()[1],
              true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op on a batch of HWC images with default values of alpha and prob
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCutMixBatchSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchSuccess2.";
  // Calling CutMixBatch on a batch of HWC images with default values of alpha and prob

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  int number_of_classes = 10;
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(number_of_classes);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op}, {"image", "label"});
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
    EXPECT_EQ(image.Shape().size() == 4 && batch_size == image.Shape()[0] && 32 == image.Shape()[1] &&
                32 == image.Shape()[2] && 3 == image.Shape()[3],
              true);
    EXPECT_EQ(label.Shape().size() == 2 && batch_size == label.Shape()[0] && number_of_classes == label.Shape()[1],
              true);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op with invalid negative alpha parameter
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCutMixBatchFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchFail1 with invalid negative alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create CutMixBatch operation with invalid input, alpha<0
  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC, -1, 0.5);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid CutMixBatch input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op with invalid negative prob parameter
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCutMixBatchFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchFail2 with invalid negative prob parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create CutMixBatch operation with invalid input, prob<0
  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC, 1, -0.5);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid CutMixBatch input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op with invalid zero alpha parameter
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCutMixBatchFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchFail3 with invalid zero alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create CutMixBatch operation with invalid input, alpha=0 (boundary case)
  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC, 0.0, 0.5);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid CutMixBatch input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CutMixBatch op
/// Description: Test CutMixBatch op with invalid prob parameter that is greater than 1
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCutMixBatchFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchFail4 with invalid greater than 1 prob parameter.";

  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 10;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create CutMixBatch operation with invalid input, prob>1
  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC, 1, 1.5);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid CutMixBatch input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CutOut op
/// Description: Test CutOut op basic usage with default HWC parameter
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCutOutDefaultHWC) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutOutDefaultHWC.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> cut_out1 = std::make_shared<vision::CutOut>(30, 5);
  std::shared_ptr<TensorTransform> cut_out2 = std::make_shared<vision::CutOut>(30);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({cut_out1, cut_out2});
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

/// Feature: CutOut op
/// Description: Test CutOut op with HWC true and HWC input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCutOutHWCTrueValid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutOutHWCTrueValid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> cut_out = std::make_shared<vision::CutOut>(30, 5, true);

  // Create a Map operation on ds
  ds = ds->Map({cut_out});
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

/// Feature: CutOut op
/// Description: Test CutOut op with HWC false and HWC input
/// Expectation: Error is caught for mismatched input image format (HWC)
TEST_F(MindDataTestPipeline, TestCutOutHWCFalseInvalid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutOutHWCFalseInvalid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> cut_out = std::make_shared<vision::CutOut>(30, 5, false);

  // Create a Map operation on ds
  ds = ds->Map({cut_out});
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

/// Feature: CutOut op
/// Description: Test CutOut op with HWC true and CHW input
/// Expectation: Error is caught for mismatched input image format (CHW)
TEST_F(MindDataTestPipeline, TestCutOutHWCTrueInvalid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutOutHWCTrueInvalid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto cut_out = std::make_shared<vision::CutOut>(30, 5, true);
  // Op to convert input image format to CHW
  auto HWC2CHW = std::make_shared<vision::HWC2CHW>();

  // Create a Map operation on ds
  ds = ds->Map({HWC2CHW, cut_out});
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

/// Feature: CutOut op
/// Description: Test CutOut op with HWC false and CHW input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCutOutHWCFalseValid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutOutHWCFalseValid.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto cut_out = std::make_shared<vision::CutOut>(30, 5, false);
  // Op to convert input image format to CHW
  auto HWC2CHW = std::make_shared<vision::HWC2CHW>();

  // Create a Map operation on ds
  ds = ds->Map({HWC2CHW, cut_out});
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

/// Feature: Invert op
/// Description: Test Invert op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestInvert) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestInvert.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> invert_op = std::make_shared<vision::Invert>();
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({invert_op});
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

/// Feature: MixUpBatch op
/// Description: Test MixUpBatch op with negative alpha parameter
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMixUpBatchFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMixUpBatchFail1 with negative alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create MixUpBatch operation with invalid input, alpha<0
  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>(-1);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid MixUpBatch input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MixUpBatch op
/// Description: Test MixUpBatch op with zero alpha parameter
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMixUpBatchFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMixUpBatchFail2 with zero alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create MixUpBatch operation with invalid input, alpha<0 (boundary case)
  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>(0.0);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid MixUpBatch input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MixUpBatch op
/// Description: Test MixUpBatch op with explicit valid alpha parameter
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMixUpBatchSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMixUpBatchSuccess1 with explicit alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>(2.0);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op}, {"image", "label"});
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

/// Feature: MixUpBatch op
/// Description: Test MixUpBatch op with default alpha parameter
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMixUpBatchSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMixUpBatchSuccess1 with default alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>();
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op}, {"image", "label"});
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

/// Feature: ConvertColor op
/// Description: Test ConvertColor op with ConvertMode::COLOR_RGB2GRAY
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConvertColorSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConvertColorSuccess1.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);
  // Create objects for the tensor ops
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{500, 1000});
  auto convert = std::make_shared<mindspore::dataset::vision::ConvertColor>(ConvertMode::COLOR_RGB2GRAY);

  ds = ds->Map({resize_op, convert});
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
    EXPECT_EQ(image.Shape().size(), 2);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ConvertColor op
/// Description: Test ConvertColor op with ConvertMode::COLOR_RGB2BGR
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConvertColorSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConvertColorSuccess2.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);
  // Create objects for the tensor ops
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{500, 1000});
  auto convert = std::make_shared<mindspore::dataset::vision::ConvertColor>(ConvertMode::COLOR_RGB2BGR);

  ds = ds->Map({resize_op, convert});
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
    EXPECT_EQ(image.Shape()[2], 3);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ConvertColor op
/// Description: Test ConvertColor op with ConvertMode::COLOR_RGB2RGBA
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConvertColorSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConvertColorSuccess3.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);
  // Create objects for the tensor ops
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{500, 1000});
  auto convert = std::make_shared<mindspore::dataset::vision::ConvertColor>(ConvertMode::COLOR_RGB2RGBA);

  ds = ds->Map({resize_op, convert});
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ConvertColor op
/// Description: Test ConvertColor op with invalid ConvertMode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConvertColorFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConvertColorFail.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  ConvertMode error_convert_mode = static_cast<ConvertMode>(50);

  // Create objects for the tensor ops
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{500, 1000});
  auto convert = std::make_shared<mindspore::dataset::vision::ConvertColor>(error_convert_mode);

  ds = ds->Map({resize_op, convert});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
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
TEST_F(MindDataTestPipeline, TestPadToSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPadToSize.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto pad_to_size = vision::PadToSize({256, 256}, {}, {0}, BorderType::kSymmetric);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({pad_to_size});
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
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PadToSize op
/// Description: Test parameter check
/// Expectation: Error logs are as expected
TEST_F(MindDataTestPipeline, TestPadToSizeInvalid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPadToSizeInvalid.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto invalid_size_shape = vision::PadToSize({});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({invalid_size_shape});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid size shape
  EXPECT_EQ(iter, nullptr);

  // Create objects for the tensor ops
  auto invalid_size_value = vision::PadToSize({0, 0});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({invalid_size_value});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  iter = ds->CreateIterator();
  // Expect failure: Invalid size value
  EXPECT_EQ(iter, nullptr);

  // Create objects for the tensor ops
  auto invalid_offset_shape = vision::PadToSize({256, 256}, {0, 1, 2});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({invalid_offset_shape});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  iter = ds->CreateIterator();
  // Expect failure: Invalid offset shape
  EXPECT_EQ(iter, nullptr);

  // Create objects for the tensor ops
  auto invalid_offset_value = vision::PadToSize({256, 256}, {-1, 0});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({invalid_offset_value});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  iter = ds->CreateIterator();
  // Expect failure: Invalid offset value
  EXPECT_EQ(iter, nullptr);
}

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
TEST_F(MindDataTestPipeline, TestAdjustBrightness) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustBrightness.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto adjustbrightness_op = vision::AdjustBrightness(2.0);

  ds = ds->Map({adjustbrightness_op});
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

/// Feature: AdjustBrightness op
/// Description: Test improper parameters for AdjustBrightness C implementation
/// Expectation: Throw ValueError exception and TypeError exception
TEST_F(MindDataTestPipeline, TestAdjustBrightnessParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustBrightnessParamCheck.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Negative brightness_factor
  // Create objects for the tensor ops
  auto adjustbrightness_op = vision::AdjustBrightness(-1);
  auto ds1 = ds->Map({adjustbrightness_op});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid value of AdjustBrightness
  EXPECT_EQ(iter1, nullptr);
}

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

  auto trivial_augment_wide_op = vision::TrivialAugmentWide(31,
                                             InterpolationMode::kNearestNeighbour, {20, 20});

  ds = ds->Map({trivial_augment_wide_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: AdjustSharpness op
/// Description: Test AdjustSharpness pipeline
/// Expectation: Create an ImageFolder dataset then do auto AjustSharpness on it 
TEST_F(MindDataTestPipeline, TestAdjustSharpness) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustSharpness.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto adjustsharpness_op = vision::AdjustSharpness(2.0);

  ds = ds->Map({adjustsharpness_op});
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

/// Feature: AdjustSharpness op
/// Description: Test parameter check
/// Expectation: Error logs are as expected
TEST_F(MindDataTestPipeline, TestAdjustSharpnessParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustSharpnessParamCheck.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Negative sharpness_factor
  // Create objects for the tensor ops
  auto adjustsharpness_op = vision::AdjustSharpness(-1);
  auto ds1 = ds->Map({adjustsharpness_op});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid value of AdjustSharpness
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: AdjustSaturation op
/// Description: Test AdjustSaturation pipeline
/// Expectation: Create an ImageFolder dataset then do auto AjustSaturation on it 
TEST_F(MindDataTestPipeline, TestAdjustSaturation) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustSaturation.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto adjustsaturation_op = vision::AdjustSaturation(2.0);

  ds = ds->Map({adjustsaturation_op});
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

/// Feature: AdjustSaturation op
/// Description: Test parameter check
/// Expectation: Error logs are as expected
TEST_F(MindDataTestPipeline, TestAdjustSaturationParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustSaturationParamCheck.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Negative saturation_factor
  // Create objects for the tensor ops
  auto adjustsaturation_op = vision::AdjustSaturation(-1);
  auto ds1 = ds->Map({adjustsaturation_op});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid value of AdjustSaturation
  EXPECT_EQ(iter1, nullptr);
}

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

/// Feature：AdjustHue op
/// Description: Test function of operation when hue_factor is 0.2
/// Expectation: Create an ImageFolder dataset then do auto AjustHue on it
TEST_F(MindDataTestPipeline, TestAdjustHue) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustHue.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  auto adjusthue_op = vision::AdjustHue(0.2);

  ds = ds->Map({adjusthue_op});
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

/// Feature：AdjustHue op
/// Description: Test improper parameters for AdjustHue C implementation
/// Expectation: Throw ValueError exception and TypeError exception
TEST_F(MindDataTestPipeline, TestAdjustHueParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustHueParamCheck.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Negative hue_factor
  // Create objects for the tensor ops
  auto adjusthue_op = vision::AdjustHue(-2);
  auto ds1 = ds->Map({adjusthue_op});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid value of AdjustHue
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: AdjustContrast op
/// Description: Test AdjustContrast C implementation Pipeline
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAdjustContrast) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustContrast.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto adjustcontrast_op = vision::AdjustContrast(2.0);

  ds = ds->Map({adjustcontrast_op});
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

/// Feature: AdjustContrast op
/// Description: Test improper parameters for AdjustContrast C implementation
/// Expectation: Throw ValueError exception
TEST_F(MindDataTestPipeline, TestAdjustContrastParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAdjustContrastParamCheck.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Negative contrast_factor
  // Create objects for the tensor ops
  auto adjustcontrast_op = vision::AdjustContrast(-1);
  auto ds1 = ds->Map({adjustcontrast_op});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid value of AdjustContrast
  EXPECT_EQ(iter1, nullptr);
}


/// Feature: Perspective
/// Description: Test Perspective pipeline
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestPerspective){
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
