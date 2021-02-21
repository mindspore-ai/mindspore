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
using mindspore::dataset::BorderType;
using mindspore::dataset::InterpolationMode;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision ops (in alphabetical order)

TEST_F(MindDataTestPipeline, TestAutoContrastSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAutoContrastSuccess1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create auto contrast object with default values
  std::shared_ptr<TensorTransform> auto_contrast(new vision::AutoContrast());
  EXPECT_NE(auto_contrast, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestAutoContrastSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAutoContrastSuccess2.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create auto contrast object
  std::shared_ptr<TensorTransform> auto_contrast(new vision::AutoContrast(10, {10, 20}));
  EXPECT_NE(auto_contrast, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestAutoContrastFail) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAutoContrastFail with invalid params.";
  // Testing invalid cutoff < 0
  std::shared_ptr<TensorTransform> auto_contrast1(new vision::AutoContrast(-1.0));
  // FIXME: Need to check error Status is returned during CreateIterator
  EXPECT_NE(auto_contrast1, nullptr);
  // Testing invalid cutoff > 100
  std::shared_ptr<TensorTransform> auto_contrast2(new vision::AutoContrast(110.0, {10, 20}));
  EXPECT_NE(auto_contrast2, nullptr);
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  /* FIXME - Resolve BoundingBoxAugment to properly handle TensorTransform input
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> bound_box_augment = std::make_shared<vision::BoundingBoxAugment>(vision::RandomRotation({90.0}), 1.0);
  EXPECT_NE(bound_box_augment, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
  */
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentFail with invalid params.";

  // FIXME: For error tests, need to check for failure from CreateIterator execution
  /*
  // Testing invalid ratio < 0.0
  std::shared_ptr<TensorTransform> bound_box_augment = std::make_shared<vision::BoundingBoxAugment>(vision::RandomRotation({90.0}), -1.0);
  EXPECT_EQ(bound_box_augment, nullptr);
  // Testing invalid ratio > 1.0
  std::shared_ptr<TensorTransform> bound_box_augment1 = std::make_shared<vision::BoundingBoxAugment>(vision::RandomRotation({90.0}), 2.0);
  EXPECT_EQ(bound_box_augment1, nullptr);
  // Testing invalid transform
  std::shared_ptr<TensorTransform> bound_box_augment2 = std::make_shared<vision::BoundingBoxAugment>(nullptr, 0.5);
  EXPECT_EQ(bound_box_augment2, nullptr);
  */
}

TEST_F(MindDataTestPipeline, TestCenterCrop) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCenterCrop with single integer input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create centre crop object with square crop
  std::shared_ptr<TensorTransform> centre_out1(new vision::CenterCrop({30}));
  EXPECT_NE(centre_out1, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 15);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCenterCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCenterCrop with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution

  // center crop height value negative
  std::shared_ptr<TensorTransform> center_crop1(new mindspore::dataset::vision::CenterCrop({-32, 32}));
  EXPECT_NE(center_crop1, nullptr);
  // center crop width value negative
  std::shared_ptr<TensorTransform> center_crop2(new mindspore::dataset::vision::CenterCrop({32, -32}));
  EXPECT_NE(center_crop2, nullptr);
  // 0 value would result in nullptr
  std::shared_ptr<TensorTransform> center_crop3(new mindspore::dataset::vision::CenterCrop({0, 32}));
  EXPECT_NE(center_crop3, nullptr);
  // center crop with 3 values
  std::shared_ptr<TensorTransform> center_crop4(new mindspore::dataset::vision::CenterCrop({10, 20, 30}));
  EXPECT_NE(center_crop4, nullptr);
}

TEST_F(MindDataTestPipeline, TestCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCrop with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // wrong width
  std::shared_ptr<TensorTransform> crop1(new mindspore::dataset::vision::Crop({0, 0}, {32, -32}));
  EXPECT_NE(crop1, nullptr);
  // wrong height
  std::shared_ptr<TensorTransform> crop2(new mindspore::dataset::vision::Crop({0, 0}, {-32, -32}));
  EXPECT_NE(crop2, nullptr);
  // zero height
  std::shared_ptr<TensorTransform> crop3(new mindspore::dataset::vision::Crop({0, 0}, {0, 32}));
  EXPECT_NE(crop3, nullptr);
  // negative coordinates
  std::shared_ptr<TensorTransform> crop4(new mindspore::dataset::vision::Crop({-1, 0}, {32, 32}));
  EXPECT_NE(crop4, nullptr);
}

TEST_F(MindDataTestPipeline, TestCutMixBatchSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchSuccess1.";
  // Testing CutMixBatch on a batch of CHW images

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  int number_of_classes = 10;
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> hwc_to_chw = std::make_shared<vision::HWC2CHW>();
  EXPECT_NE(hwc_to_chw, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({hwc_to_chw}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);


  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(number_of_classes);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNCHW, 1.0, 1.0);
  EXPECT_NE(cutmix_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op}, {"image", "label"});
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
    // auto label = row["label"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // MS_LOG(INFO) << "Label shape: " << label->shape();
    // EXPECT_EQ(image->shape().AsVector().size() == 4 && batch_size == image->shape()[0] && 3 == image->shape()[1] &&
    //             32 == image->shape()[2] && 32 == image->shape()[3],
    //           true);
    // EXPECT_EQ(label->shape().AsVector().size() == 2 && batch_size == label->shape()[0] &&
    //             number_of_classes == label->shape()[1],
    //           true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCutMixBatchSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchSuccess2.";
  // Calling CutMixBatch on a batch of HWC images with default values of alpha and prob

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  int number_of_classes = 10;
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(number_of_classes);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> cutmix_batch_op = std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC);
  EXPECT_NE(cutmix_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op}, {"image", "label"});
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
    // auto label = row["label"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // MS_LOG(INFO) << "Label shape: " << label->shape();
    // EXPECT_EQ(image->shape().AsVector().size() == 4 && batch_size == image->shape()[0] && 32 == image->shape()[1] &&
    //             32 == image->shape()[2] && 3 == image->shape()[3],
    //           true);
    // EXPECT_EQ(label->shape().AsVector().size() == 2 && batch_size == label->shape()[0] &&
    //             number_of_classes == label->shape()[1],
    //           true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCutMixBatchFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchFail1 with invalid negative alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create CutMixBatch operation with invalid input, alpha<0
  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC, -1, 0.5);
  EXPECT_NE(cutmix_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid CutMixBatch input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCutMixBatchFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchFail2 with invalid negative prob parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create CutMixBatch operation with invalid input, prob<0
  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC, 1, -0.5);
  EXPECT_NE(cutmix_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid CutMixBatch input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCutMixBatchFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchFail3 with invalid zero alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create CutMixBatch operation with invalid input, alpha=0 (boundary case)
  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC, 0.0, 0.5);
  EXPECT_NE(cutmix_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid CutMixBatch input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCutMixBatchFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutMixBatchFail4 with invalid greater than 1 prob parameter.";

  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 10;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create CutMixBatch operation with invalid input, prob>1
  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNHWC, 1, 1.5);
  EXPECT_NE(cutmix_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid CutMixBatch input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestCutOutFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutOutFail1 with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create object for the tensor op
  // Invalid negative length
  std::shared_ptr<TensorTransform> cutout_op = std::make_shared<vision::CutOut>(-10);
  EXPECT_NE(cutout_op, nullptr);
  // Invalid negative number of patches
  cutout_op = std::make_shared<vision::CutOut>(10, -1);
  EXPECT_NE(cutout_op, nullptr);
}

TEST_F(MindDataTestPipeline, TestCutOutFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutOutFail2 with invalid params, boundary cases.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create object for the tensor op
  // Invalid zero length
  std::shared_ptr<TensorTransform> cutout_op = std::make_shared<vision::CutOut>(0);
  EXPECT_NE(cutout_op, nullptr);
  // Invalid zero number of patches
  cutout_op = std::make_shared<vision::CutOut>(10, 0);
  EXPECT_NE(cutout_op, nullptr);
}

TEST_F(MindDataTestPipeline, TestCutOut) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCutOut.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> cut_out1 = std::make_shared<vision::CutOut>(30, 5);
  EXPECT_NE(cut_out1, nullptr);

  std::shared_ptr<TensorTransform> cut_out2 = std::make_shared<vision::CutOut>(30);
  EXPECT_NE(cut_out2, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDecode.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create Decode object
  vision::Decode decode = vision::Decode(true);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestHwcToChw) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestHwcToChw.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> channel_swap = std::make_shared<vision::HWC2CHW>();
  EXPECT_NE(channel_swap, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // check if the image is in NCHW
    // EXPECT_EQ(batch_size == image->shape()[0] && 3 == image->shape()[1] && 2268 == image->shape()[2] &&
    //             4032 == image->shape()[3],
    //           true);
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestInvert) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestInvert.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> invert_op = std::make_shared<vision::Invert>();
  EXPECT_NE(invert_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({invert_op});
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
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMixUpBatchFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMixUpBatchFail1 with negative alpha parameter.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create MixUpBatch operation with invalid input, alpha<0
  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>(-1);
  EXPECT_NE(mixup_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid MixUpBatch input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestMixUpBatchFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMixUpBatchFail2 with zero alpha parameter.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create MixUpBatch operation with invalid input, alpha<0 (boundary case)
  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>(0.0);
  EXPECT_NE(mixup_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid MixUpBatch input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestMixUpBatchSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMixUpBatchSuccess1 with explicit alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>(2.0);
  EXPECT_NE(mixup_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op}, {"image", "label"});
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMixUpBatchSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMixUpBatchSuccess1 with default alpha parameter.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>();
  EXPECT_NE(mixup_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op}, {"image", "label"});
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNormalize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> normalize(new vision::Normalize({121.0, 115.0, 0.0}, {70.0, 68.0, 71.0}));
  EXPECT_NE(normalize, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNormalizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizeFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // std value at 0.0
  std::shared_ptr<TensorTransform> normalize1(
    new mindspore::dataset::vision::Normalize({121.0, 115.0, 100.0}, {0.0, 68.0, 71.0}));
  EXPECT_NE(normalize1, nullptr);
  // mean out of range
  std::shared_ptr<TensorTransform> normalize2(
    new mindspore::dataset::vision::Normalize({121.0, 0.0, 100.0}, {256.0, 68.0, 71.0}));
  EXPECT_NE(normalize2, nullptr);
  // mean out of range
  std::shared_ptr<TensorTransform> normalize3(
    new mindspore::dataset::vision::Normalize({256.0, 0.0, 100.0}, {70.0, 68.0, 71.0}));
  EXPECT_NE(normalize3, nullptr);
  // mean out of range
  std::shared_ptr<TensorTransform> normalize4(
    new mindspore::dataset::vision::Normalize({-1.0, 0.0, 100.0}, {70.0, 68.0, 71.0}));
  EXPECT_NE(normalize4, nullptr);
  // normalize with 2 values (not 3 values) for mean
  std::shared_ptr<TensorTransform> normalize5(
    new mindspore::dataset::vision::Normalize({121.0, 115.0}, {70.0, 68.0, 71.0}));
  EXPECT_NE(normalize5, nullptr);
  // normalize with 2 values (not 3 values) for standard deviation
  std::shared_ptr<TensorTransform> normalize6(
    new mindspore::dataset::vision::Normalize({121.0, 115.0, 100.0}, {68.0, 71.0}));
  EXPECT_NE(normalize6, nullptr);
}

TEST_F(MindDataTestPipeline, TestNormalizePad) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePad.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> normalizepad(
    new vision::NormalizePad({121.0, 115.0, 100.0}, {70.0, 68.0, 71.0}, "float32"));
  EXPECT_NE(normalizepad, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({normalizepad});
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
    // EXPECT_EQ(image->shape()[2], 4);
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNormalizePadFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestNormalizePadFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // std value at 0.0
  std::shared_ptr<TensorTransform> normalizepad1(
    new mindspore::dataset::vision::NormalizePad({121.0, 115.0, 100.0}, {0.0, 68.0, 71.0}));
  EXPECT_NE(normalizepad1, nullptr);
  // normalizepad with 2 values (not 3 values) for mean
  std::shared_ptr<TensorTransform> normalizepad2(
    new mindspore::dataset::vision::NormalizePad({121.0, 115.0}, {70.0, 68.0, 71.0}));
  EXPECT_NE(normalizepad2, nullptr);
  // normalizepad with 2 values (not 3 values) for standard deviation
  std::shared_ptr<TensorTransform> normalizepad3(
    new mindspore::dataset::vision::NormalizePad({121.0, 115.0, 100.0}, {68.0, 71.0}));
  EXPECT_NE(normalizepad3, nullptr);
  // normalizepad with invalid dtype
  std::shared_ptr<TensorTransform> normalizepad4(
    new mindspore::dataset::vision::NormalizePad({121.0, 115.0, 100.0}, {68.0, 71.0, 71.0}, "123"));
  EXPECT_NE(normalizepad4, nullptr);
}

TEST_F(MindDataTestPipeline, TestPad) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPad.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> pad_op1(new vision::Pad({1, 2, 3, 4}, {0}, BorderType::kSymmetric));
  EXPECT_NE(pad_op1, nullptr);

  std::shared_ptr<TensorTransform> pad_op2(new vision::Pad({1}, {1, 1, 1}, BorderType::kEdge));
  EXPECT_NE(pad_op2, nullptr);

  std::shared_ptr<TensorTransform> pad_op3(new vision::Pad({1, 4}));
  EXPECT_NE(pad_op3, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomAffineFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine1(new vision::RandomAffine({0.0, 0.0}, {}));
  EXPECT_NE(affine1, nullptr);
  // Invalid number of values for translate
  std::shared_ptr<TensorTransform> affine2(new vision::RandomAffine({0.0, 0.0}, {1, 1, 1, 1, 1}));
  EXPECT_NE(affine2, nullptr);
  // Invalid number of values for shear
  std::shared_ptr<TensorTransform> affine3(new vision::RandomAffine({30.0, 30.0}, {0.0, 0.0}, {2.0, 2.0}, {10.0}));
  EXPECT_NE(affine3, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomAffineSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineSuccess1 with non-default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine(
    new vision::RandomAffine({30.0, 30.0}, {-1.0, 1.0, -1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0, 20.0, 20.0}));
  EXPECT_NE(affine, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({affine});
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomAffineSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine(new vision::RandomAffine({0.0, 0.0}));
  EXPECT_NE(affine, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({affine});
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomColor) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColor with non-default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Valid case: Set lower bound and upper bound to be the same value zero
  std::shared_ptr<TensorTransform> random_color_op_1 = std::make_shared<vision::RandomColor>(0.0, 0.0);
  EXPECT_NE(random_color_op_1, nullptr);

  // Failure case: Set invalid lower bound greater than upper bound
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  std::shared_ptr<TensorTransform> random_color_op_2 = std::make_shared<vision::RandomColor>(1.0, 0.1);
  EXPECT_NE(random_color_op_2, nullptr);

  // Valid case: Set lower bound as zero and less than upper bound
  std::shared_ptr<TensorTransform> random_color_op_3 = std::make_shared<vision::RandomColor>(0.0, 1.1);
  EXPECT_NE(random_color_op_3, nullptr);

  // Failure case: Set invalid negative lower bound
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  std::shared_ptr<TensorTransform> random_color_op_4 = std::make_shared<vision::RandomColor>(-0.5, 0.5);
  EXPECT_NE(random_color_op_4, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_color_op_1, random_color_op_3});
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomColorAdjust) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColorAdjust.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use single value for vectors
  std::shared_ptr<TensorTransform> random_color_adjust1(new vision::RandomColorAdjust({1.0}, {0.0}, {0.5}, {0.5}));
  EXPECT_NE(random_color_adjust1, nullptr);

  // Use same 2 values for vectors
  std::shared_ptr<TensorTransform> random_color_adjust2(new
    vision::RandomColorAdjust({1.0, 1.0}, {0.0, 0.0}, {0.5, 0.5}, {0.5, 0.5}));
  EXPECT_NE(random_color_adjust2, nullptr);

  // Use different 2 value for vectors
  std::shared_ptr<TensorTransform> random_color_adjust3(new
    vision::RandomColorAdjust({0.5, 1.0}, {0.0, 0.5}, {0.25, 0.5}, {0.25, 0.5}));
  EXPECT_NE(random_color_adjust3, nullptr);

  // Use default input values
  std::shared_ptr<TensorTransform> random_color_adjust4(new vision::RandomColorAdjust());
  EXPECT_NE(random_color_adjust4, nullptr);

  // Use subset of explicitly set parameters
  std::shared_ptr<TensorTransform> random_color_adjust5(new vision::RandomColorAdjust({0.0, 0.5}, {0.25}));
  EXPECT_NE(random_color_adjust5, nullptr);

  // Create a Map operation on ds
  ds = ds->Map(
    {random_color_adjust1, random_color_adjust2, random_color_adjust3, random_color_adjust4, random_color_adjust5});
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomColorAdjustFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColorAdjustFail.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // brightness out of range
  std::shared_ptr<TensorTransform> random_color_adjust1(new vision::RandomColorAdjust({-1.0}));
  EXPECT_NE(random_color_adjust1, nullptr);

  // contrast out of range
  std::shared_ptr<TensorTransform> random_color_adjust2(new vision::RandomColorAdjust({1.0}, {-0.1}));
  EXPECT_NE(random_color_adjust2, nullptr);

  // saturation out of range
  std::shared_ptr<TensorTransform> random_color_adjust3(new vision::RandomColorAdjust({0.0}, {0.0}, {-0.2}));
  EXPECT_NE(random_color_adjust3, nullptr);

  // hue out of range
  std::shared_ptr<TensorTransform> random_color_adjust4(new vision::RandomColorAdjust({0.0}, {0.0}, {0.0}, {-0.6}));
  EXPECT_NE(random_color_adjust4, nullptr);

  std::shared_ptr<TensorTransform> random_color_adjust5(new vision::RandomColorAdjust({0.0}, {0.0}, {0.0}, {-0.5, 0.6}));
  EXPECT_NE(random_color_adjust5, nullptr);

  std::shared_ptr<TensorTransform> random_color_adjust6(new vision::RandomColorAdjust({0.0}, {0.0}, {0.0}, {0.51}));
  EXPECT_NE(random_color_adjust6, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomCropSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing siez of size vector is 1
  std::shared_ptr<TensorTransform> random_crop(new vision::RandomCrop({20}));
  EXPECT_NE(random_crop, nullptr);

  // Testing siez of size vector is 2
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCrop({20, 20}));
  EXPECT_NE(random_crop1, nullptr);

  // Testing siez of paddiing vector is 1
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCrop({20, 20}, {10}));
  EXPECT_NE(random_crop2, nullptr);

  // Testing siez of paddiing vector is 2
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCrop({20, 20}, {10, 20}));
  EXPECT_NE(random_crop3, nullptr);

  // Testing siez of paddiing vector is 2
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}));
  EXPECT_NE(random_crop4, nullptr);

  // Testing siez of fill_value vector is 1
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}, false, {5}));
  EXPECT_NE(random_crop5, nullptr);

  // Testing siez of fill_value vector is 3
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}, false, {4, 4, 4}));
  EXPECT_NE(random_crop6, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_crop, random_crop1, random_crop2, random_crop3, random_crop4, random_crop5, random_crop6});
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

  EXPECT_EQ(i, 10);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing the size parameter is negative.
  std::shared_ptr<TensorTransform> random_crop(new vision::RandomCrop({-28, 28}));
  EXPECT_NE(random_crop, nullptr);
  // Testing the size parameter is None.
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCrop({}));
  EXPECT_NE(random_crop1, nullptr);
  // Testing the size of size vector is 3.
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCrop({28, 28, 28}));
  EXPECT_NE(random_crop2, nullptr);
  // Testing the padding parameter is negative.
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCrop({28, 28}, {-5}));
  EXPECT_NE(random_crop3, nullptr);
  // Testing the size of padding vector is empty.
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCrop({28, 28}, {}));
  EXPECT_NE(random_crop4, nullptr);
  // Testing the size of padding vector is 3.
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCrop({28, 28}, {5, 5, 5}));
  EXPECT_NE(random_crop5, nullptr);
  // Testing the size of padding vector is 5.
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCrop({28, 28}, {5, 5, 5, 5, 5}));
  EXPECT_NE(random_crop6, nullptr);
  // Testing the size of fill_value vector is empty.
  std::shared_ptr<TensorTransform> random_crop7(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {}));
  EXPECT_NE(random_crop7, nullptr);
  // Testing the size of fill_value vector is 2.
  std::shared_ptr<TensorTransform> random_crop8(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {0, 0}));
  EXPECT_NE(random_crop8, nullptr);
  // Testing the size of fill_value vector is 4.
  std::shared_ptr<TensorTransform> random_crop9(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {0, 0, 0, 0}));
  EXPECT_NE(random_crop9, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomCropWithBboxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithBboxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop(new mindspore::dataset::vision::RandomCropWithBBox({128, 128}));
  EXPECT_NE(random_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_crop}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0], 128);
    // EXPECT_EQ(image->shape()[1], 128);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomCropWithBboxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithBboxFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // The size parameter is negative.
  std::shared_ptr<TensorTransform> random_crop0(new vision::RandomCropWithBBox({-10}));
  EXPECT_NE(random_crop0, nullptr);
  // The parameter in the padding vector is negative.
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCropWithBBox({10, 10}, {-2, 2, 2, 2}));
  EXPECT_NE(random_crop1, nullptr);
  // The size container is empty.
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCropWithBBox({}));
  EXPECT_NE(random_crop2, nullptr);
  // The size of the size container is too large.
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCropWithBBox({10, 10, 10}));
  EXPECT_NE(random_crop3, nullptr);
  // The padding container is empty.
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCropWithBBox({10, 10}, {}));
  EXPECT_NE(random_crop4, nullptr);
  // The size of the padding container is too large.
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5, 5}));
  EXPECT_NE(random_crop5, nullptr);
  // The fill_value container is empty.
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5}, false, {}));
  EXPECT_NE(random_crop6, nullptr);
  // The size of the fill_value container is too large.
  std::shared_ptr<TensorTransform> random_crop7(new
    vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5}, false, {3, 3, 3, 3}));
  EXPECT_NE(random_crop7, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomHorizontalFlipFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalFlipFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create object for the tensor op
  // Invalid negative input
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(-0.5);
  EXPECT_NE(random_horizontal_flip_op, nullptr);
  // Invalid >1 input
  random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(2);
  EXPECT_NE(random_horizontal_flip_op, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomHorizontalFlipWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalFlipWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlipWithBBox>(0.5);
  EXPECT_NE(random_horizontal_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_horizontal_flip_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomHorizontalFlipWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalFlipWithBBoxFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Incorrect prob parameter.
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlipWithBBox>(-1.0);
  EXPECT_NE(random_horizontal_flip_op, nullptr);
  // Incorrect prob parameter.
  std::shared_ptr<TensorTransform> random_horizontal_flip_op1 = std::make_shared<vision::RandomHorizontalFlipWithBBox>(2.0);
  EXPECT_NE(random_horizontal_flip_op1, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomHorizontalAndVerticalFlip) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalAndVerticalFlip for horizontal and vertical flips.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(0.75);
  EXPECT_NE(random_vertical_flip_op, nullptr);

  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(0.5);
  EXPECT_NE(random_horizontal_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_vertical_flip_op, random_horizontal_flip_op});
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomPosterizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create objects for the tensor ops
  // Invalid max > 8
  std::shared_ptr<TensorTransform> posterize1(new vision::RandomPosterize({1, 9}));
  EXPECT_NE(posterize1, nullptr);
  // Invalid min < 1
  std::shared_ptr<TensorTransform> posterize2(new vision::RandomPosterize({0, 8}));
  EXPECT_NE(posterize2, nullptr);
  // min > max
  std::shared_ptr<TensorTransform> posterize3(new vision::RandomPosterize({8, 1}));
  EXPECT_NE(posterize3, nullptr);
  // empty
  //std::shared_ptr<TensorTransform> posterize4(new vision::RandomPosterize({}));
  // EXPECT_NE(posterize4, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomPosterizeSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeSuccess1 with non-default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> posterize(new vision::RandomPosterize({1, 4}));
  EXPECT_NE(posterize, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomPosterizeSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> posterize(new vision::RandomPosterize());
  EXPECT_NE(posterize, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeSuccess1 with single integer input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResize({66}));
  EXPECT_NE(random_resize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 66, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeSuccess2 with (height, width) input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResize({66, 77}));
  EXPECT_NE(random_resize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 66 && image->shape()[1] == 77, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeFail incorrect size.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // RandomResize : size must only contain positive integers
  std::shared_ptr<TensorTransform> random_resize1(new vision::RandomResize({-66, 77}));
  EXPECT_NE(random_resize1, nullptr);

  // RandomResize : size must only contain positive integers
  std::shared_ptr<TensorTransform> random_resize2(new vision::RandomResize({0, 77}));
  EXPECT_NE(random_resize2, nullptr);

  // RandomResize : size must be a vector of one or two values
  std::shared_ptr<TensorTransform> random_resize3(new vision::RandomResize({1, 2, 3}));
  EXPECT_NE(random_resize3, nullptr);

  // RandomResize : size must be a vector of one or two values
  std::shared_ptr<TensorTransform> random_resize4(new vision::RandomResize({}));
  EXPECT_NE(random_resize4, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizeWithBBoxSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithBBoxSuccess1 with single integer input.";

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResizeWithBBox({88}));
  EXPECT_NE(random_resize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0] == 88, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeWithBBoxSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithBBoxSuccess2 with (height, width) input.";

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResizeWithBBox({88, 99}));
  EXPECT_NE(random_resize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0] == 88 && image->shape()[1] == 99, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithBBoxFail incorrect size.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // RandomResizeWithBBox : size must only contain positive integers
  std::shared_ptr<TensorTransform> random_resize_with_bbox1(new vision::RandomResizeWithBBox({-66, 77}));
  EXPECT_NE(random_resize_with_bbox1, nullptr);

  // RandomResizeWithBBox : size must be a vector of one or two values
  std::shared_ptr<TensorTransform> random_resize_with_bbox2(new vision::RandomResizeWithBBox({1, 2, 3}));
  EXPECT_NE(random_resize_with_bbox2, nullptr);

  // RandomResizeWithBBox : size must be a vector of one or two values
  std::shared_ptr<TensorTransform> random_resize_with_bbox3(new vision::RandomResizeWithBBox({}));
  EXPECT_NE(random_resize_with_bbox3, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropSuccess1) {
  // Testing RandomResizedCrop with default values
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 5 && image->shape()[1] == 5, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropSuccess2) {
  // Testing RandomResizedCrop with non-default values
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new
    vision::RandomResizedCrop({5, 10}, {0.25, 0.75}, {0.5, 1.25}, mindspore::dataset::InterpolationMode::kArea, 20));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 5 && image->shape()[1] == 10, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail1) {
  // This should fail because size has negative value
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, -10}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail2) {
  // This should fail because scale isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10}, {4, 3}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail3) {
  // This should fail because ratio isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10}, {4, 5}, {7, 6}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail4) {
  // This should fail because scale has a size of more than 2
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10, 20}, {4, 5}, {7, 6}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxSuccess1) {
  // Testing RandomResizedCropWithBBox with default values
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0] == 5 && image->shape()[1] == 5, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxSuccess2) {
  // Testing RandomResizedCropWithBBox with non-default values
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox(
    {5, 10}, {0.25, 0.75}, {0.5, 1.25}, mindspore::dataset::InterpolationMode::kArea, 20));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0] == 5 && image->shape()[1] == 10, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail1) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // This should fail because size has negative value
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, -10}));
  EXPECT_NE(random_resized_crop, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail2) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // This should fail because scale isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, 10}, {4, 3}));
  EXPECT_NE(random_resized_crop, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail3) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // This should fail because ratio isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, 10}, {4, 5}, {7, 6}));
  EXPECT_NE(random_resized_crop, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail4) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // This should fail because scale has a size of more than 2
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, 10, 20}, {4, 5}, {7, 6}));
  EXPECT_NE(random_resized_crop, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomRotation) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomRotation.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing the size of degrees is 1
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({180}));
  EXPECT_NE(random_rotation_op, nullptr);
  // Testing the size of degrees is 2
  std::shared_ptr<TensorTransform> random_rotation_op1(new vision::RandomRotation({-180, 180}));
  EXPECT_NE(random_rotation_op1, nullptr);
  // Testing the size of fill_value is 1
  std::shared_ptr<TensorTransform> random_rotation_op2(new
    vision::RandomRotation({180}, InterpolationMode::kNearestNeighbour, false, {-1, -1}, {2}));
  EXPECT_NE(random_rotation_op2, nullptr);
  // Testing the size of fill_value is 3
  std::shared_ptr<TensorTransform> random_rotation_op3(new
    vision::RandomRotation({180}, InterpolationMode::kNearestNeighbour, false, {-1, -1}, {2, 2, 2}));
  EXPECT_NE(random_rotation_op3, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_rotation_op, random_rotation_op1, random_rotation_op2, random_rotation_op3});
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomRotationFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomRotationFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing the size of degrees vector is 0
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({}));
  EXPECT_NE(random_rotation_op, nullptr);
  // Testing the size of degrees vector is 3
  std::shared_ptr<TensorTransform> random_rotation_op1(new vision::RandomRotation({-50.0, 50.0, 100.0}));
  EXPECT_NE(random_rotation_op1, nullptr);
  // Test the case where the first column value of degrees is greater than the second column value
  std::shared_ptr<TensorTransform> random_rotation_op2(new vision::RandomRotation({50.0, -50.0}));
  EXPECT_NE(random_rotation_op2, nullptr);
  // Testing the size of center vector is 1
  std::shared_ptr<TensorTransform> random_rotation_op3(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0}));
  EXPECT_NE(random_rotation_op3, nullptr);
  // Testing the size of center vector is 3
  std::shared_ptr<TensorTransform> random_rotation_op4(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0, -1.0}));
  EXPECT_NE(random_rotation_op4, nullptr);
  // Testing the size of fill_value vector is 2
  std::shared_ptr<TensorTransform> random_rotation_op5(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0}, {2, 2}));
  EXPECT_NE(random_rotation_op5, nullptr);
  // Testing the size of fill_value vector is 4
  std::shared_ptr<TensorTransform> random_rotation_op6(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0}, {2, 2, 2, 2}));
  EXPECT_NE(random_rotation_op6, nullptr);
}

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

TEST_F(MindDataTestPipeline, TestRandomSharpness) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSharpness.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Valid case: Input start degree and end degree
  std::shared_ptr<TensorTransform> random_sharpness_op_1(new vision::RandomSharpness({0.4, 2.3}));
  EXPECT_NE(random_sharpness_op_1, nullptr);

  // Failure case: Empty degrees vector
  //
  // std::shared_ptr<TensorTransform> random_sharpness_op_2(new vision::RandomSharpness({}));
  //
  // EXPECT_NE(random_sharpness_op_2, nullptr);

  // Valid case: Use default input values
  std::shared_ptr<TensorTransform> random_sharpness_op_3(new vision::RandomSharpness());
  EXPECT_NE(random_sharpness_op_3, nullptr);

  // Failure case: Single degree value
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  std::shared_ptr<TensorTransform> random_sharpness_op_4(new vision::RandomSharpness({0.1}));
  EXPECT_NE(random_sharpness_op_4, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_sharpness_op_1, random_sharpness_op_3});
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeSucess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarizeSucess1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::vector<uint8_t> threshold = {10, 100};
  std::shared_ptr<TensorTransform> random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_solarize});
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

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeSucess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarizeSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>();
  EXPECT_NE(random_solarize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_solarize});
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

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarizeFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  std::vector<uint8_t> threshold = {13, 1};
  std::shared_ptr<TensorTransform> random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);

  threshold = {1, 2, 3};
  random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);

  threshold = {1};
  random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);

  threshold = {};
  random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomVerticalFlipFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomVerticalFlipFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create object for the tensor op
  // Invalid negative input
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(-0.5);
  EXPECT_NE(random_vertical_flip_op, nullptr);
  // Invalid >1 input
  random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(1.1);
  EXPECT_NE(random_vertical_flip_op, nullptr);
}

TEST_F(MindDataTestPipeline, TestResizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResize with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // negative resize value
  std::shared_ptr<TensorTransform> resize_op1(new mindspore::dataset::vision::Resize({30, -30}));
  EXPECT_NE(resize_op1, nullptr);
  // zero resize value
  std::shared_ptr<TensorTransform> resize_op2(new mindspore::dataset::vision::Resize({0, 30}));
  EXPECT_NE(resize_op2, nullptr);
  // resize with 3 values
  std::shared_ptr<TensorTransform> resize_op3(new mindspore::dataset::vision::Resize({30, 20, 10}));
  EXPECT_NE(resize_op3, nullptr);
}

TEST_F(MindDataTestPipeline, TestResizeWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizeWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> resize_with_bbox_op(new vision::ResizeWithBBox({30}));
  EXPECT_NE(resize_with_bbox_op, nullptr);

  std::shared_ptr<TensorTransform> resize_with_bbox_op1(new vision::ResizeWithBBox({30, 30}));
  EXPECT_NE(resize_with_bbox_op1, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({resize_with_bbox_op, resize_with_bbox_op1}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestResizeWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizeWithBBoxFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Testing negative resize value
  std::shared_ptr<TensorTransform> resize_with_bbox_op(new vision::ResizeWithBBox({10, -10}));
  EXPECT_NE(resize_with_bbox_op, nullptr);
  // Testing negative resize value
  std::shared_ptr<TensorTransform> resize_with_bbox_op1(new vision::ResizeWithBBox({-10}));
  EXPECT_NE(resize_with_bbox_op1, nullptr);
  // Testinig zero resize value
  std::shared_ptr<TensorTransform> resize_with_bbox_op2(new vision::ResizeWithBBox({0, 10}));
  EXPECT_NE(resize_with_bbox_op2, nullptr);
  // Testing resize with 3 values
  std::shared_ptr<TensorTransform> resize_with_bbox_op3(new vision::ResizeWithBBox({10, 10, 10}));
  EXPECT_NE(resize_with_bbox_op3, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomVerticalFlipWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomVerticalFlipWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlipWithBBox>(0.4);
  EXPECT_NE(random_vertical_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_vertical_flip_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomVerticalFlipWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomVerticalFlipWithBBoxFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Incorrect prob parameter.
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlipWithBBox>(-0.5);
  EXPECT_NE(random_vertical_flip_op, nullptr);
  // Incorrect prob parameter.
  std::shared_ptr<TensorTransform> random_vertical_flip_op1 = std::make_shared<vision::RandomVerticalFlipWithBBox>(3.0);
  EXPECT_NE(random_vertical_flip_op1, nullptr);
}

TEST_F(MindDataTestPipeline, TestResize1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResize1 with single integer input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 6));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 4;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create resize object with single integer input
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30}));
  EXPECT_NE(resize_op, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 24);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRescaleSucess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleSucess1.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, SequentialSampler(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  auto image = row["image"];

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> rescale(new mindspore::dataset::vision::Rescale(1.0, 0.0));
  EXPECT_NE(rescale, nullptr);

  // Convert to the same type
  std::shared_ptr<TensorTransform> type_cast(new transforms::TypeCast("uint8"));
  EXPECT_NE(type_cast, nullptr);

  ds = ds->Map({rescale, type_cast}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_NE(iter1, nullptr);

  // Iterate the dataset and get each row1
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  iter1->GetNextRow(&row1);

  auto image1 = row1["image"];

  // EXPECT_EQ(*image, *image1);

  // Manually terminate the pipeline
  iter1->Stop();
}

TEST_F(MindDataTestPipeline, TestRescaleSucess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleSucess2 with different params.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> rescale(new mindspore::dataset::vision::Rescale(1.0 / 255, 1.0));
  EXPECT_NE(rescale, nullptr);

  ds = ds->Map({rescale}, {"image"});
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRescaleFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleFail with invalid params.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // incorrect negative rescale parameter
  std::shared_ptr<TensorTransform> rescale(new mindspore::dataset::vision::Rescale(-1.0, 0.0));
  EXPECT_NE(rescale, nullptr);
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeRandomCropResizeJpegSuccess1) {
  MS_LOG(INFO)
    << "Doing MindDataTestPipeline-TestSoftDvppDecodeRandomCropResizeJpegSuccess1 with single integer input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> soft_dvpp_decode_random_crop_resize_jpeg(new
    vision::SoftDvppDecodeRandomCropResizeJpeg({500}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({soft_dvpp_decode_random_crop_resize_jpeg}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 500 && image->shape()[1] == 500, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeRandomCropResizeJpegSuccess2) {
  MS_LOG(INFO)
    << "Doing MindDataTestPipeline-TestSoftDvppDecodeRandomCropResizeJpegSuccess2 with (height, width) input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 6));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> soft_dvpp_decode_random_crop_resize_jpeg(new 
    vision::SoftDvppDecodeRandomCropResizeJpeg({500, 600}, {0.25, 0.75}, {0.5, 1.25}, 20));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({soft_dvpp_decode_random_crop_resize_jpeg}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 500 && image->shape()[1] == 600, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeRandomCropResizeJpegFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeRandomCropResizeJpegFail with incorrect parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // SoftDvppDecodeRandomCropResizeJpeg: size must only contain positive integers
  auto soft_dvpp_decode_random_crop_resize_jpeg1(new vision::SoftDvppDecodeRandomCropResizeJpeg({-500, 600}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg1, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: size must only contain positive integers
  auto soft_dvpp_decode_random_crop_resize_jpeg2(new vision::SoftDvppDecodeRandomCropResizeJpeg({-500}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg2, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: size must be a vector of one or two values
  auto soft_dvpp_decode_random_crop_resize_jpeg3(new vision::SoftDvppDecodeRandomCropResizeJpeg({500, 600, 700}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg3, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be greater than or equal to 0
  auto soft_dvpp_decode_random_crop_resize_jpeg4(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {-0.1, 0.9}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg4, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be in the format of (min, max)
  auto soft_dvpp_decode_random_crop_resize_jpeg5(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.6, 0.2}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg5, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be a vector of two values
  auto soft_dvpp_decode_random_crop_resize_jpeg6(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.6, 0.7}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg6, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be greater than or equal to 0
  auto soft_dvpp_decode_random_crop_resize_jpeg7(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.9}, {-0.2, 0.4}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg7, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be in the format of (min, max)
  auto soft_dvpp_decode_random_crop_resize_jpeg8(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.9}, {0.4, 0.2}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg8, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be a vector of two values
  auto soft_dvpp_decode_random_crop_resize_jpeg9(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.9}, {0.1, 0.2, 0.3}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg9, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: max_attempts must be greater than or equal to 1
  auto soft_dvpp_decode_random_crop_resize_jpeg10(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.9}, {0.1, 0.2}, 0));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg10, nullptr);
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeResizeJpegSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeResizeJpegSuccess1 with single integer input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create SoftDvppDecodeResizeJpeg object with single integer input
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op(new vision::SoftDvppDecodeResizeJpeg({1134}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({soft_dvpp_decode_resize_jpeg_op});
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

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeResizeJpegSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeResizeJpegSuccess2 with (height, width) input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create SoftDvppDecodeResizeJpeg object with single integer input
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op(new vision::SoftDvppDecodeResizeJpeg({100, 200}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({soft_dvpp_decode_resize_jpeg_op});
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeResizeJpegFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeResizeJpegFail with incorrect size.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // CSoftDvppDecodeResizeJpeg: size must be a vector of one or two values
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op1(new vision::SoftDvppDecodeResizeJpeg({}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op1, nullptr);

  // SoftDvppDecodeResizeJpeg: size must be a vector of one or two values
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op2(new vision::SoftDvppDecodeResizeJpeg({1, 2, 3}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op2, nullptr);

  // SoftDvppDecodeResizeJpeg: size must only contain positive integers
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op3(new vision::SoftDvppDecodeResizeJpeg({20, -20}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op3, nullptr);

  // SoftDvppDecodeResizeJpeg: size must only contain positive integers
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op4(new vision::SoftDvppDecodeResizeJpeg({0}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op4, nullptr);
}

TEST_F(MindDataTestPipeline, TestUniformAugmentFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail1 with invalid num_ops parameter.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  /*
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  EXPECT_NE(random_crop_op, nullptr);

  std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
  EXPECT_NE(center_crop_op, nullptr);

  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // UniformAug: num_ops must be greater than 0
  std::shared_ptr<TensorTransform> uniform_aug_op1(new vision::UniformAugment({random_crop_op, center_crop_op}, 0));
  EXPECT_EQ(uniform_aug_op1, nullptr);

  // UniformAug: num_ops must be greater than 0
  std::shared_ptr<TensorTransform> uniform_aug_op2(new vision::UniformAugment({random_crop_op, center_crop_op}, -1));
  EXPECT_EQ(uniform_aug_op2, nullptr);

  // UniformAug: num_ops is greater than transforms size
  std::shared_ptr<TensorTransform> uniform_aug_op3(new vision::UniformAugment({random_crop_op, center_crop_op}, 3));
  EXPECT_EQ(uniform_aug_op3, nullptr);
  */

}

TEST_F(MindDataTestPipeline, TestUniformAugmentFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail2 with invalid transform.";

  // FIXME: For error tests, need to check for failure from CreateIterator execution
  /*
  // UniformAug: transform ops must not be null
  std::shared_ptr<TensorTransform> uniform_aug_op1(new vision::UniformAugment({vision::RandomCrop({-28})}, 1));
  EXPECT_NE(uniform_aug_op1, nullptr);

  // UniformAug: transform ops must not be null
  std::shared_ptr<TensorTransform> uniform_aug_op2(new vision::UniformAugment({vision::RandomCrop({28}), nullptr}, 2));
  EXPECT_NE(uniform_aug_op2, nullptr);

  // UniformAug: transform list must not be empty
  std::shared_ptr<TensorTransform> uniform_aug_op3(new vision::UniformAugment({}, 1));
  EXPECT_NE(uniform_aug_op3, nullptr);
  */
}

TEST_F(MindDataTestPipeline, TestUniformAugWithOps) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugWithOps.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", RandomSampler(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 1;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30, 30}));
  EXPECT_NE(resize_op, nullptr);

  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  EXPECT_NE(random_crop_op, nullptr);

  std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
  EXPECT_NE(center_crop_op, nullptr);

  std::shared_ptr<TensorTransform> uniform_aug_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 2));
  EXPECT_NE(uniform_aug_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({resize_op, uniform_aug_op});
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestVisionOperationName) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVisionOperationName.";

  std::string correct_name;

  // Create object for the tensor op, and check the name
  /* FIXME - Update and move test to IR level
  std::shared_ptr<TensorOperation> random_vertical_flip_op = vision::RandomVerticalFlip(0.5);
  correct_name = "RandomVerticalFlip";
  EXPECT_EQ(correct_name, random_vertical_flip_op->Name());

  // Create object for the tensor op, and check the name
  std::shared_ptr<TensorOperation> softDvpp_decode_resize_jpeg_op = vision::SoftDvppDecodeResizeJpeg({1, 1});
  correct_name = "SoftDvppDecodeResizeJpeg";
  EXPECT_EQ(correct_name, softDvpp_decode_resize_jpeg_op->Name());
  */
}
