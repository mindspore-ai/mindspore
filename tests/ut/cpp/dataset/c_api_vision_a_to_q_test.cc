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

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API A to Q TensorTransform Operations (in alphabetical order)

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
