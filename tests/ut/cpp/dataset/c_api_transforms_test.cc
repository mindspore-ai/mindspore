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
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for data transforms ops (in alphabetical order)

TEST_F(MindDataTestPipeline, TestComposeSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 3));
  EXPECT_NE(ds, nullptr);
  /* FIXME - Disable until proper external API for Compose is provided
  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> compose = transforms::Compose({vision::Decode(), vision::Resize({777, 777})});
  EXPECT_NE(compose, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
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
    // EXPECT_EQ(image->shape()[0], 777);
    // EXPECT_EQ(image->shape()[1], 777);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
  */
}

TEST_F(MindDataTestPipeline, TestComposeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail with invalid transform.";
  /* FIXME - Disable until proper external API for Compose is provided
  // Resize: Non-positive size value: -1 at element: 0
  // Compose: transform ops must not be null
  std::shared_ptr<TensorOperation> compose1 = transforms::Compose({vision::Decode(), vision::Resize({-1})});
  EXPECT_EQ(compose1, nullptr);

  // Compose: transform ops must not be null
  std::shared_ptr<TensorOperation> compose2 = transforms::Compose({vision::Decode(), nullptr});
  EXPECT_EQ(compose2, nullptr);

  // Compose: transform list must not be empty
  std::shared_ptr<TensorOperation> compose3 = transforms::Compose({});
  EXPECT_EQ(compose3, nullptr);
  */
}

TEST_F(MindDataTestPipeline, TestDuplicateSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDuplicateSuccess.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  transforms::Duplicate duplicate = transforms::Duplicate();

  // Create a Map operation on ds
  ds = ds->Map({duplicate}, {"image"}, {"image", "image_copy"});
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
    // auto image_copy = row["image_copy"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // EXPECT_EQ(*image, *image_copy);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestOneHotSuccess1) {
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

TEST_F(MindDataTestPipeline, TestOneHotSuccess2) {
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

TEST_F(MindDataTestPipeline, TestOneHotFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotFail1 with invalid params.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect num_class
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(0);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid OneHot input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestOneHotFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotFail2 with invalid params.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect num_class
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(-5);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid OneHot input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomApplySuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplySuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 5));
  EXPECT_NE(ds, nullptr);
  /* FIXME - Disable until proper external API for RandomApply is provided
  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> random_apply = transforms::RandomApply({vision::Resize({777, 777})}, 0.8);
  EXPECT_NE(random_apply, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
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
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
  */
}

TEST_F(MindDataTestPipeline, TestRandomApplyFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail with invalid transform.";
  /* FIXME - Disable until proper external API for RandomApply is provided
  // Resize: Non-positive size value: -1 at element: 0
  // RandomApply: transform ops must not be null
  std::shared_ptr<TensorOperation> random_apply1 = transforms::RandomApply({vision::Decode(), vision::Resize({-1})});
  EXPECT_EQ(random_apply1, nullptr);

  // RandomApply: transform ops must not be null
  std::shared_ptr<TensorOperation> random_apply2 = transforms::RandomApply({vision::Decode(), nullptr});
  EXPECT_EQ(random_apply2, nullptr);

  // RandomApply: transform list must not be empty
  std::shared_ptr<TensorOperation> random_apply3 = transforms::RandomApply({});
  EXPECT_EQ(random_apply3, nullptr);

  // RandomApply: Probability has to be between 0 and 1
  std::shared_ptr<TensorOperation> random_apply4 = transforms::RandomApply({vision::Resize({100})}, -1);
  EXPECT_EQ(random_apply4, nullptr);
  */
}

TEST_F(MindDataTestPipeline, TestRandomChoiceSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 3));
  EXPECT_NE(ds, nullptr);
  /* FIXME - Disable until proper external API for RandomChoice is provided
  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> random_choice =
    transforms::RandomChoice({vision::Resize({777, 777}), vision::Resize({888, 888})});
  EXPECT_NE(random_choice, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_choice}, {"image"});
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
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
  */
}

TEST_F(MindDataTestPipeline, TestRandomChoiceFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceFail with invalid transform.";
  /* FIXME - Disable until proper external API for RandomChoice is provided
  // Resize: Non-positive size value: -1 at element: 0
  // RandomChoice: transform ops must not be null
  std::shared_ptr<TensorOperation> random_choice1 = transforms::RandomChoice({vision::Decode(), vision::Resize({-1})});
  EXPECT_EQ(random_choice1, nullptr);

  // RandomChoice: transform ops must not be null
  std::shared_ptr<TensorOperation> random_choice2 = transforms::RandomChoice({vision::Decode(), nullptr});
  EXPECT_EQ(random_choice2, nullptr);

  // RandomChoice: transform list must not be empty
  std::shared_ptr<TensorOperation> random_choice3 = transforms::RandomChoice({});
  EXPECT_EQ(random_choice3, nullptr);
  */
}

TEST_F(MindDataTestPipeline, TestTypeCastSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTypeCastSuccess.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check original data type of dataset
  // auto image = row["image"];
  // std::string ori_type = image->type().ToString();
  // MS_LOG(INFO) << "Original data type: " << ori_type;
  // EXPECT_NE(ori_type.c_str(), "uint8");

  // Manually terminate the pipeline
  iter->Stop();

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> type_cast = std::make_shared<transforms::TypeCast>("uint16");
  EXPECT_NE(type_cast, nullptr);

  // Create a Map operation on ds
  std::shared_ptr<Dataset> ds2 = ds->Map({type_cast}, {"image"});
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);

  // Check current data type of dataset
  iter2->GetNextRow(&row);
  // auto image2 = row["image"];
  // std::string cur_type = image2->type().ToString();
  // MS_LOG(INFO) << "Current data type: " << cur_type;
  // EXPECT_NE(cur_type.c_str(), "uint16");

  // Manually terminate the pipeline
  iter2->Stop();
}

TEST_F(MindDataTestPipeline, TestTypeCastFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTypeCastFail with invalid params.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect data type
  std::shared_ptr<TensorTransform> type_cast = std::make_shared<transforms::TypeCast>("char");
  EXPECT_NE(type_cast, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({type_cast}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid TypeCast input
  EXPECT_EQ(iter, nullptr);
}
