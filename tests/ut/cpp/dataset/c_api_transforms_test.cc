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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> decode_op(new vision::Decode());
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({777, 777}));
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    EXPECT_EQ(image.Shape()[0], 777);
    EXPECT_EQ(image.Shape()[1], 777);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestComposeFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail1 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
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

TEST_F(MindDataTestPipeline, TestComposeFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail2 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Compose: transform ops must not be null
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> compose(new transforms::Compose({decode_op, nullptr}));

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Compose parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestComposeFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail3 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
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

TEST_F(MindDataTestPipeline, TestDuplicateSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDuplicateSuccess.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
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
    auto image = row["image"];
    auto image_copy = row["image_copy"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_MSTENSOR_EQ(image, image_copy);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestOneHotSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotSuccess1.";
  // Testing CutMixBatch on a batch of CHW images
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  int number_of_classes = 10;
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> hwc_to_chw = std::make_shared<vision::HWC2CHW>();

  // Create a Map operation on ds
  ds = ds->Map({hwc_to_chw}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(number_of_classes);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNCHW, 1.0, 1.0);

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
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    EXPECT_EQ(image.Shape().size() == 4 && batch_size == image.Shape()[0] && 3 == image.Shape()[1] &&
                32 == image.Shape()[2] && 32 == image.Shape()[3],
              true);
    EXPECT_EQ(label.Shape().size() == 2 && batch_size == label.Shape()[0] && number_of_classes == label.Shape()[1],
              true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestOneHotSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotSuccess2.";
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

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>(2.0);

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect num_class
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(0);

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
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect num_class
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(-5);

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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize_op = vision::Resize({777, 777});
  auto random_apply = transforms::RandomApply({resize_op}, 0.8);

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
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomApplyFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail1 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Resize: Non-positive size value: -1 at element: 0
  // RandomApply: transform ops must not be null
  auto decode_op = vision::Decode();
  auto resize_op = vision::Resize({-1});
  auto random_apply = transforms::RandomApply({decode_op, resize_op});

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomApplyFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail2 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomApply: transform ops must not be null
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> random_apply(new transforms::RandomApply({decode_op, nullptr}));

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomApplyFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail3 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomApply: Probability has to be between 0 and 1
  auto resize_op = vision::Resize({100});
  auto random_apply = transforms::RandomApply({resize_op}, -1);

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (Probability has to be between 0 and 1)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomApplyFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail4 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomApply: transform list must not be empty
  std::vector<std::shared_ptr<TensorTransform>> list = {};
  auto random_apply = transforms::RandomApply(list);

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform list must not be empty)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomChoiceSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> resize_op1(new vision::Resize({777, 777}));
  std::shared_ptr<TensorTransform> resize_op2(new vision::Resize({888, 888}));
  auto random_choice = transforms::RandomChoice({resize_op1, resize_op2});

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
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomChoiceFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceFail1 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  RandomSampler sampler = RandomSampler(false, 10);
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", sampler);
  EXPECT_NE(ds, nullptr);

  // Resize: Non-positive size value: -1 at element: 0
  // RandomChoice: transform ops must not be null
  auto decode_op = vision::Decode();
  auto resize_op = vision::Resize({-1});
  auto random_choice = transforms::RandomChoice({decode_op, resize_op});

  // Create a Map operation on ds
  ds = ds->Map({random_choice}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomChoiceFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceFail2 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomChoice: transform ops must not be null
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> random_choice(new transforms::RandomApply({decode_op, nullptr}));

  // Create a Map operation on ds
  ds = ds->Map({random_choice}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomChoiceFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceFail3 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomChoice: transform list must not be empty
  std::vector<std::shared_ptr<TensorTransform>> list = {};
  auto random_choice = transforms::RandomChoice(list);

  // Create a Map operation on ds
  ds = ds->Map({random_choice}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform list must not be empty)
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTypeCastSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTypeCastSuccess.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check original data type of dataset
  auto image = row["image"];
  auto ori_type = image.DataType();
  MS_LOG(INFO) << "Original data type id: " << ori_type;
  EXPECT_EQ(ori_type, mindspore::DataType(mindspore::TypeId::kNumberTypeUInt8));

  // Manually terminate the pipeline
  iter->Stop();

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> type_cast = std::make_shared<transforms::TypeCast>("uint16");

  // Create a Map operation on ds
  std::shared_ptr<Dataset> ds2 = ds->Map({type_cast}, {"image"});
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);

  // Check current data type of dataset
  iter2->GetNextRow(&row);
  auto image2 = row["image"];
  auto cur_type = image2.DataType();
  MS_LOG(INFO) << "Current data type id: " << cur_type;
  EXPECT_EQ(cur_type, mindspore::DataType(mindspore::TypeId::kNumberTypeUInt16));

  // Manually terminate the pipeline
  iter2->Stop();
}

TEST_F(MindDataTestPipeline, TestTypeCastFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTypeCastFail with invalid params.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect data type
  std::shared_ptr<TensorTransform> type_cast = std::make_shared<transforms::TypeCast>("char");

  // Create a Map operation on ds
  ds = ds->Map({type_cast}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid TypeCast input
  EXPECT_EQ(iter, nullptr);
}
