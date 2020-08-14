/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

using namespace mindspore::dataset::api;
using mindspore::dataset::BorderType;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestCutMixBatchSuccess1) {
  // Testing CutMixBatch on a batch of CHW images
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  int number_of_classes = 10;
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> hwc_to_chw = vision::HWC2CHW();
  EXPECT_NE(hwc_to_chw, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({hwc_to_chw},{"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> one_hot_op = vision::OneHot(number_of_classes);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op},{"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorOperation> cutmix_batch_op = vision::CutMixBatch(mindspore::dataset::ImageBatchFormat::kNCHW, 1.0, 1.0);
  EXPECT_NE(cutmix_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    MS_LOG(INFO) << "Label shape: " << label->shape();
    EXPECT_EQ(image->shape().AsVector().size() == 4 && batch_size == image->shape()[0] && 3 == image->shape()[1]
              && 32 == image->shape()[2] && 32 == image->shape()[3], true);
    EXPECT_EQ(label->shape().AsVector().size() == 2 && batch_size == label->shape()[0] &&
                number_of_classes == label->shape()[1], true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCutMixBatchSuccess2) {
  // Calling CutMixBatch on a batch of HWC images with default values of alpha and prob
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  int number_of_classes = 10;
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> one_hot_op = vision::OneHot(number_of_classes);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op},{"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorOperation> cutmix_batch_op = vision::CutMixBatch(mindspore::dataset::ImageBatchFormat::kNHWC);
  EXPECT_NE(cutmix_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    MS_LOG(INFO) << "Label shape: " << label->shape();
    EXPECT_EQ(image->shape().AsVector().size() == 4 && batch_size == image->shape()[0] && 32 == image->shape()[1]
              && 32 == image->shape()[2] && 3 == image->shape()[3], true);
    EXPECT_EQ(label->shape().AsVector().size() == 2 && batch_size == label->shape()[0] &&
              number_of_classes == label->shape()[1], true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCutMixBatchFail1) {
  // Must fail because alpha can't be negative
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> one_hot_op = vision::OneHot(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op},{"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorOperation> cutmix_batch_op = vision::CutMixBatch(mindspore::dataset::ImageBatchFormat::kNHWC, -1, 0.5);
  EXPECT_EQ(cutmix_batch_op, nullptr);
}

TEST_F(MindDataTestPipeline, TestCutMixBatchFail2) {
  // Must fail because prob can't be negative
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> one_hot_op = vision::OneHot(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op},{"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorOperation> cutmix_batch_op = vision::CutMixBatch(mindspore::dataset::ImageBatchFormat::kNHWC, 1, -0.5);
  EXPECT_EQ(cutmix_batch_op, nullptr);

}

TEST_F(MindDataTestPipeline, TestCutOut) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> cut_out1 = vision::CutOut(30, 5);
  EXPECT_NE(cut_out1, nullptr);

  std::shared_ptr<TensorOperation> cut_out2 = vision::CutOut(30);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestDecode) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> decode = vision::Decode(true);
  EXPECT_NE(decode, nullptr);

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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestHwcToChw) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> channel_swap = vision::HWC2CHW();
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // check if the image is in NCHW
    EXPECT_EQ(batch_size == image->shape()[0] && 3 == image->shape()[1] && 2268 == image->shape()[2] &&
                4032 == image->shape()[3],
              true);
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMixUpBatchFail1) {
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> one_hot_op = vision::OneHot(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorOperation> mixup_batch_op = vision::MixUpBatch(-1);
  EXPECT_EQ(mixup_batch_op, nullptr);
}

TEST_F(MindDataTestPipeline, TestMixUpBatchSuccess1) {
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> one_hot_op = vision::OneHot(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorOperation> mixup_batch_op = vision::MixUpBatch(0.5);
  EXPECT_NE(mixup_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMixUpBatchSuccess2) {
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> one_hot_op = vision::OneHot(10);
  EXPECT_NE(one_hot_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorOperation> mixup_batch_op = vision::MixUpBatch();
  EXPECT_NE(mixup_batch_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestNormalize) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> normalize = vision::Normalize({121.0, 115.0, 100.0}, {70.0, 68.0, 71.0});
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestPad) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> pad_op1 = vision::Pad({1, 2, 3, 4}, {0}, BorderType::kSymmetric);
  EXPECT_NE(pad_op1, nullptr);

  std::shared_ptr<TensorOperation> pad_op2 = vision::Pad({1}, {1, 1, 1}, BorderType::kEdge);
  EXPECT_NE(pad_op2, nullptr);

  std::shared_ptr<TensorOperation> pad_op3 = vision::Pad({1, 4});
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomAffineFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineFail with invalid params.";

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> affine = vision::RandomAffine({0.0, 0.0}, {});
  EXPECT_EQ(affine, nullptr);
  // Invalid number of values for translate
  affine = vision::RandomAffine({0.0, 0.0}, {1, 1, 1, 1});
  EXPECT_EQ(affine, nullptr);
  // Invalid number of values for shear
  affine = vision::RandomAffine({30.0, 30.0}, {0.0, 0.0}, {2.0, 2.0}, {10.0, 10.0});
  EXPECT_EQ(affine, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomAffineSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineSuccess1 with non-default params.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> affine =
    vision::RandomAffine({30.0, 30.0}, {0.0, 0.0}, {2.0, 2.0}, {10.0, 10.0, 20.0, 20.0});
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomAffineSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineSuccess2 with default params.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> affine = vision::RandomAffine({0.0, 0.0});
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomColor) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColor with non-default params.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> random_color_op_1 = vision::RandomColor(0.0, 0.0);
  EXPECT_NE(random_color_op_1, nullptr);

  std::shared_ptr<TensorOperation> random_color_op_2 = vision::RandomColor(1.0, 0.1);
  EXPECT_EQ(random_color_op_2, nullptr);

  std::shared_ptr<TensorOperation> random_color_op_3 = vision::RandomColor(0.0, 1.1);
  EXPECT_NE(random_color_op_3, nullptr);

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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomColorAdjust) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> random_color_adjust1 = vision::RandomColorAdjust({1.0}, {0.0}, {0.5}, {0.5});
  EXPECT_NE(random_color_adjust1, nullptr);

  std::shared_ptr<TensorOperation> random_color_adjust2 =
    vision::RandomColorAdjust({1.0, 1.0}, {0.0, 0.0}, {0.5, 0.5}, {0.5, 0.5});
  EXPECT_NE(random_color_adjust2, nullptr);

  std::shared_ptr<TensorOperation> random_color_adjust3 =
    vision::RandomColorAdjust({0.5, 1.0}, {0.0, 0.5}, {0.25, 0.5}, {0.25, 0.5});
  EXPECT_NE(random_color_adjust3, nullptr);

  std::shared_ptr<TensorOperation> random_color_adjust4 = vision::RandomColorAdjust();
  EXPECT_NE(random_color_adjust4, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_color_adjust1, random_color_adjust2, random_color_adjust3, random_color_adjust4});
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomPosterizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterize with invalid params.";

  // Create objects for the tensor ops
  // Invalid max > 8
  std::shared_ptr<TensorOperation> posterize = vision::RandomPosterize(1, 9);
  EXPECT_EQ(posterize, nullptr);
  // Invalid min < 1
  posterize = vision::RandomPosterize(0, 8);
  EXPECT_EQ(posterize, nullptr);
  // min > max
  posterize = vision::RandomPosterize(8, 1);
  EXPECT_EQ(posterize, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomPosterizeSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeSuccess1 with non-default params.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> posterize =
    vision::RandomPosterize(1, 4);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomPosterizeSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeSuccess2 with default params.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> posterize = vision::RandomPosterize();
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
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
  std::shared_ptr<TensorOperation> random_sharpness_op_1 = vision::RandomSharpness({0.4, 2.3});
  EXPECT_NE(random_sharpness_op_1, nullptr);

  std::shared_ptr<TensorOperation> random_sharpness_op_2 = vision::RandomSharpness({});
  EXPECT_EQ(random_sharpness_op_2, nullptr);

  std::shared_ptr<TensorOperation> random_sharpness_op_3 = vision::RandomSharpness();
  EXPECT_NE(random_sharpness_op_3, nullptr);

  std::shared_ptr<TensorOperation> random_sharpness_op_4 = vision::RandomSharpness({0.1});
  EXPECT_EQ(random_sharpness_op_4, nullptr);

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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomFlip) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> random_vertical_flip_op = vision::RandomVerticalFlip(0.5);
  EXPECT_NE(random_vertical_flip_op, nullptr);

  std::shared_ptr<TensorOperation> random_horizontal_flip_op = vision::RandomHorizontalFlip(0.5);
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomRotation) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> random_rotation_op = vision::RandomRotation({-180, 180});
  EXPECT_NE(random_rotation_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_rotation_op});
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
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUniformAugWithOps) {
  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, RandomSampler(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 1;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> resize_op = vision::Resize({30, 30});
  EXPECT_NE(resize_op, nullptr);

  std::shared_ptr<TensorOperation> random_crop_op = vision::RandomCrop({28, 28});
  EXPECT_NE(random_crop_op, nullptr);

  std::shared_ptr<TensorOperation> center_crop_op = vision::CenterCrop({16, 16});
  EXPECT_NE(center_crop_op, nullptr);

  std::shared_ptr<TensorOperation> uniform_aug_op = vision::UniformAugment({random_crop_op, center_crop_op}, 2);
  EXPECT_NE(uniform_aug_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({resize_op, uniform_aug_op});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeSucess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::vector<uint8_t> threshold = {10, 100};
  std::shared_ptr<TensorOperation> random_solarize = mindspore::dataset::api::vision::RandomSolarize(threshold);
  EXPECT_NE(random_solarize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_solarize});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeSucess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarize with default params.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorOperation> random_solarize = mindspore::dataset::api::vision::RandomSolarize();
  EXPECT_NE(random_solarize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_solarize});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarize with invalid params.";
  std::vector<uint8_t> threshold = {13, 1};
  std::shared_ptr<TensorOperation> random_solarize = mindspore::dataset::api::vision::RandomSolarize(threshold);
  EXPECT_EQ(random_solarize, nullptr);

  threshold = {1, 2, 3};
  random_solarize = mindspore::dataset::api::vision::RandomSolarize(threshold);
  EXPECT_EQ(random_solarize, nullptr);

  threshold = {1};
  random_solarize = mindspore::dataset::api::vision::RandomSolarize(threshold);
  EXPECT_EQ(random_solarize, nullptr);
}
