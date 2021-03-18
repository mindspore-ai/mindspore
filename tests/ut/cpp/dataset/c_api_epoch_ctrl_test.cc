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

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestEpochCtrl : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestEpochCtrl, TestAutoInjectEpoch) {
  MS_LOG(INFO) << "Doing MindDataTestEpochCtrl-TestAutoInjectEpoch.";

  int32_t img_class[4] = {0, 1, 2, 3};
  int32_t num_epochs = 2 + std::rand() % 3;
  int32_t sampler_size = 44;
  int32_t class_size = 11;
  MS_LOG(INFO) << "num_epochs: " << num_epochs;

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, sampler_size));
  ds = ds->SetNumWorkers(2);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect a valid iterator
  ASSERT_NE(iter, nullptr);

  uint64_t i = 0;
  std::unordered_map<std::string, mindspore::MSTensor> row;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    // Iterate the dataset and get each row
    iter->GetNextRow(&row);

    while (row.size() != 0) {
      auto label = row["label"];
      std::shared_ptr<Tensor> de_label;
      int64_t label_value;
      ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
      de_label->GetItemAt(&label_value, {0});
      EXPECT_TRUE(img_class[(i % sampler_size) / class_size] == label_value);

      iter->GetNextRow(&row);
      i++;
    }
  }

  EXPECT_EQ(i, sampler_size * num_epochs);

  // Try to fetch data beyond the specified number of epochs.
  iter->GetNextRow(&row);
  EXPECT_EQ(row.size(), 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestEpochCtrl, TestEpoch) {
  MS_LOG(INFO) << "Doing MindDataTestEpochCtrl-TestEpoch.";

  int32_t num_epochs = 1 + std::rand() % 4;
  int32_t sampler_size = 7;
  MS_LOG(INFO) << "num_epochs: " << num_epochs;

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(0, sampler_size));
  ds = ds->SetNumWorkers(3);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect a valid iterator
  ASSERT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  uint64_t i = 0;
  std::unordered_map<std::string, mindspore::MSTensor> row;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    iter->GetNextRow(&row);
    while (row.size() != 0) {
      auto label = row["label"];
      std::shared_ptr<Tensor> de_label;
      int64_t label_value;
      ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
      de_label->GetItemAt(&label_value, {0});
      EXPECT_TRUE(label_value >= 0 && label_value <= 3);

      iter->GetNextRow(&row);
      i++;
    }
  }

  // Verify correct number of rows fetched
  EXPECT_EQ(i, sampler_size * num_epochs);

  // Try to fetch data beyond the specified number of epochs.
  iter->GetNextRow(&row);
  EXPECT_EQ(row.size(), 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestEpochCtrl, TestRepeatEpoch) {
  MS_LOG(INFO) << "Doing MindDataTestEpochCtrl-TestRepeatEpoch.";

  int32_t num_epochs = 2 + std::rand() % 5;
  int32_t num_repeats = 3;
  int32_t sampler_size = 7;
  MS_LOG(INFO) << "num_epochs: " << num_epochs;

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(0, sampler_size));
  ds = ds->SetNumWorkers(3);
  ds = ds->Repeat(num_repeats);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect a valid iterator
  ASSERT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  uint64_t i = 0;
  std::unordered_map<std::string, mindspore::MSTensor> row;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    iter->GetNextRow(&row);
    while (row.size() != 0) {
      auto label = row["label"];
      std::shared_ptr<Tensor> de_label;
      int64_t label_value;
      ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
      de_label->GetItemAt(&label_value, {0});
      EXPECT_TRUE(label_value >= 0 && label_value <= 3);

      iter->GetNextRow(&row);
      i++;
    }
  }

  // Verify correct number of rows fetched
  EXPECT_EQ(i, sampler_size * num_repeats * num_epochs);

  // Try to fetch data beyond the specified number of epochs.
  iter->GetNextRow(&row);
  EXPECT_EQ(row.size(), 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestEpochCtrl, TestRepeatRepeatEpoch) {
  MS_LOG(INFO) << "Doing MindDataTestEpochCtrl-TestRepeatRepeatEpoch.";

  int32_t num_epochs = 1 + std::rand() % 5;
  int32_t num_repeats[2] = {2, 3};
  int32_t sampler_size = 11;
  MS_LOG(INFO) << "num_epochs: " << num_epochs;

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(5, sampler_size));
  ds = ds->Repeat(num_repeats[0]);
  ds = ds->Repeat(num_repeats[1]);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect a valid iterator
  ASSERT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  uint64_t i = 0;
  std::unordered_map<std::string, mindspore::MSTensor> row;

  for (int epoch = 0; epoch < num_epochs; epoch++) {
    iter->GetNextRow(&row);
    while (row.size() != 0) {
      auto label = row["label"];
      std::shared_ptr<Tensor> de_label;
      int64_t label_value;
      ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
      de_label->GetItemAt(&label_value, {0});
      EXPECT_TRUE(label_value >= 0 && label_value <= 3);

      iter->GetNextRow(&row);
      i++;
    }
  }

  // Verify correct number of rows fetched
  EXPECT_EQ(i, sampler_size * num_repeats[0] * num_repeats[1] * num_epochs);

  // Try to fetch data beyond the specified number of epochs.
  iter->GetNextRow(&row);
  EXPECT_EQ(row.size(), 2);

  // Manually terminate the pipeline
  iter->Stop();
}
