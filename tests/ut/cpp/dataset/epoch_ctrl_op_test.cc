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
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include <memory>

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

std::shared_ptr<ImageFolderOp> ImageFolder(int64_t num_works, int64_t rows, int64_t conns, std::string path,
                                           bool shuf = false, std::shared_ptr<Sampler> sampler = nullptr,
                                           std::map<std::string, int32_t> map = {}, bool decode = false);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

class MindDataTestEpochCtrlOp : public UT::DatasetOpTesting {
public:
  void SetUp() override {
    DatasetOpTesting::SetUp();
    folder_path = datasets_root_path_ + "/testPK/data";

    GlobalInit();

    // Start with an empty execution tree
    my_tree_ = std::make_shared<ExecutionTree>();

    my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false)});
    rc = my_tree_->Prepare();
    EXPECT_TRUE(rc.IsOk());
    rc = my_tree_->Launch();
    EXPECT_TRUE(rc.IsOk());

    // Start the loop of reading tensors from our pipeline
    DatasetIterator di(my_tree_);
    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    int32_t i = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      golden_imgs.append((char *) tensor_map["image"]->GetBuffer(), (int64_t) tensor_map["image"]->Size());
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
  }

  std::shared_ptr<ExecutionTree> my_tree_;
  Status rc;
  std::string golden_imgs;
  std::string folder_path;
  int32_t label = 0;
  std::string result;
  int32_t img_class[4] = {0, 1, 2, 3};

};

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_AutoInjectEpoch) {
  MS_LOG(WARNING) << "Doing ImageFolder_AutoInjectEpoch.";

  int32_t num_epoch = 2 + std::rand() % 5;

  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false)});
  rc = my_tree_->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "num_epoch: " << num_epoch;
  std::string golden = golden_imgs;

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree_);
  TensorMap tensor_map;
  uint64_t i = 0;
  for (int epoch = 0; epoch < num_epoch; epoch++) {
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_map["image"]->GetBuffer(), (int64_t) tensor_map["image"]->Size());
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
    EXPECT_TRUE(result == golden);
    result.clear();

    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
  }

  EXPECT_TRUE(i == 44 * num_epoch);

  // Try to fetch data beyond the specified number of epochs.
  rc = di.GetNextAsMap(&tensor_map);
  EXPECT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Epoch) {
  MS_LOG(WARNING) << "Doing ImageFolder_Epoch.";

  int32_t num_epoch = 2 + std::rand() % 5;

  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false)});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "num_epoch: " << num_epoch;
  std::string golden = golden_imgs;

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree_);
  TensorMap tensor_map;
  uint64_t i = 0;
  for (int epoch = 0; epoch < num_epoch; epoch++) {
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_map["image"]->GetBuffer(), (int64_t) tensor_map["image"]->Size());
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
    EXPECT_TRUE(result == golden);
    result.clear();

    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
  }

  EXPECT_TRUE(i == 44 * num_epoch);

  // Try to fetch data beyond the specified number of epochs.
  rc = di.GetNextAsMap(&tensor_map);
  EXPECT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Repeat_Epoch) {
  MS_LOG(WARNING) << "Doing ImageFolder_Repeat_Epoch.";

  int32_t num_epoch = 2 + std::rand() % 5;

  int32_t num_repeats = 2;
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(num_repeats).Build(&repeat_op);
  EXPECT_TRUE(rc.IsOk());

  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false), repeat_op});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". num_repeat: " << num_repeats;
  std::string golden = golden_imgs;
  for (int i = 1; i < num_repeats; i++) {
    golden += golden_imgs;
  }

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree_);
  TensorMap tensor_map;
  uint64_t i = 0;
  for (int epoch = 0; epoch < num_epoch; epoch++) {
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_map["image"]->GetBuffer(), (int64_t) tensor_map["image"]->Size());
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
    EXPECT_TRUE(result == golden);
    result.clear();

    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
  }

  EXPECT_TRUE(i == 44 * num_repeats * num_epoch);

  // Try to fetch data beyond the specified number of epochs.
  rc = di.GetNextAsMap(&tensor_map);
  EXPECT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Repeat_Repeat_Epoch) {
  MS_LOG(WARNING) << "Doing ImageFolder_Repeat_Repeat_Epoch.";

  int32_t num_epoch = 2 + std::rand() % 5;

  int32_t num_repeats = 2;
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(num_repeats).Build(&repeat_op);
  EXPECT_TRUE(rc.IsOk());

  int32_t num_repeats_2 = 3;
  std::shared_ptr<RepeatOp> repeat_op_2;
  rc = RepeatOp::Builder(num_repeats_2).Build(&repeat_op_2);
  EXPECT_TRUE(rc.IsOk());

  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false), repeat_op, repeat_op_2});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". num_repeat: " << num_repeats << ". num_repeat_2: " << num_repeats_2;
  std::string golden;
  for (int j = 0; j < num_repeats_2; j++) {
    for (int i = 0; i < num_repeats; i++) {
      golden += golden_imgs;
    }
  }

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree_);
  TensorMap tensor_map;
  uint64_t i = 0;
  for (int epoch = 0; epoch < num_epoch; epoch++) {
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_map["image"]->GetBuffer(), (int64_t) tensor_map["image"]->Size());
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
    EXPECT_EQ(result.size(), golden.size());
    EXPECT_TRUE(result == golden);
    result.clear();

    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
  }

  EXPECT_EQ(i, 44 * num_epoch * num_repeats * num_repeats_2);

  // Try to fetch data beyond the specified number of epochs.
  rc = di.GetNextAsMap(&tensor_map);
  EXPECT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Epoch_Inf) {
  MS_LOG(WARNING) << "Doing ImageFolder_Epoch_Inf.";

  // if num_epoch == -1, it means infinity.
  int32_t num_epoch = -1;
  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false)});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree_);
  TensorMap tensor_map;
  uint64_t i = 0;

  // For this test, we stop at stop_at_epoch number.
  int32_t stop_at_epoch = 2 + std::rand() % 6;
  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". Stop at epoch: " << stop_at_epoch;
  for (int epoch = 0; epoch < stop_at_epoch; epoch++) {
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_map["image"]->GetBuffer(), (int64_t) tensor_map["image"]->Size());
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
    EXPECT_EQ(result, golden_imgs);
    result.clear();
    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
  }
  EXPECT_TRUE(i == 44 * stop_at_epoch);
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Repeat_Repeat_Epoch_Inf) {
  MS_LOG(WARNING) << "Doing ImageFolder_Repeat_Epoch_Inf.";

  // if num_epoch == -1, it means infinity.
  int32_t num_epoch = -1;

  int32_t num_repeats = 2;
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(num_repeats).Build(&repeat_op);
  EXPECT_TRUE(rc.IsOk());

  int32_t num_repeats_2 = 3;
  std::shared_ptr<RepeatOp> repeat_op_2;
  rc = RepeatOp::Builder(num_repeats_2).Build(&repeat_op_2);
  EXPECT_TRUE(rc.IsOk());

  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false), repeat_op, repeat_op_2});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". num_repeat: " << num_repeats << ". num_repeat_2: " << num_repeats_2;
  std::string golden;
  for (int j = 0; j < num_repeats_2; j++) {
    for (int i = 0; i < num_repeats; i++) {
      golden += golden_imgs;
    }
  }

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree_);
  TensorMap tensor_map;
  uint64_t i = 0;

  // For this test, we stop at stop_at_epoch number.
  int32_t stop_at_epoch = 2 + std::rand() % 6;
  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". Stop at epoch: " << stop_at_epoch;
  for (int epoch = 0; epoch < stop_at_epoch; epoch++) {
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_map["image"]->GetBuffer(), (int64_t) tensor_map["image"]->Size());
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
    EXPECT_EQ(result, golden);
    result.clear();
    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
  }
  EXPECT_TRUE(i == 44 * stop_at_epoch * num_repeats * num_repeats_2);
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Epoch_ChildItr) {
  MS_LOG(WARNING) << "Doing ImageFolder_Epoch_ChildItr.";

  int32_t num_epoch = 2 + std::rand() % 5;
  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false)});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "num_epoch: " << num_epoch;

  // Start the loop of reading tensors from our pipeline
  ChildIterator ci(my_tree_->root().get(), 0, 0);
  TensorRow tensor_row;
  uint64_t total_sample = 0;
  uint64_t i = 0;
  uint32_t epoch = 0;
  rc = ci.FetchNextTensorRow(&tensor_row);
  EXPECT_TRUE(rc.IsOk());
  while(!ci.eof_handled()) {
    i = 0;
    while (tensor_row.size() != 0) {
      tensor_row[1]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_row[0]->GetBuffer(), (int64_t) tensor_row[0]->Size());
      rc = ci.FetchNextTensorRow(&tensor_row);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }

    epoch++;
    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
    EXPECT_TRUE(result == golden_imgs);
    result.clear();
    EXPECT_TRUE(i == 44);
    total_sample += i;
    rc = ci.FetchNextTensorRow(&tensor_row);
    EXPECT_TRUE(rc.IsOk());
  }
  EXPECT_TRUE(total_sample == 44 * num_epoch);

  // Try to fetch data after last epoch ends.
  rc = ci.FetchNextTensorRow(&tensor_row);
  EXPECT_TRUE(tensor_row.empty());
  EXPECT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Repeat_Epoch_ChildItr) {
  MS_LOG(WARNING) << "Doing ImageFolder_Repeat_Epoch_ChildItr.";

  int32_t num_epoch = 2 + std::rand() % 5;

  int32_t num_repeats = 2;
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(num_repeats).Build(&repeat_op);
  EXPECT_TRUE(rc.IsOk());

  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false), repeat_op});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". num_repeat: " << num_repeats;
  std::string golden;
  for (int i = 0; i < num_repeats; i++) {
    golden += golden_imgs;
  }

  // Start the loop of reading tensors from our pipeline
  ChildIterator ci(my_tree_->root().get(), 0, 0);
  TensorRow tensor_row;
  uint64_t total_sample = 0;
  uint64_t i = 0;
  uint32_t epoch = 0;
  rc = ci.FetchNextTensorRow(&tensor_row);
  EXPECT_TRUE(rc.IsOk());
  while(!ci.eof_handled()) {
    i = 0;
    while (tensor_row.size() != 0) {
      tensor_row[1]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_row[0]->GetBuffer(), (int64_t) tensor_row[0]->Size());
      rc = ci.FetchNextTensorRow(&tensor_row);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }

    epoch++;
    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
    EXPECT_TRUE(result == golden);
    result.clear();
    EXPECT_TRUE(i == 44 * num_repeats);
    total_sample += i;
    rc = ci.FetchNextTensorRow(&tensor_row);
    EXPECT_TRUE(rc.IsOk());
  }
  EXPECT_TRUE(total_sample == 44 * num_epoch * num_repeats);

  // Try to fetch data after last epoch ends.
  rc = ci.FetchNextTensorRow(&tensor_row);
  EXPECT_TRUE(tensor_row.empty());
  EXPECT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Repeat_Repeat_Epoch_ChildItr) {
  MS_LOG(WARNING) << "Doing ImageFolder_Repeat_Repeat_Epoch_ChildItr.";

  int32_t num_epoch = 2 + std::rand() % 5;

  int32_t num_repeats = 2;
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(num_repeats).Build(&repeat_op);
  EXPECT_TRUE(rc.IsOk());

  int32_t num_repeats_2 = 3;
  std::shared_ptr<RepeatOp> repeat_op_2;
  rc = RepeatOp::Builder(num_repeats_2).Build(&repeat_op_2);
  EXPECT_TRUE(rc.IsOk());

  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false), repeat_op, repeat_op_2});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". num_repeat: " << num_repeats << ". num_repeat_2: " << num_repeats_2;
  std::string golden;
  for (int j = 0; j < num_repeats_2; j++) {
    for (int i = 0; i < num_repeats; i++) {
      golden += golden_imgs;
    }
  }

  // Start the loop of reading tensors from our pipeline
  ChildIterator ci(my_tree_->root().get(), 0, 0);
  TensorRow tensor_row;
  uint64_t total_sample = 0;
  uint64_t i = 0;
  uint32_t epoch = 0;
  rc = ci.FetchNextTensorRow(&tensor_row);
  EXPECT_TRUE(rc.IsOk());
  while(!ci.eof_handled()) {
    i = 0;
    while (tensor_row.size() != 0) {
      tensor_row[1]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_row[0]->GetBuffer(), (int64_t) tensor_row[0]->Size());
      rc = ci.FetchNextTensorRow(&tensor_row);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }

    epoch++;
    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
    EXPECT_TRUE(result == golden);
    result.clear();
    EXPECT_TRUE(i == 44 * num_repeats * num_repeats_2);
    total_sample += i;
    rc = ci.FetchNextTensorRow(&tensor_row);
    EXPECT_TRUE(rc.IsOk());
  }
  EXPECT_TRUE(total_sample == 44 * num_epoch * num_repeats * num_repeats_2);

  // Try to fetch data after last epoch ends.
  rc = ci.FetchNextTensorRow(&tensor_row);
  EXPECT_TRUE(tensor_row.empty());
  EXPECT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Epoch_Inf_ChildItr) {
  MS_LOG(WARNING) << "Doing ImageFolder_Epoch_Inf_ChildItr.";

  // if num_epoch == -1, it means infinity.
  int32_t num_epoch = -1;
  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false)});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  ChildIterator ci(my_tree_->root().get(), 0, 0);
  TensorRow tensor_row;
  uint64_t i = 0;

  // For this test, we stop at a random number between 0 - 100 epochs.
  int32_t stop_at_epoch = 2 + std::rand() % 5;
  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". Stop at epoch: " << stop_at_epoch;
  for (int epoch = 0; epoch < stop_at_epoch; epoch++) {
    rc = ci.FetchNextTensorRow(&tensor_row);
    EXPECT_TRUE(rc.IsOk());
    while (tensor_row.size() != 0) {
      tensor_row[1]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_row[0]->GetBuffer(), (int64_t) tensor_row[0]->Size());
      rc = ci.FetchNextTensorRow(&tensor_row);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
    EXPECT_TRUE(result == golden_imgs);
    result.clear();
    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
  }
  EXPECT_TRUE(i == 44 * stop_at_epoch);
}

TEST_F(MindDataTestEpochCtrlOp, ImageFolder_Repeat_Epoch_Inf_ChildItr) {
  MS_LOG(WARNING) << "Doing ImageFolder_Repeat_Epoch_Inf_ChildItr.";

  // if num_epoch == -1, it means infinity.
  int32_t num_epoch = -1;
  int32_t num_repeats = 2;
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(num_repeats).Build(&repeat_op);
  EXPECT_TRUE(rc.IsOk());

  my_tree_ = Build({ImageFolder(2, 2, 32, folder_path, false), repeat_op});
  rc = my_tree_->Prepare(num_epoch);
  EXPECT_TRUE(rc.IsOk());
  rc = my_tree_->Launch();
  EXPECT_TRUE(rc.IsOk());

  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". num_repeat: " << num_repeats;
  std::string golden;
  for (int i = 0; i < num_repeats; i++) {
    golden += golden_imgs;
  }

  // Start the loop of reading tensors from our pipeline
  ChildIterator ci(my_tree_->root().get(), 0, 0);
  TensorRow tensor_row;
  uint64_t i = 0;

  // For this test, we stop at a random number between 0 - 100 epochs.
  int32_t stop_at_epoch = 2 + std::rand() % 5;
  MS_LOG(DEBUG) << "num_epoch: " << num_epoch << ". Stop at epoch: " << stop_at_epoch;
  for (int epoch = 0; epoch < stop_at_epoch; epoch++) {
    rc = ci.FetchNextTensorRow(&tensor_row);
    EXPECT_TRUE(rc.IsOk());
    while (tensor_row.size() != 0) {
      tensor_row[1]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      EXPECT_TRUE(img_class[(i % 44) / 11] == label);
      // Dump all the image into string, to be used as a comparison later.
      result.append((char *) tensor_row[0]->GetBuffer(), (int64_t) tensor_row[0]->Size());
      rc = ci.FetchNextTensorRow(&tensor_row);
      EXPECT_TRUE(rc.IsOk());
      i++;
    }
    EXPECT_TRUE(result == golden);
    result.clear();
    MS_LOG(DEBUG) << "Current epoch: " << epoch << ". Sample count: " << i;
  }
  EXPECT_TRUE(i == 44 * stop_at_epoch * num_repeats);
}
