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

using namespace mindspore::dataset::api;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestManifestBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestBasic.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path);
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

TEST_F(MindDataTestPipeline, TestManifestDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestDecode.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", nullptr, {}, true);
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
    auto shape = image->shape();
    MS_LOG(INFO) << "Tensor image shape size: " << shape.Size();
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    EXPECT_GT(shape.Size(), 1); // Verify decode=true took effect
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestManifestEval) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestEval.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path, "eval");
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestManifestClassIndex) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestClassIndex.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  std::map<std::string, int32_t> map;
  map["cat"] = 111;  // forward slash is not good, but we need to add this somewhere, also in windows, its a '\'
  map["dog"] = 222;  // forward slash is not good, but we need to add this somewhere, also in windows, its a '\'
  map["wrong folder name"] = 1234;  // this is skipped
  std::vector<int> expected_label = {111, 222};

  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", nullptr, map, true);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  int32_t label_idx = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    row["label"]->GetItemAt<int32_t>(&label_idx, {});
    MS_LOG(INFO) << "Tensor label value: " << label_idx;
    auto label_it = std::find(expected_label.begin(), expected_label.end(), label_idx);
    EXPECT_NE(label_it, expected_label.end());
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestManifestNumSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestNumSamplers.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", SequentialSampler(0, 1), {}, true);
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestManifestError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestError.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset with not exist file
  std::shared_ptr<Dataset> ds0 = Manifest("NotExistFile", "train");
  EXPECT_EQ(ds0, nullptr);

  // Create a Manifest Dataset with invalid usage
  std::shared_ptr<Dataset> ds1 = Manifest(file_path, "invalid_usage");
  EXPECT_EQ(ds1, nullptr);
}
