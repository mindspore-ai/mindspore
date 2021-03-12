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

using namespace mindspore::dataset;
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

TEST_F(MindDataTestPipeline, TestManifestBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestBasicWithPipeline.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create two Manifest Dataset
  std::shared_ptr<Dataset> ds1 = Manifest(file_path);
  std::shared_ptr<Dataset> ds2 = Manifest(file_path);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"image"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
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

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestManifestGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestGetters.";

  std::string file_path1 = datasets_root_path_ + "/testManifestData/cpp.json";
  std::string file_path2 = datasets_root_path_ + "/testManifestData/cpp2.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds1 = Manifest(file_path1);
  std::shared_ptr<Dataset> ds2 = Manifest(file_path2);
  std::vector<std::string> column_names = {"image", "label"};

  EXPECT_NE(ds1, nullptr);
  EXPECT_EQ(ds1->GetDatasetSize(), 2);
  EXPECT_EQ(ds1->GetNumClasses(), 2);
  EXPECT_EQ(ds1->GetColumnNames(), column_names);

  EXPECT_NE(ds2, nullptr);
  EXPECT_EQ(ds2->GetDatasetSize(), 4);
  EXPECT_EQ(ds2->GetNumClasses(), 3);

  std::vector<std::pair<std::string, std::vector<int32_t>>> class_index1 = ds1->GetClassIndexing();
  EXPECT_EQ(class_index1.size(), 2);
  EXPECT_EQ(class_index1[0].first, "cat");
  EXPECT_EQ(class_index1[0].second[0], 0);
  EXPECT_EQ(class_index1[1].first, "dog");
  EXPECT_EQ(class_index1[1].second[0], 1);

  std::vector<std::pair<std::string, std::vector<int32_t>>> class_index2 = ds2->GetClassIndexing();
  EXPECT_EQ(class_index2.size(), 3);
  EXPECT_EQ(class_index2[0].first, "cat");
  EXPECT_EQ(class_index2[0].second[0], 0);
  EXPECT_EQ(class_index2[1].first, "dog");
  EXPECT_EQ(class_index2[1].second[0], 1);
  EXPECT_EQ(class_index2[2].first, "flower");
  EXPECT_EQ(class_index2[2].second[0], 2);
}

TEST_F(MindDataTestPipeline, TestManifestDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestDecode.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", std::make_shared<RandomSampler>(), {}, true);
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
    auto shape = image.Shape();
    MS_LOG(INFO) << "Tensor image shape size: " << shape.size();
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_GT(shape.size(), 1);  // Verify decode=true took effect
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::vector<int64_t> expected_label = {111, 222};

  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", std::make_shared<RandomSampler>(), map, true);
  EXPECT_NE(ds, nullptr);

  std::vector<std::pair<std::string, std::vector<int32_t>>> class_index1 = ds->GetClassIndexing();
  EXPECT_EQ(class_index1.size(), 2);
  EXPECT_EQ(class_index1[0].first, "cat");
  EXPECT_EQ(class_index1[0].second[0], 111);
  EXPECT_EQ(class_index1[1].first, "dog");
  EXPECT_EQ(class_index1[1].second[0], 222);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  int32_t label_idx = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();

    std::shared_ptr<Tensor> de_label;
    ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
    de_label->GetItemAt<int32_t>(&label_idx, {});
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
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", std::make_shared<SequentialSampler>(0, 1), {}, true);
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestManifestError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestError.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset with non-existing file
  std::shared_ptr<Dataset> ds0 = Manifest("NotExistFile", "train");
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid Manifest input
  EXPECT_EQ(iter0, nullptr);

  // Create a Manifest Dataset with invalid usage
  std::shared_ptr<Dataset> ds1 = Manifest(file_path, "invalid_usage");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid Manifest input
  EXPECT_EQ(iter1, nullptr);

  // Create a Manifest Dataset with invalid string
  std::shared_ptr<Dataset> ds2 = Manifest(":*?\"<>|`&;'", "train");
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid Manifest input
  EXPECT_EQ(iter2, nullptr);
}

TEST_F(MindDataTestPipeline, TestManifestWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestManifestWithNullSamplerError.";
  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Manifest input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
