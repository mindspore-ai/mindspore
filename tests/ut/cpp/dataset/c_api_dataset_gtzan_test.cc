/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/dataset/datasets.h"
#include "include/dataset/transforms.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: GTZANDataset
/// Description: Test GTZAN
/// Expectation: Get correct GTZAN dataset
TEST_F(MindDataTestPipeline, TestGTZANBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGTZANBasic.";

  std::string file_path = datasets_root_path_ + "/testGTZANData";
  // Create a GTZAN Dataset
  std::shared_ptr<Dataset> ds = GTZAN(file_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::string_view label_idx;
  uint32_t rate = 0;
  uint64_t i = 0;

  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    auto label = row["label"];
    auto sample_rate = row["sample_rate"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();

    std::shared_ptr<Tensor> trate;
    ASSERT_OK(Tensor::CreateFromMSTensor(sample_rate, &trate));
    ASSERT_OK(trate->GetItemAt<uint32_t>(&rate, {}));
    EXPECT_EQ(rate, 22050);
    MS_LOG(INFO) << "Tensor label rate: " << rate;

    std::shared_ptr<Tensor> de_label;
    ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
    ASSERT_OK(de_label->GetItemAt(&label_idx, {}));
    std::string s_label(label_idx);
    std::string expected("blues");
    EXPECT_STREQ(s_label.c_str(), expected.c_str());
    MS_LOG(INFO) << "Tensor label value: " << label_idx;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: GTZANDataset
/// Description: Test GTZAN with Pipeline
/// Expectation: Get correct GTZAN dataset
TEST_F(MindDataTestPipeline, TestGTZANBasicWithPipeline) {
  MS_LOG(INFO) << "Doing DataSetOpBatchTest-TestGTZANBasicWithPipeline.";

  // Create a GTZANDataset Dataset.
  std::string folder_path = datasets_root_path_ + "/testGTZANData";
  std::shared_ptr<Dataset> ds = GTZAN(folder_path, "all", std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);
  auto op = transforms::PadEnd({1, 50000});
  std::vector<std::string> input_columns = {"waveform"};
  std::vector<std::string> output_columns = {"waveform"};
  ds = ds->Map({op}, input_columns, output_columns);
  EXPECT_NE(ds, nullptr);
  ds = ds->Repeat(10);
  EXPECT_NE(ds, nullptr);
  ds = ds->Batch(5);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  std::vector<uint32_t> expected_rate = {22050, 22050, 22050, 22050, 22050};
  std::vector<std::string> expected_label = {"blues", "blues", "blues", "blues", "blues"};

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    auto label = row["label"];
    auto sample_rate = row["sample_rate"];

    std::shared_ptr<Tensor> de_expected_rate;
    ASSERT_OK(Tensor::CreateFromVector(expected_rate, &de_expected_rate));
    mindspore::MSTensor fix_expected_rate =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_rate));
    EXPECT_MSTENSOR_EQ(sample_rate, fix_expected_rate);

    std::shared_ptr<Tensor> de_expected_label;
    ASSERT_OK(Tensor::CreateFromVector(expected_label, &de_expected_label));
    mindspore::MSTensor fix_expected_label =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_label));
    EXPECT_MSTENSOR_EQ(label, fix_expected_label);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);
  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: GTZANDataset
/// Description: Test GTZAN with invalid directory
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestGTZANError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGTZANError.";

  // Create a GTZAN Dataset with non-existing dataset dir.
  std::shared_ptr<Dataset> ds0 = GTZAN("NotExistFile");
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid GTZAN30k input.
  EXPECT_EQ(iter0, nullptr);

  // Create a GTZAN Dataset with invalid string of dataset dir.
  std::shared_ptr<Dataset> ds1 = GTZAN(":*?\"<>|`&;'");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid GTZAN input.
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: GTZANDataset
/// Description: Test GTZAN with Getters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGTZANGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGTZANGetters.";

  std::string folder_path = datasets_root_path_ + "/testGTZANData";
  // Create a GTZAN Dataset.
  std::shared_ptr<Dataset> ds1 = GTZAN(folder_path);
  std::shared_ptr<Dataset> ds2 = GTZAN(folder_path, "all");
  std::shared_ptr<Dataset> ds3 = GTZAN(folder_path, "valid");

  std::vector<std::string> column_names = {"waveform", "sample_rate", "label"};

  EXPECT_NE(ds1, nullptr);
  EXPECT_EQ(ds1->GetDatasetSize(), 3);
  EXPECT_EQ(ds1->GetColumnNames(), column_names);

  EXPECT_NE(ds2, nullptr);
  EXPECT_EQ(ds2->GetDatasetSize(), 3);
  EXPECT_EQ(ds2->GetColumnNames(), column_names);

  EXPECT_NE(ds3, nullptr);
  EXPECT_EQ(ds3->GetDatasetSize(), 3);
  EXPECT_EQ(ds3->GetColumnNames(), column_names);
}

/// Feature: GTZANDataset
/// Description: Test GTZAN dataset with invalid usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestGTZANWithInvalidUsageError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGTZANWithInvalidUsageError.";

  std::string folder_path = datasets_root_path_ + "/testGTZANData";
  // Create a GTZAN Dataset.
  std::shared_ptr<Dataset> ds1 = GTZAN(folder_path, "----");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();

  EXPECT_EQ(iter1, nullptr);

  std::shared_ptr<Dataset> ds2 = GTZAN(folder_path, "csacs");
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_EQ(iter2, nullptr);
}

/// Feature: GTZANDataset
/// Description: Test GTZAN dataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestGTZANWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGTZANWithNullSamplerError.";

  std::string folder_path = datasets_root_path_ + "/testGTZANData";
  // Create a GTZAN Dataset.
  std::shared_ptr<Dataset> ds = GTZAN(folder_path, "all ", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid GTZAN input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: GTZANDataset
/// Description: Test GTZAN with sequential sampler
/// Expectation: Get correct GTZAN dataset
TEST_F(MindDataTestPipeline, TestGTZANNumSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGTZANWithSequentialSampler.";

  std::string folder_path = datasets_root_path_ + "/testGTZANData";
  // Create a GTZAN Dataset.
  std::shared_ptr<Dataset> ds = GTZAN(folder_path, "all", std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  uint32_t rate = 0;
  uint64_t i = 0;

  while (row.size() != 0) {
    auto waveform = row["waveform"];
    auto sample_rate = row["sample_rate"];

    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();

    std::shared_ptr<Tensor> t_rate;
    ASSERT_OK(Tensor::CreateFromMSTensor(sample_rate, &t_rate));
    ASSERT_OK(t_rate->GetItemAt<uint32_t>(&rate, {}));
    EXPECT_EQ(rate, 22050);
    MS_LOG(INFO) << "Tensor sample rate: " << rate;
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  iter->Stop();
}
