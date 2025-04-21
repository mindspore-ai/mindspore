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

/// Feature: CMUArcticDataset
/// Description: Test CMUArcticDataset basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCMUArcticBasic) {
  MS_LOG(INFO) << "Doing CMUArcticDataTestPipeline-TestCMUArcticBasic.";

  std::string folder_path = datasets_root_path_ + "/testCMUArcticData";
  // Create a CMUArctic Dataset.
  std::shared_ptr<Dataset> ds = CMUArctic(folder_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::string_view transcript_idx, utterance_id_idx;
  uint32_t rate = 0;
  uint64_t i = 0;

  while (row.size() != 0) {
    auto waveform = row["waveform"];
    auto sample_rate = row["sample_rate"];
    auto transcript = row["transcript"];
    auto utterance_id = row["utterance_id"];

    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();

    std::shared_ptr<Tensor> trate;
    ASSERT_OK(Tensor::CreateFromMSTensor(sample_rate, &trate));
    ASSERT_OK(trate->GetItemAt<uint32_t>(&rate, {}));
    MS_LOG(INFO) << "Audio sample rate: " << rate;

    std::shared_ptr<Tensor> de_transcript;
    ASSERT_OK(Tensor::CreateFromMSTensor(transcript, &de_transcript));
    ASSERT_OK(de_transcript->GetItemAt(&transcript_idx, {}));
    std::string s_transcript(transcript_idx);
    MS_LOG(INFO) << "Tensor transcript value: " << transcript_idx;

    std::shared_ptr<Tensor> de_utterance_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(utterance_id, &de_utterance_id));
    ASSERT_OK(de_utterance_id->GetItemAt(&utterance_id_idx, {}));
    std::string s_utterance_id(utterance_id_idx);
    MS_LOG(INFO) << "Tensor utterance_id value: " << utterance_id_idx;
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  iter->Stop();
}

/// Feature: CMUArcticDataset
/// Description: Test CMUArcticDataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCMUArcticBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCMUArcticBasicWithPipeline.";

  // Create a CMUArcticDataset Dataset
  std::string folder_path = datasets_root_path_ + "/testCMUArcticData";
  std::shared_ptr<Dataset> ds = CMUArctic(folder_path, "aew", std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);
  auto op = transforms::PadEnd({1, 50000});;
  std::vector<std::string> input_columns = {"waveform"};
  std::vector<std::string> output_columns = {"waveform"};
  ds = ds->Map({op}, input_columns, output_columns);
  EXPECT_NE(ds, nullptr);
  ds = ds->Repeat(10);
  EXPECT_NE(ds, nullptr);
  ds = ds->Batch(2);
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  std::vector<std::string> expected_utterance = {"Dog.", "Cat."};
  std::vector<std::string> expected_utterance_id = {"a0001", "a0002"};
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    auto transcript = row["transcript"];
    auto utterance_id = row["utterance_id"];

    std::shared_ptr<Tensor> de_expected_transcript;
    ASSERT_OK(Tensor::CreateFromVector(expected_utterance, &de_expected_transcript));
    mindspore::MSTensor fix_expected_transcript =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_transcript));
    EXPECT_MSTENSOR_EQ(transcript, fix_expected_transcript);

    std::shared_ptr<Tensor> de_expected_utterance_id;
    ASSERT_OK(Tensor::CreateFromVector(expected_utterance_id, &de_expected_utterance_id));
    mindspore::MSTensor fix_expected_utterance_id =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_utterance_id));
    EXPECT_MSTENSOR_EQ(utterance_id, fix_expected_utterance_id);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  iter->Stop();
}

/// Feature: CMUArcticDataset
/// Description: Test CMUArcticDataset with non-existing dataset directory
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCMUArcticError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCMUArcticError.";

  // Create a CMUArctic Dataset with non-existing dataset dir
  std::shared_ptr<Dataset> ds0 = CMUArctic("NotExistFile");
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid CMUArctic input
  EXPECT_EQ(iter0, nullptr);

  // Create a CMUArctic Dataset with invalid string of dataset dir
  std::shared_ptr<Dataset> ds1 = CMUArctic(":*?\"<>|`&;'");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid CMUArctic input
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: CMUArcticDataset
/// Description: Test CMUArcticDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestCMUArcticGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCMUArcticGetters.";

  std::string folder_path = datasets_root_path_ + "/testCMUArcticData";
  // Create a CMUArctic Dataset.
  std::shared_ptr<Dataset> ds1 = CMUArctic(folder_path);
  std::shared_ptr<Dataset> ds2 = CMUArctic(folder_path, "aew");

  std::vector<std::string> column_names = {"waveform", "sample_rate", "transcript", "utterance_id"};

  EXPECT_NE(ds1, nullptr);
  EXPECT_EQ(ds1->GetDatasetSize(), 3);
  EXPECT_EQ(ds1->GetColumnNames(), column_names);

  EXPECT_NE(ds2, nullptr);
  EXPECT_EQ(ds2->GetDatasetSize(), 3);
  EXPECT_EQ(ds2->GetColumnNames(), column_names);
}

/// Feature: CMUArcticDataset
/// Description: Test CMUArcticDataset with invalid name
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCMUArcticWithInvalidNameError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCMUArcticWithInvalidNameError.";

  std::string folder_path = datasets_root_path_ + "/testCMUArcticData";
  // Create a CMUArctic Dataset.
  std::shared_ptr<Dataset> ds1 = CMUArctic(folder_path, "----");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid CMUArctic input, invalid name
  EXPECT_EQ(iter1, nullptr);

  std::shared_ptr<Dataset> ds2 = CMUArctic(folder_path, "csacs");
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid CMUArctic input, invalid name
  EXPECT_EQ(iter2, nullptr);
}

/// Feature: CMUArcticDataset
/// Description: Test CMUArcticDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCMUArcticWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCMUArcticWithNullSamplerError.";

  std::string folder_path = datasets_root_path_ + "/testCMUArcticData";
  // Create a CMUArctic Dataset.
  std::shared_ptr<Dataset> ds = CMUArctic(folder_path, "aew", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid CMUArctic input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

/// Feature: CMUArcticDataset
/// Description: Test CMUArcticDataset with SequentialSampler
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestCMUArcticNumSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCMUArcticWithSequentialSampler.";

  std::string folder_path = datasets_root_path_ + "/testCMUArcticData";
  // Create a CMUArctic Dataset.
  std::shared_ptr<Dataset> ds = CMUArctic(folder_path, "aew", std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::string_view transcript_idx, utterance_id_idx;
  std::vector<std::string> expected_utterance = {"Dog.", "Cat."};
  std::vector<std::string> expected_utterance_id = {"a0001", "a0002"};
  uint32_t rate = 0;
  uint64_t i = 0;

  while (row.size() != 0) {
    auto waveform = row["waveform"];
    auto sample_rate = row["sample_rate"];
    auto transcript = row["transcript"];
    auto utterance_id = row["utterance_id"];

    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();

    std::shared_ptr<Tensor> trate;
    ASSERT_OK(Tensor::CreateFromMSTensor(sample_rate, &trate));
    ASSERT_OK(trate->GetItemAt<uint32_t>(&rate, {}));
    EXPECT_EQ(rate, 16000);
    MS_LOG(INFO) << "Tensor sample rate: " << rate;

    std::shared_ptr<Tensor> de_transcript;
    ASSERT_OK(Tensor::CreateFromMSTensor(transcript, &de_transcript));
    ASSERT_OK(de_transcript->GetItemAt(&transcript_idx, {}));
    std::string s_transcript(transcript_idx);
    EXPECT_STREQ(s_transcript.c_str(), expected_utterance[i].c_str());
    MS_LOG(INFO) << "Tensor transcript value: " << transcript_idx;

    std::shared_ptr<Tensor> de_utterance_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(utterance_id, &de_utterance_id));
    ASSERT_OK(de_utterance_id->GetItemAt(&utterance_id_idx, {}));
    std::string s_utterance_id(utterance_id_idx);
    EXPECT_STREQ(s_utterance_id.c_str(), expected_utterance_id[i].c_str());
    MS_LOG(INFO) << "Tensor utterance_id value: " << utterance_id_idx;
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  iter->Stop();
}
