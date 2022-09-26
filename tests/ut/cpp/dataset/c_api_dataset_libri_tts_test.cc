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
#include "minddata/dataset/include/dataset/datasets.h"
#include "include/dataset/transforms.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: LibriTTSDataset
/// Description: Test LibriTTSDataset basic usage
/// Expectation: Get correct LibriTTS dataset
TEST_F(MindDataTestPipeline, TestLibriTTSBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLibriTTSBasic.";

  std::string folder_path = datasets_root_path_ + "/testLibriTTSData";
  std::shared_ptr<Dataset> ds = LibriTTS(folder_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  uint64_t i = 0;

  while (row.size() != 0) {
    auto waveform = row["waveform"];
    auto sample_rate = row["sample_rate"];
    auto original_text = row["original_text"];
    auto normalized_text = row["normalized_text"];
    auto speaker_id = row["speaker_id"];
    auto chapter_id = row["chapter_id"];
    auto utterance_id = row["utterance_id"];
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 3);
  iter->Stop();
}

/// Feature: LibriTTSDataset
/// Description: Test LibriTTSDataset with pipeline mode
/// Expectation: Get correct LibriTTS dataset
TEST_F(MindDataTestPipeline, TestLibriTTSBasicWithPipeline) {
  MS_LOG(INFO) << "Doing DataSetOpBatchTest-TestLibriTTSBasicWithPipeline.";

  // Create a LibriTTSDataset Dataset
  std::string folder_path = datasets_root_path_ + "/testLibriTTSData";
  std::shared_ptr<Dataset> ds = LibriTTS(folder_path, "train-clean-100", std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);
  auto op = transforms::PadEnd({1, 500000});
  std::vector<std::string> input_columns = {"waveform"};
  std::vector<std::string> output_columns = {"waveform"};
  ds = ds->Map({op}, input_columns, output_columns);
  EXPECT_NE(ds, nullptr);
  ds = ds->Repeat(5);
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
  std::vector<std::string> expected_original_text = {"good morning", "good afternoon"};
  std::vector<std::string> expected_normalized_text = {"Good morning", "Good afternoon"};
  std::vector<uint32_t> expected_speaker_id = {2506, 2506};
  std::vector<uint32_t> expected_chapter_id = {11267, 11267};
  std::vector<std::string> expected_utterance_id = {"2506_11267_000001_000000", "2506_11267_000002_000000"};
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    auto original_text = row["original_text"];
    auto normalized_text = row["normalized_text"];
    auto sample_rate = row["sample_rate"];
    auto speaker_id = row["speaker_id"];
    auto chapter_id = row["chapter_id"];
    auto utterance_id = row["utterance_id"];

    std::shared_ptr<Tensor> de_original_text;
    ASSERT_OK(Tensor::CreateFromVector(expected_original_text, &de_original_text));
    mindspore::MSTensor fix_original_text =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_original_text));
    EXPECT_MSTENSOR_EQ(original_text, fix_original_text);

    std::shared_ptr<Tensor> de_normalized_text;
    ASSERT_OK(Tensor::CreateFromVector(expected_normalized_text, &de_normalized_text));
    mindspore::MSTensor fix_normalized_text =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_normalized_text));
    EXPECT_MSTENSOR_EQ(normalized_text, fix_normalized_text);

    std::shared_ptr<Tensor> de_expected_speaker_id;
    ASSERT_OK(Tensor::CreateFromVector(expected_speaker_id, &de_expected_speaker_id));
    mindspore::MSTensor fix_expected_speaker_id =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_speaker_id));
    EXPECT_MSTENSOR_EQ(speaker_id, fix_expected_speaker_id);

    std::shared_ptr<Tensor> de_expected_chapter_id;
    ASSERT_OK(Tensor::CreateFromVector(expected_chapter_id, &de_expected_chapter_id));
    mindspore::MSTensor fix_expected_chapter_id =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_chapter_id));
    EXPECT_MSTENSOR_EQ(chapter_id, fix_expected_chapter_id);

    std::shared_ptr<Tensor> de_expected_utterance_id;
    ASSERT_OK(Tensor::CreateFromVector(expected_utterance_id, &de_expected_utterance_id));
    mindspore::MSTensor fix_expected_utterance_id =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_utterance_id));
    EXPECT_MSTENSOR_EQ(utterance_id, fix_expected_utterance_id);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);
  iter->Stop();
}

/// Feature: LibriTTSDataset
/// Description: Test LibriTTSDataset with invalid directory
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLibriTTSError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLibriTTSError.";

  // Create a LibriTTS Dataset with non-existing dataset dir
  std::shared_ptr<Dataset> ds0 = LibriTTS("NotExistFile");
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid LibriTTS input
  EXPECT_EQ(iter0, nullptr);

  // Create a LibriTTS Dataset with invalid string of dataset dir
  std::shared_ptr<Dataset> ds1 = LibriTTS(":*?\"<>|`&;'");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid LibriTTS input
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: LibriTTSDataset
/// Description: Test LibriTTSDataset with Getters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLibriTTSGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLibriTTSGetters.";

  std::string folder_path = datasets_root_path_ + "/testLibriTTSData";
  // Create a LibriTTS Dataset.
  std::shared_ptr<Dataset> ds1 = LibriTTS(folder_path);
  std::shared_ptr<Dataset> ds2 = LibriTTS(folder_path, "train-clean-100");

  std::vector<std::string> column_names = {"waveform",   "sample_rate", "original_text", "normalized_text",
                                           "speaker_id", "chapter_id",  "utterance_id"};

  EXPECT_NE(ds1, nullptr);
  EXPECT_EQ(ds1->GetDatasetSize(), 3);
  EXPECT_EQ(ds1->GetColumnNames(), column_names);

  EXPECT_NE(ds2, nullptr);
  EXPECT_EQ(ds2->GetDatasetSize(), 3);
  EXPECT_EQ(ds2->GetColumnNames(), column_names);
}

/// Feature: LibriTTSDataset
/// Description: Test LibriTTSDataset with invalid usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLibriTTSWithInvalidUsageError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLibriTTSWithInvalidUsageError.";

  std::string folder_path = datasets_root_path_ + "/testLibriTTSData";
  // Create a LibriTTS Dataset.
  std::shared_ptr<Dataset> ds1 = LibriTTS(folder_path, "----");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid LibriTTS input, sampler cannot be nullptr
  EXPECT_EQ(iter1, nullptr);

  std::shared_ptr<Dataset> ds2 = LibriTTS(folder_path, "csacs");
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid LibriTTS input, sampler cannot be nullptr
  EXPECT_EQ(iter2, nullptr);
}

/// Feature: LibriTTSDataset
/// Description: Test LibriTTSDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLibriTTSWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLibriTTSWithNullSamplerError.";

  std::string folder_path = datasets_root_path_ + "/testLibriTTSData";
  // Create a LibriTTS Dataset.
  std::shared_ptr<Dataset> ds = LibriTTS(folder_path, "all", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid LibriTTS input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

/// Feature: LibriTTSDataset
/// Description: Test LibriTTSDataset with SequentialSampler
/// Expectation: Get correct LibriTTS dataset
TEST_F(MindDataTestPipeline, TestLibriTTSSequentialSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLibriTTSSequentialSamplers.";

  std::string folder_path = datasets_root_path_ + "/testLibriTTSData";
  std::shared_ptr<Dataset> ds = LibriTTS(folder_path, "all", std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::string_view original_text_idx, normalized_text_idx, utterance_id_idx;
  uint32_t speaker_idx_id = 0, chapter_idx_id = 0;
  std::vector<std::string> expected_original_text = {"good morning", "good afternoon"};
  std::vector<std::string> expected_normalized_text = {"Good morning", "Good afternoon"};
  std::vector<uint32_t> expected_speaker_id = {2506, 2506};
  std::vector<uint32_t> expected_chapter_id = {11267, 11267};
  std::vector<std::string> expected_utterance_id = {"2506_11267_000001_000000", "2506_11267_000002_000000"};
  uint32_t rate = 0;
  uint64_t i = 0;
  while (row.size() != 0) {
    auto waveform = row["waveform"];
    auto sample_rate = row["sample_rate"];
    auto original_text = row["original_text"];
    auto normalized_text = row["normalized_text"];
    auto speaker_id = row["speaker_id"];
    auto chapter_id = row["chapter_id"];
    auto utterance_id = row["utterance_id"];

    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();

    std::shared_ptr<Tensor> trate;
    ASSERT_OK(Tensor::CreateFromMSTensor(sample_rate, &trate));
    ASSERT_OK(trate->GetItemAt<uint32_t>(&rate, {}));
    EXPECT_EQ(rate, 24000);

    std::shared_ptr<Tensor> de_original_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(original_text, &de_original_text));
    ASSERT_OK(de_original_text->GetItemAt(&original_text_idx, {}));
    std::string s_original_text(original_text_idx);
    EXPECT_STREQ(s_original_text.c_str(), expected_original_text[i].c_str());

    std::shared_ptr<Tensor> de_normalized_text;
    ASSERT_OK(Tensor::CreateFromMSTensor(normalized_text, &de_normalized_text));
    ASSERT_OK(de_normalized_text->GetItemAt(&normalized_text_idx, {}));
    std::string s_normalized_text(normalized_text_idx);
    EXPECT_STREQ(s_normalized_text.c_str(), expected_normalized_text[i].c_str());

    std::shared_ptr<Tensor> de_speaker_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(speaker_id, &de_speaker_id));
    ASSERT_OK(de_speaker_id->GetItemAt<uint32_t>(&speaker_idx_id, {}));
    EXPECT_EQ(speaker_idx_id, expected_speaker_id[i]);

    std::shared_ptr<Tensor> de_chapter_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(chapter_id, &de_chapter_id));
    ASSERT_OK(de_chapter_id->GetItemAt<uint32_t>(&chapter_idx_id, {}));
    EXPECT_EQ(chapter_idx_id, expected_chapter_id[i]);

    std::shared_ptr<Tensor> de_utterance_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(utterance_id, &de_utterance_id));
    ASSERT_OK(de_utterance_id->GetItemAt(&utterance_id_idx, {}));
    std::string s_utterance_id(utterance_id_idx);
    EXPECT_STREQ(s_utterance_id.c_str(), expected_utterance_id[i].c_str());

    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  iter->Stop();
}
