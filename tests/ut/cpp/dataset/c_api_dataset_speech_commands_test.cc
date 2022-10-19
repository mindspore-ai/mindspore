/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Test SpeechCommands dataset.
/// Description: Read data from a single file.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSpeechCommandsDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDataset.";
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "all", std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  MS_LOG(INFO) << "iter->GetNextRow(&row) OK";

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  EXPECT_NE(row.find("speaker_id"), row.end());
  EXPECT_NE(row.find("utterance_number"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Test SpeechCommands dataset.
/// Description: Test SpeechCommands dataset in pipeline.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSpeechCommandsDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDatasetWithPipeline.";

  // Create two SpeechCommands Dataset.
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds1 = SpeechCommands(folder_path, "all", std::make_shared<RandomSampler>(false, 1));
  std::shared_ptr<Dataset> ds2 = SpeechCommands(folder_path, "all", std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds.
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds.
  std::vector<std::string> column_project = {"waveform", "sample_rate", "label", "speaker_id", "utterance_number"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds.
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  EXPECT_NE(row.find("speaker_id"), row.end());
  EXPECT_NE(row.find("utterance_number"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: TestSpeechCommandsDatasetIteratorOneColumn.
/// Description: Test iterator of SpeechCommands dataset with only the "waveform" column.
/// Expectation: Get correct data.
TEST_F(MindDataTestPipeline, TestSpeechCommandsDatasetIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDatasetIteratorOneColumn.";
  // Create a  SpeechCommands dataset
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "all", std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "waveform" column and drop others
  std::vector<std::string> columns = {"waveform"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expect_shape = {1, 1, 16000};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "waveform shape:" << v.Shape();
      EXPECT_EQ(expect_shape, v.Shape());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: TestSpeechCommandsDatasetIteratorWrongColumn.
/// Description: Test iterator of SpeechCommandsDataset with wrong column.
/// Expectation: Error message is logged, and CreateIterator for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSpeechCommandsDatasetIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDatasetIteratorWrongColumn.";
  // Create a  SpeechCommands dataset
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "all", std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test SpeechCommands dataset.
/// Description: Get the size of SpeechCommands dataset.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSpeechCommandsGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsGetDatasetSize.";

  // Create a SpeechCommands Dataset.
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "all");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
}

/// Feature: Test SpeechCommands dataset.
/// Description: Getter functions.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSpeechCommandsGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsMixGetter.";
  // Create a SpeechCommands Dataset.
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";

  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetDatasetSize(), 3);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"waveform", "sample_rate", "label", "speaker_id", "utterance_number"};
  EXPECT_EQ(types.size(), 5);
  EXPECT_EQ(types[0].ToString(), "float32");
  EXPECT_EQ(types[1].ToString(), "int32");
  EXPECT_EQ(types[2].ToString(), "string");
  EXPECT_EQ(types[3].ToString(), "string");
  EXPECT_EQ(types[4].ToString(), "int32");

  EXPECT_EQ(shapes.size(), 5);
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(shapes[2].ToString(), "<>");
  EXPECT_EQ(shapes[3].ToString(), "<>");
  EXPECT_EQ(shapes[4].ToString(), "<>");
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: Test SpeechCommands dataset.
/// Description: Test usage "train".
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSpeechCommandsUsageTrain) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDataset.";
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "train", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  MS_LOG(INFO) << "iter->GetNextRow(&row) OK";

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  EXPECT_NE(row.find("speaker_id"), row.end());
  EXPECT_NE(row.find("utterance_number"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Test SpeechCommands dataset.
/// Description: Test usage "test".
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSpeechCommandsUsageTest) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDataset.";
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "test", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  MS_LOG(INFO) << "iter->GetNextRow(&row) OK";

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  EXPECT_NE(row.find("speaker_id"), row.end());
  EXPECT_NE(row.find("utterance_number"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Test SpeechCommands dataset.
/// Description: Test usage "valid".
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestSpeechCommandsUsageValid) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDataset.";
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "valid", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  MS_LOG(INFO) << "iter->GetNextRow(&row) OK";

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("label"), row.end());
  EXPECT_NE(row.find("speaker_id"), row.end());
  EXPECT_NE(row.find("utterance_number"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Test SpeechCommands dataset.
/// Description: Test invalid folder path.
/// Expectation: Throw error messages when certain errors occur.
TEST_F(MindDataTestPipeline, TestSpeechCommandsDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDatasetFail.";

  // Create a SpeechCommands Dataset.
  std::shared_ptr<Dataset> ds = SpeechCommands("", "all", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SpeechCommands input.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test SpeechCommands dataset.
/// Description: Test error usages.
/// Expectation: Throw error messages when certain errors occur.
TEST_F(MindDataTestPipeline, TestSpeechCommandsDatasetWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDatasetFail.";

  // Create a SpeechCommands Dataset.
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "eval", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SpeechCommands input.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test SpeechCommands dataset.
/// Description: Test null sample error.
/// Expectation: Throw error messages when certain errors occur.
TEST_F(MindDataTestPipeline, TestSpeechCommandsDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpeechCommandsDatasetWithNullSamplerFail.";

  // Create a SpeechCommands Dataset.
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "all", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid SpeechCommands input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}
