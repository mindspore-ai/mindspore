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

/// Feature: TedliumDataset.
/// Description: Read some samples from all files according to different versions.
/// Expectation: 4 * 2 samples.
TEST_F(MindDataTestPipeline, TestTedliumDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDataset.";

  // Create a Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::string folder_path3 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release3";
  std::shared_ptr<Dataset> ds1 =
    Tedlium(folder_path12, "release1", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);
  std::shared_ptr<Dataset> ds3 =
    Tedlium(folder_path3, "release3", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);

  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  EXPECT_NE(iter3, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  std::unordered_map<std::string, mindspore::MSTensor> row3;
  ASSERT_OK(iter1->GetNextRow(&row1));

  EXPECT_NE(row1.find("waveform"), row1.end());
  EXPECT_NE(row1.find("sample_rate"), row1.end());
  EXPECT_NE(row1.find("transcript"), row1.end());
  EXPECT_NE(row1.find("talk_id"), row1.end());
  EXPECT_NE(row1.find("speaker_id"), row1.end());
  EXPECT_NE(row1.find("identifier"), row1.end());

  ASSERT_OK(iter3->GetNextRow(&row3));

  EXPECT_NE(row3.find("waveform"), row3.end());
  EXPECT_NE(row3.find("sample_rate"), row3.end());
  EXPECT_NE(row3.find("transcript"), row3.end());
  EXPECT_NE(row3.find("talk_id"), row3.end());
  EXPECT_NE(row3.find("speaker_id"), row3.end());
  EXPECT_NE(row3.find("identifier"), row3.end());

  uint64_t i = 0;
  while (row1.size() != 0) {
    i++;
    auto audio = row1["waveform"];
    MS_LOG(INFO) << "Tensor audio shape: " << audio.Shape();
    ASSERT_OK(iter1->GetNextRow(&row1));
  }
  while (row3.size() != 0) {
    i++;
    auto audio = row3["waveform"];
    MS_LOG(INFO) << "Tensor audio shape: " << audio.Shape();
    ASSERT_OK(iter3->GetNextRow(&row3));
  }

  EXPECT_EQ(i, 4 * 2);

  // Manually terminate the pipeline.
  iter1->Stop();
  iter3->Stop();
}

/// Feature: TedliumDataset.
/// Description: Read some samples with pipeline from all files according to different versions.
/// Expectation: 8 * 2 samples.
TEST_F(MindDataTestPipeline, TestTedliumDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDatasetWithPipeline.";

  // Create two Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::string folder_path3 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release3";
  std::shared_ptr<Dataset> ds11 =
    Tedlium(folder_path12, "release1", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);
  std::shared_ptr<Dataset> ds31 =
    Tedlium(folder_path3, "release3", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);
  std::shared_ptr<Dataset> ds12 =
    Tedlium(folder_path12, "release1", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);
  std::shared_ptr<Dataset> ds32 =
    Tedlium(folder_path3, "release3", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);

  EXPECT_NE(ds11, nullptr);
  EXPECT_NE(ds12, nullptr);
  EXPECT_NE(ds31, nullptr);
  EXPECT_NE(ds32, nullptr);

  // Create two Repeat operation on ds.
  int32_t repeat_num = 1;
  ds11 = ds11->Repeat(repeat_num);
  ds31 = ds31->Repeat(repeat_num);
  EXPECT_NE(ds11, nullptr);
  EXPECT_NE(ds31, nullptr);
  repeat_num = 1;
  ds12 = ds12->Repeat(repeat_num);
  ds32 = ds32->Repeat(repeat_num);
  EXPECT_NE(ds12, nullptr);
  EXPECT_NE(ds32, nullptr);

  // Create two Project operation on ds.
  std::vector<std::string> column_project = {"waveform", "sample_rate", "transcript",
                                             "talk_id",  "speaker_id",  "identifier"};
  ds11 = ds11->Project(column_project);
  EXPECT_NE(ds11, nullptr);
  ds12 = ds12->Project(column_project);
  EXPECT_NE(ds12, nullptr);
  ds31 = ds31->Project(column_project);
  EXPECT_NE(ds31, nullptr);
  ds32 = ds32->Project(column_project);
  EXPECT_NE(ds32, nullptr);

  // Create a Concat operation on the ds.
  ds11 = ds11->Concat({ds12});
  ds31 = ds31->Concat({ds32});
  EXPECT_NE(ds11, nullptr);
  EXPECT_NE(ds31, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds11->CreateIterator();
  std::shared_ptr<Iterator> iter3 = ds31->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  EXPECT_NE(iter3, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  std::unordered_map<std::string, mindspore::MSTensor> row3;
  ASSERT_OK(iter1->GetNextRow(&row1));
  ASSERT_OK(iter3->GetNextRow(&row3));

  EXPECT_NE(row1.find("waveform"), row1.end());
  EXPECT_NE(row1.find("sample_rate"), row1.end());
  EXPECT_NE(row1.find("transcript"), row1.end());
  EXPECT_NE(row1.find("talk_id"), row1.end());
  EXPECT_NE(row1.find("speaker_id"), row1.end());
  EXPECT_NE(row1.find("identifier"), row1.end());

  EXPECT_NE(row3.find("waveform"), row3.end());
  EXPECT_NE(row3.find("sample_rate"), row3.end());
  EXPECT_NE(row3.find("transcript"), row3.end());
  EXPECT_NE(row3.find("talk_id"), row3.end());
  EXPECT_NE(row3.find("speaker_id"), row3.end());
  EXPECT_NE(row3.find("identifier"), row3.end());

  uint64_t i = 0;
  while (row1.size() != 0) {
    i++;
    auto audio = row1["waveform"];
    MS_LOG(INFO) << "Tensor audio shape: " << audio.Shape();
    ASSERT_OK(iter1->GetNextRow(&row1));
  }
  while (row3.size() != 0) {
    i++;
    auto audio = row3["waveform"];
    MS_LOG(INFO) << "Tensor audio shape: " << audio.Shape();
    ASSERT_OK(iter3->GetNextRow(&row3));
  }

  EXPECT_EQ(i, 8 * 2);

  // Manually terminate the pipeline.
  iter1->Stop();
  iter3->Stop();
}

/// Feature: TedliumDataset.
/// Description: Test iterator of Tedlium dataset with only the "waveform" column.
/// Expectation: Get correct data.
TEST_F(MindDataTestPipeline, TestTedliumDatasetIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDatasetIteratorOneColumn.";
  // Create a Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::shared_ptr<Dataset> ds =
    Tedlium(folder_path12, "release1", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto audio = row["waveform"];
    MS_LOG(INFO) << "Tensor audio shape: " << audio.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: TedliumDataset.
/// Description: Test iterator of TedliumDataset with wrong column.
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr.
TEST_F(MindDataTestPipeline, TestTedliumDatasetIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDatasetIteratorWrongColumn.";
  // Create a Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::shared_ptr<Dataset> ds =
    Tedlium(folder_path12, "release1", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: TedliumDataset.
/// Description: Read number of all samples from all files according to different versions.
/// Expectation: TEDLIUM_release12 : 1 + 2 + 3
///              TEDLIUM_release3 : 3 + 4
TEST_F(MindDataTestPipeline, TestTedliumGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumGetDatasetSize.";

  // Create a Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::string folder_path3 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release3";
  std::shared_ptr<Dataset> ds1 = Tedlium(folder_path12, "release1", "all", ".sph");
  std::shared_ptr<Dataset> ds3 = Tedlium(folder_path3, "release3", "all", ".sph");
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds3, nullptr);

  EXPECT_EQ(ds1->GetDatasetSize(), 1 + 2 + 3);
  EXPECT_EQ(ds3->GetDatasetSize(), 3 + 4);
}

/// Feature: TedliumDataset.
/// Description: Test TedliumDataset Getters method.
/// Expectation: Correct shape, type, size.
TEST_F(MindDataTestPipeline, TestTedliumGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumGetters.";

  // Create a Tedlium Dataset.
  std::string folder_path = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::shared_ptr<Dataset> ds = Tedlium(folder_path, "release1", "all", ".sph");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 1 + 2 + 3);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"waveform", "sample_rate", "transcript",
                                           "talk_id",  "speaker_id",  "identifier"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 6);
  EXPECT_EQ(types[0].ToString(), "float32");
  EXPECT_EQ(types[1].ToString(), "int32");
  EXPECT_EQ(types[2].ToString(), "string");
  EXPECT_EQ(types[3].ToString(), "string");
  EXPECT_EQ(types[4].ToString(), "string");
  EXPECT_EQ(types[5].ToString(), "string");

  EXPECT_EQ(shapes.size(), 6);
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(shapes[2].ToString(), "<>");
  EXPECT_EQ(shapes[3].ToString(), "<>");
  EXPECT_EQ(shapes[4].ToString(), "<>");
  EXPECT_EQ(shapes[5].ToString(), "<>");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 1 + 2 + 3);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), -1);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 1 + 2 + 3);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetDatasetSize(), 1 + 2 + 3);
}

/// Feature: TedliumDataset.
/// Description: Test with invalid release.
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr.
TEST_F(MindDataTestPipeline, TestTedliumDatasetWithInvalidReleaseFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDatasetWithInvalidReleaseFail.";

  // Create a Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::string folder_path3 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release3";
  std::shared_ptr<Dataset> ds1 = Tedlium(folder_path12, "", "all", ".sph");
  std::shared_ptr<Dataset> ds2 = Tedlium(folder_path12, "RELEASE2", "all", ".sph");
  std::shared_ptr<Dataset> ds3 = Tedlium(folder_path3, "2", "all", ".sph");
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid Tedlium input, "", "RELEASE2" and "2" are not a valid release.
  EXPECT_EQ(iter1, nullptr);
  EXPECT_EQ(iter2, nullptr);
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: TedliumDataset.
/// Description: Test with invalid path.
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr.
TEST_F(MindDataTestPipeline, TestTedliumDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDatasetFail.";

  // Create a Tedlium Dataset.
  std::shared_ptr<Dataset> ds1 = Tedlium("", "release1", "all", ".sph", std::make_shared<RandomSampler>(false, 4));
  std::shared_ptr<Dataset> ds2 =
    Tedlium("validation", "release2", "all", ".sph", std::make_shared<RandomSampler>(false, 4));
  std::shared_ptr<Dataset> ds3 = Tedlium("2", "release3", "all", ".sph", std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid Tedlium input, "", "validation" and "2" are not a valid path.
  EXPECT_EQ(iter1, nullptr);
  EXPECT_EQ(iter2, nullptr);
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: TedliumDataset.
/// Description: Test with invalid usage.
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr.
TEST_F(MindDataTestPipeline, TestTedliumDatasetWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDatasetWithInvalidUsageFail.";

  // Create a Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::string folder_path3 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release3";
  std::shared_ptr<Dataset> ds1 = Tedlium(folder_path12, "release1", "", ".sph");
  std::shared_ptr<Dataset> ds2 = Tedlium(folder_path12, "release2", "DEV", ".sph");
  std::shared_ptr<Dataset> ds3 = Tedlium(folder_path3, "release3", "2", ".sph");
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid Tedlium input, "", "DEV" and "2" are not a valid usage.
  EXPECT_EQ(iter1, nullptr);
  EXPECT_EQ(iter2, nullptr);
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: TedliumDataset.
/// Description: Test with invalid extensions.
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr.
TEST_F(MindDataTestPipeline, TestTedliumDatasetWithInvalidExtensionsFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDatasetWithInvalidExtensionsFail.";

  // Create a Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::string folder_path3 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release3";
  std::shared_ptr<Dataset> ds1 = Tedlium(folder_path12, "release1", "all", "sph");
  std::shared_ptr<Dataset> ds2 = Tedlium(folder_path12, "release2", "all", ".SPH");
  std::shared_ptr<Dataset> ds3 = Tedlium(folder_path3, "release3", "all", ".stm");
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid Tedlium input, "sph", ".SPH", ".stm" are not a valid extensions.
  EXPECT_EQ(iter1, nullptr);
  EXPECT_EQ(iter2, nullptr);
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: TedliumDataset.
/// Description: Test with null sampler.
/// Expectation: Unable to read in data.
TEST_F(MindDataTestPipeline, TestTedliumDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTedliumDatasetWithNullSamplerFail.";

  // Create a Tedlium Dataset.
  std::string folder_path = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::shared_ptr<Dataset> ds = Tedlium(folder_path, "release1", "all", ".sph", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Tedlium input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}
