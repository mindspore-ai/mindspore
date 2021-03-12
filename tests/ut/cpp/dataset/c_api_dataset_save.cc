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
#include <stdio.h>
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/transforms.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestSaveCifar10AndLoad) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSaveCifar10AndLoad(single mindrecord file).";

  // Stage 1: load original dataset
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<SequentialSampler>(0, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  std::vector<mindspore::MSTensor> original_data;
  iter->GetNextRow(&row);

  // Save original data for comparison
  uint64_t i = 0;
  while (row.size() != 0) {
    auto label = row["label"];
    original_data.push_back(label);
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor label: ", label);

    iter->GetNextRow(&row);
    i++;
  }

  // Expect 10 samples
  EXPECT_EQ(i, 10);
  // Manually terminate the pipeline
  iter->Stop();

  // Stage 2: Save data processed by the dataset pipeline
  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::string temp_file = datasets_root_path_ + "/testCifar10Data/mind.mind";
  std::string temp_file_db = datasets_root_path_ + "/testCifar10Data/mind.mind.db";
  bool rc = ds->Save(temp_file);
  // if save fails, no need to continue the execution
  // save could fail if temp_file already exists
  ASSERT_EQ(rc, true);

  // Stage 3: Load dataset from file output by stage 2
  // Create a MindData Dataset
  std::shared_ptr<Dataset> ds_minddata = MindData(temp_file, {}, std::make_shared<SequentialSampler>(0, 10));

  // Create objects for the tensor ops
  // uint32 will be casted to int64 implicitly in mindrecord file, so we have to cast it back to uint32
  std::shared_ptr<TensorTransform> type_cast = std::make_shared<transforms::TypeCast>("uint32");
  EXPECT_NE(type_cast, nullptr);

  // Create a Map operation on ds
  ds_minddata = ds_minddata->Map({type_cast}, {"label"});
  EXPECT_NE(ds_minddata, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter_minddata = ds_minddata->CreateIterator();
  EXPECT_NE(iter_minddata, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row_minddata;
  iter_minddata->GetNextRow(&row_minddata);

  // Check column name for each row
  EXPECT_NE(row_minddata.find("image"), row_minddata.end());
  EXPECT_NE(row_minddata.find("label"), row_minddata.end());

  // Expect the output data is same with original_data
  uint64_t j = 0;
  while (row_minddata.size() != 0) {
    auto label = row_minddata["label"];
    EXPECT_MSTENSOR_EQ(original_data[j], label);
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor label: ", label);

    iter_minddata->GetNextRow(&row_minddata);
    j++;
  }

  // Expect 10 samples
  EXPECT_EQ(j, 10);
  // Manually terminate the pipeline
  iter_minddata->Stop();

  // Delete temp file
  EXPECT_EQ(remove(temp_file.c_str()), 0);
  EXPECT_EQ(remove(temp_file_db.c_str()), 0);
}

TEST_F(MindDataTestPipeline, TestSaveFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSaveFail with incorrect param.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<SequentialSampler>(0, 10));
  EXPECT_NE(ds, nullptr);

  // fail with invalid dataset_path
  std::string temp_file1 = "";
  bool rc1 = ds->Save(temp_file1);
  EXPECT_EQ(rc1, false);

  // fail with invalid dataset_path
  std::string temp_file2 = datasets_root_path_ + "/testCifar10Data/";
  bool rc2 = ds->Save(temp_file2);
  EXPECT_EQ(rc2, false);

  // fail with invalid num_files
  std::string temp_file3 = datasets_root_path_ + "/testCifar10Data/mind.mind";
  bool rc3 = ds->Save(temp_file3, 0);
  EXPECT_EQ(rc3, false);

  // fail with invalid num_files
  std::string temp_file4 = datasets_root_path_ + "/testCifar10Data/mind.mind";
  bool rc4 = ds->Save(temp_file4, 1001);
  EXPECT_EQ(rc4, false);

  // fail with invalid dataset_type
  std::string temp_file5 = datasets_root_path_ + "/testCifar10Data/mind.mind";
  bool rc5 = ds->Save(temp_file5, 5, "tfrecord");
  EXPECT_EQ(rc5, false);
}
