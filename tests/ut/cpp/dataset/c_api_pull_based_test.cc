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

namespace common = mindspore::common;

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestPullBasedBatch) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumBasic.";

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"label"};
  // Create a Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names);
  EXPECT_NE(ds, nullptr);

  int32_t batch_size = 4;
  ds = ds->Batch(batch_size, true);
  EXPECT_NE(ds, nullptr);

  auto iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  std::vector<mindspore::MSTensor> row;
  Status rc = iter->GetNextRow(&row);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_EQ(row.size(), 1);
  auto temp = row[0].Shape();
  std::vector<int64_t> result = {batch_size, 2};
  EXPECT_EQ(row[0].Shape(), result);
}

TEST_F(MindDataTestPipeline, TestPullBasedProject) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAlbumBasic.";

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"label", "image"};
  // Create a Album Dataset
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names);
  EXPECT_NE(ds, nullptr);

  std::vector<mindspore::MSTensor> row;
  auto iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);
  Status rc = iter->GetNextRow(&row);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_EQ(row.size(), 2);

  std::shared_ptr<Dataset> ds2 = Album(folder_path, schema_file, column_names);
  EXPECT_NE(ds2, nullptr);
  std::vector<std::string> columns_to_project = {"image"};
  ds2 = ds2->Project(columns_to_project);
  EXPECT_NE(ds2, nullptr);

  auto iter2 = ds2->CreatePullBasedIterator();
  EXPECT_NE(iter2, nullptr);

  std::vector<mindspore::MSTensor> new_row;
  rc = iter2->GetNextRow(&new_row);
  EXPECT_EQ(rc, Status::OK());
  EXPECT_EQ(new_row.size(), 1);
}