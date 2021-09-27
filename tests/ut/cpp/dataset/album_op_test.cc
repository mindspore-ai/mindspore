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
#include <memory>
#include <string>
#include "common/common.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/source/album_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

// std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

// std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<AlbumOp> AlbumSchema(int64_t num_works, int64_t conns, std::string path, std::string schema_file,
                                     std::vector<std::string> column_names = {}, bool shuf = false,
                                     std::shared_ptr<SamplerRT> sampler = nullptr, bool decode = false) {
  auto schema = std::make_unique<DataSchema>();
  // AlbumOp constructor for reference
  // AlbumOp(int32_t num_wkrs, int32_t rows_per_buffer, std::string file_dir, int32_t queue_size, bool do_decode,
  //         const std::set<std::string> &exts, std::unique_ptr<DataSchema> data_schema,
  //         std::shared_ptr<SamplerRT> sampler)

  // default schema construction:

  (void)schema->LoadSchemaFile(schema_file, column_names);
  std::set<std::string> ext = {".json"};
  if (sampler == nullptr) {
    const int64_t num_samples = 0;  // default num samples of 0 means to sample entire set of data
    const int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  }
  std::shared_ptr<AlbumOp> so =
    std::make_shared<AlbumOp>(num_works, path, conns, decode, ext, std::move(schema), std::move(sampler));
  return so;
}

class MindDataTestAlbum : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestAlbum, TestSequentialAlbumWithSchema) {
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  auto op1 = AlbumSchema(16, 32, folder_path, schema_file, column_names, false);
  std::shared_ptr<RepeatOp> op2 = Repeat(2);
  op1->SetTotalRepeats(2);
  op1->SetNumRepeatsPerEpoch(2);
  std::shared_ptr<ExecutionTree> tree = Build({op1, op2});
  ASSERT_OK(tree->Prepare());
  ASSERT_OK(tree->Launch());
  DatasetIterator di(tree);
  TensorMap tensor_map;
  ASSERT_OK(di.GetNextAsMap(&tensor_map));
  uint64_t i = 0;
  std::string_view label = 0;
  while (tensor_map.size() != 0) {
    EXPECT_TRUE(tensor_map["label"]->GetItemAt(&label, {0}));
    MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "label shape"
                  << tensor_map["label"] << "\n";
    i++;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
  }
  MS_LOG(INFO) << "got rows: " << i << "\n";
  EXPECT_TRUE(i == 14);
}

TEST_F(MindDataTestAlbum, TestSequentialAlbumWithSchemaNoOrder) {
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  auto op1 = AlbumSchema(16, 32, folder_path, schema_file);
  std::shared_ptr<RepeatOp> op2 = Repeat(2);
  op1->SetTotalRepeats(2);
  op1->SetNumRepeatsPerEpoch(2);
  std::shared_ptr<ExecutionTree> tree = Build({op1, op2});
  ASSERT_OK(tree->Prepare());
  ASSERT_OK(tree->Launch());
  DatasetIterator di(tree);
  TensorMap tensor_map;
  ASSERT_OK(di.GetNextAsMap(&tensor_map));
  uint64_t i = 0;
  std::string_view label;
  while (tensor_map.size() != 0) {
    EXPECT_OK(tensor_map["label"]->GetItemAt(&label, {0}));
    MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "label shape"
                  << tensor_map["label"] << "\n";
    i++;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
  }
  MS_LOG(INFO) << "got rows: " << i << "\n";
  EXPECT_TRUE(i == 14);
}

TEST_F(MindDataTestAlbum, TestSequentialAlbumWithSchemaFloat) {
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  // add the priority column
  std::string schema_file = datasets_root_path_ + "/testAlbum/floatSchema.json";
  auto op1 = AlbumSchema(16, 32, folder_path, schema_file);
  std::shared_ptr<RepeatOp> op2 = Repeat(2);
  op1->SetTotalRepeats(2);
  op1->SetNumRepeatsPerEpoch(2);
  std::shared_ptr<ExecutionTree> tree = Build({op1, op2});
  tree->Prepare();
  ASSERT_OK(tree->Launch());
  DatasetIterator di(tree);
  TensorMap tensor_map;
  ASSERT_OK(di.GetNextAsMap(&tensor_map));
  uint64_t i = 0;
  std::string_view label;
  double priority = 0;
  while (tensor_map.size() != 0) {
    EXPECT_OK(tensor_map["label"]->GetItemAt(&label, {0}));
    EXPECT_OK(tensor_map["_priority"]->GetItemAt<double>(&priority, {0}));
    MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "label shape"
                  << tensor_map["label"] << "priority: " << priority << "\n";
    i++;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
  }
  MS_LOG(INFO) << "got rows: " << i << "\n";
  EXPECT_TRUE(i == 14);
}

TEST_F(MindDataTestAlbum, TestSequentialAlbumWithFullSchema) {
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  // add the priority column
  std::string schema_file = datasets_root_path_ + "/testAlbum/fullSchema.json";
  auto op1 = AlbumSchema(16, 32, folder_path, schema_file);
  std::shared_ptr<RepeatOp> op2 = Repeat(2);
  op1->SetTotalRepeats(2);
  op1->SetNumRepeatsPerEpoch(2);
  std::shared_ptr<ExecutionTree> tree = Build({op1, op2});
  ASSERT_OK(tree->Prepare());
  ASSERT_OK(tree->Launch());
  DatasetIterator di(tree);
  TensorMap tensor_map;
  ASSERT_OK(di.GetNextAsMap(&tensor_map));
  uint64_t i = 0;
  std::string_view label = 0;
  double priority = 0;
  int64_t id = 0;
  while (tensor_map.size() != 0) {
    EXPECT_OK(tensor_map["label"]->GetItemAt(&label, {0}));
    EXPECT_OK(tensor_map["_priority"]->GetItemAt<double>(&priority, {0}));
    EXPECT_OK(tensor_map["id"]->GetItemAt<int64_t>(&id, {}));
    MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "label shape"
                  << tensor_map["label"] << "priority: " << priority
                  << " embedding : " << tensor_map["_embedding"]->shape() << " id: " << id << "\n";
    i++;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
  }
  MS_LOG(INFO) << "got rows: " << i << "\n";
  EXPECT_TRUE(i == 14);
}
