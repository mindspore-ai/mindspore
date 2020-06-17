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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "common/common.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/source/album_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/transforms.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

std::shared_ptr<BatchOp> Batch(int batch_size = 1, bool drop = false, int rows_per_buf = 2);

std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<AlbumOp> Album(int64_t num_works, int64_t rows, int64_t conns, std::string path,
                                           bool shuf = false, std::unique_ptr<Sampler> sampler = nullptr,
                                           bool decode = false) {
  std::shared_ptr<AlbumOp> so;
  AlbumOp::Builder builder;
  Status rc = builder.SetNumWorkers(num_works)
                     .SetAlbumDir(path)
                     .SetRowsPerBuffer(rows)
                     .SetOpConnectorSize(conns)
                     .SetExtensions({".json"})
                     .SetSampler(std::move(sampler))
                     .SetDecode(decode)
                     .Build(&so);
  return so;
}

std::shared_ptr<AlbumOp> AlbumSchema(int64_t num_works, int64_t rows, int64_t conns, std::string path,
                                       std::string schema_file, std::vector<std::string> column_names = {},
                                       bool shuf = false, std::unique_ptr<Sampler> sampler = nullptr,
                                       bool decode = false) {
  std::shared_ptr<AlbumOp> so;
  AlbumOp::Builder builder;
  Status rc = builder.SetNumWorkers(num_works)
    .SetSchemaFile(schema_file)
    .SetColumnsToLoad(column_names)
    .SetAlbumDir(path)
    .SetRowsPerBuffer(rows)
    .SetOpConnectorSize(conns)
    .SetExtensions({".json"})
    .SetSampler(std::move(sampler))
    .SetDecode(decode)
    .Build(&so);
  return so;
}

class MindDataTestAlbum : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestAlbum, TestSequentialAlbumWithSchema) {
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  auto tree = Build({AlbumSchema(16, 2, 32, folder_path, schema_file, column_names, false), Repeat(2)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " <<  ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "label shape"
                    << tensor_map["label"] << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    MS_LOG(INFO) << "got rows" << i << "\n";
    EXPECT_TRUE(i == 14);
  }
}

TEST_F(MindDataTestAlbum, TestSequentialAlbumWithSchemaNoOrder) {
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  auto tree = Build({AlbumSchema(16, 2, 32, folder_path, schema_file), Repeat(2)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "label shape"
                    << tensor_map["label"] << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    MS_LOG(INFO) << "got rows" << i << "\n";
    EXPECT_TRUE(i == 14);
  }
}

TEST_F(MindDataTestAlbum, TestSequentialAlbumWithSchemaFloat) {
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  // add the priority column
  std::string schema_file = datasets_root_path_ + "/testAlbum/floatSchema.json";
  auto tree = Build({AlbumSchema(16, 2, 32, folder_path, schema_file), Repeat(2)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    double priority = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      tensor_map["_priority"]->GetItemAt<double>(&priority, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "label shape"
                    << tensor_map["label"]  << "priority: " << priority << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    MS_LOG(INFO) << "got rows" << i << "\n";
    EXPECT_TRUE(i == 14);
  }
}

TEST_F(MindDataTestAlbum, TestSequentialAlbumWithFullSchema) {
  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  // add the priority column
  std::string schema_file = datasets_root_path_ + "/testAlbum/fullSchema.json";
  auto tree = Build({AlbumSchema(16, 2, 32, folder_path, schema_file), Repeat(2)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    double priority = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      tensor_map["_priority"]->GetItemAt<double>(&priority, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "label shape"
                    << tensor_map["label"]  << "priority: " << priority << " embedding : " <<
		    tensor_map["_embedding"]->shape() << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    MS_LOG(INFO) << "got rows" << i << "\n";
    EXPECT_TRUE(i == 14);
  }
}

