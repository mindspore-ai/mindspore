/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "common/utils.h"
#include "dataset/core/client.h"
#include "dataset/core/global_context.h"
#include "dataset/engine/datasetops/source/cifar_op.h"
#include "dataset/engine/datasetops/source/sampler/sampler.h"
#include "dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "dataset/util/de_error.h"
#include "dataset/util/path.h"
#include "dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

std::shared_ptr<RepeatOp> Repeat(int repeatCnt);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<CifarOp> Cifarop(uint64_t num_works, uint64_t rows, uint64_t conns, std::string path,
                                 std::unique_ptr<Sampler> sampler = nullptr,
                                 uint64_t num_samples = 0, bool cifar10 = true) {
  std::shared_ptr<CifarOp> so;
  CifarOp::Builder builder;
  Status rc = builder.SetNumWorkers(num_works).SetCifarDir(path).SetRowsPerBuffer(rows)
                     .SetOpConnectorSize(conns).SetSampler(std::move(sampler)).SetCifarType(cifar10)
                     .SetNumSamples(num_samples).Build(&so);
  return so;
}

class MindDataTestCifarOp : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestCifarOp, TestSequentialSamplerCifar10) {
  //Note: CIFAR and Mnist datasets are not included
  //as part of the build tree.
  //Download datasets and rebuild if data doesn't
  //appear in this dataset
  //Example: python tests/dataset/data/prep_data.py
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  auto tree = Build({Cifarop(16, 2, 32, folder_path, nullptr, 100)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    uint32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<uint32_t>(&label, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 100);
  }
}

TEST_F(MindDataTestCifarOp, TestRandomSamplerCifar10) {
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(0);
  std::unique_ptr<Sampler> sampler = std::make_unique<RandomSampler>(true, 12);
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  auto tree = Build({Cifarop(16, 2, 32, folder_path, std::move(sampler), 100)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    uint32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<uint32_t>(&label, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 12);
  }
  GlobalContext::config_manager()->set_seed(original_seed);
}

TEST_F(MindDataTestCifarOp, TestCifar10NumSample) {
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  auto tree = Build({Cifarop(16, 2, 32, folder_path, nullptr, 100)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    uint32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<uint32_t>(&label, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 100);
  }
}

TEST_F(MindDataTestCifarOp, TestSequentialSamplerCifar100) {
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  auto tree = Build({Cifarop(16, 2, 32, folder_path, nullptr, 100, false)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    uint32_t coarse = 0;
    uint32_t fine = 0;
    while (tensor_map.size() != 0) {
      tensor_map["coarse_label"]->GetItemAt<uint32_t>(&coarse, {});
      tensor_map["fine_label"]->GetItemAt<uint32_t>(&fine, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << " coarse:"
                << coarse << " fine:" << fine << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 100);
  }
}
