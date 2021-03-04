/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "utils/ms_utils.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<CifarOp> Cifarop(uint64_t num_works, uint64_t rows, uint64_t conns, std::string path,
                                 std::shared_ptr<SamplerRT> sampler = nullptr, bool cifar10 = true) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  auto num_workers = cfg->num_parallel_workers();
  std::string usage = "";
  CifarOp::CifarType cifar_type;
  if (cifar10) {
    cifar_type = CifarOp::kCifar10;
  } else {
    cifar_type = CifarOp::kCifar100;
  }
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  (void)schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1));
  if (cifar_type == CifarOp::kCifar10) {
    (void)schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar));
  } else {
    (void)schema->AddColumn(
      ColDescriptor("coarse_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar));
    TensorShape another_scalar = TensorShape::CreateScalar();
    (void)schema->AddColumn(
      ColDescriptor("fine_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &another_scalar));
  }

  if (sampler == nullptr) {
    const int64_t num_samples = 0;
    const int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  }

  std::shared_ptr<CifarOp> so =
    std::make_shared<CifarOp>(cifar_type, usage, num_workers, path, conns, std::move(schema), std::move(sampler));
  return so;
}

class MindDataTestCifarOp : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestCifarOp, TestSequentialSamplerCifar10) {
  // Note: CIFAR and Mnist datasets are not included
  // as part of the build tree.
  // Download datasets and rebuild if data doesn't
  // appear in this dataset
  // Example: python tests/dataset/data/prep_data.py
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  auto tree = Build({Cifarop(16, 2, 32, folder_path, nullptr)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    uint32_t label = 0;
    // Note: only iterating first 100 rows then break out.
    while (tensor_map.size() != 0 && i < 100) {
      tensor_map["label"]->GetItemAt<uint32_t>(&label, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(i == 100);
  }
}

TEST_F(MindDataTestCifarOp, TestRandomSamplerCifar10) {
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(0);
  std::shared_ptr<SamplerRT> sampler = std::make_unique<RandomSamplerRT>(true, 12, true);
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  auto tree = Build({Cifarop(16, 2, 32, folder_path, std::move(sampler))});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    uint32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<uint32_t>(&label, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(i == 12);
  }
  GlobalContext::config_manager()->set_seed(original_seed);
}

TEST_F(MindDataTestCifarOp, TestSequentialSamplerCifar100) {
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  auto tree = Build({Cifarop(16, 2, 32, folder_path, nullptr, false)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    uint32_t coarse = 0;
    uint32_t fine = 0;
    // only iterate to 100 then break out of loop
    while (tensor_map.size() != 0 && i < 100) {
      tensor_map["coarse_label"]->GetItemAt<uint32_t>(&coarse, {});
      tensor_map["fine_label"]->GetItemAt<uint32_t>(&fine, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << " coarse:" << coarse << " fine:" << fine
                    << "\n";
      i++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(i == 100);
  }
}
