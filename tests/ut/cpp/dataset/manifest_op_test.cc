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
#include "minddata/dataset/engine/datasetops/source/manifest_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

std::shared_ptr<ManifestOp> Manifest(int32_t num_works, int32_t rows, int32_t conns, const std::string &file,
                                     std::string usage = "train", std::shared_ptr<SamplerRT> sampler = nullptr,
                                     std::map<std::string, int32_t> map = {}, bool decode = false) {
  if (sampler == nullptr) {
    const int64_t num_samples = 0;
    const int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  }
  auto schema = std::make_unique<DataSchema>();
  schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1));
  schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1));
  return std::make_shared<ManifestOp>(num_works, file, conns, decode, map, std::move(schema), std::move(sampler),
                                      usage);
}

class MindDataTestManifest : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestManifest, TestSequentialManifestWithRepeat) {
  std::string file = datasets_root_path_ + "/testManifestData/cpp.json";
  auto op1 = Manifest(16, 2, 32, file);
  auto op2 = Repeat(2);
  op1->SetTotalRepeats(2);
  op1->SetNumRepeatsPerEpoch(2);
  auto tree = Build({op1, op2});
  tree->Prepare();
  uint32_t res[] = {0, 1, 0, 1};
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(res[i] == label);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(i == 4);
  }
}

TEST_F(MindDataTestManifest, TestSubsetRandomSamplerManifest) {
  std::vector<int64_t> indices({1});
  int64_t num_samples = 0;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<SubsetRandomSamplerRT>(indices, num_samples);
  std::string file = datasets_root_path_ + "/testManifestData/cpp.json";
  // Expect 6 samples for label 0 and 1
  auto tree = Build({Manifest(16, 2, 32, file, "train", std::move(sampler))});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      i++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
      EXPECT_EQ(label, 1);
    }
    EXPECT_TRUE(i == 1);
  }
}

TEST_F(MindDataTestManifest, MindDataTestManifestClassIndex) {
  std::string file = datasets_root_path_ + "/testManifestData/cpp.json";
  std::map<std::string, int32_t> map;
  map["cat"] = 111;  // forward slash is not good, but we need to add this somewhere, also in windows, its a '\'
  map["dog"] = 222;  // forward slash is not good, but we need to add this somewhere, also in windows, its a '\'
  map["wrong folder name"] = 1234;  // this is skipped
  auto tree = Build({Manifest(16, 2, 32, file, "train", nullptr, map)});
  uint64_t res[2] = {111, 222};
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(label == res[i]);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(i == 2);
  }
}

TEST_F(MindDataTestManifest, MindDataTestManifestNumSamples) {
  std::string file = datasets_root_path_ + "/testManifestData/cpp.json";
  int64_t num_samples = 1;
  int64_t start_index = 0;
  auto seq_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto op1 = Manifest(16, 2, 32, file, "train", std::move(seq_sampler), {});
  auto op2 = Repeat(4);
  op1->SetTotalRepeats(4);
  op1->SetNumRepeatsPerEpoch(4);
  auto tree = Build({op1, op2});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(0 == label);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(i == 4);
  }
}

TEST_F(MindDataTestManifest, MindDataTestManifestEval) {
  std::string file = datasets_root_path_ + "/testManifestData/cpp.json";
  int64_t num_samples = 1;
  int64_t start_index = 0;
  auto seq_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto tree = Build({Manifest(16, 2, 32, file, "eval", std::move(seq_sampler), {})});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(0 == label);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(i == 1);
  }
}
