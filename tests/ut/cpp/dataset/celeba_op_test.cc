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
#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

// std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

// std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<CelebAOp> Celeba(int32_t num_workers, int32_t queue_size, const std::string &dir,
                                 std::shared_ptr<SamplerRT> sampler = nullptr, bool decode = false,
                                 const std::string &dataset_type = "all") {
  if (sampler == nullptr) {
    const int64_t num_samples = 0;
    const int64_t start_index = 0;
    sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  }

  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  (void)schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1));
  (void)schema->AddColumn(ColDescriptor("attr", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1));

  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  auto op_connector_size = config_manager->op_connector_size();
  std::set<std::string> extensions = {};
  std::shared_ptr<CelebAOp> so = std::make_shared<CelebAOp>(num_workers, dir, op_connector_size, decode, dataset_type,
                                                            extensions, std::move(schema), std::move(sampler));
  return so;
}

class MindDataTestCelebaDataset : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestCelebaDataset, TestSequentialCeleba) {
  std::string dir = datasets_root_path_ + "/testCelebAData/";
  uint32_t expect_labels[4][40] = {{0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1},
                                   {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
                                   {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
                                   {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1}};
  uint32_t count = 0;
  auto tree = Build({Celeba(16, 2, dir)});
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
    while (tensor_map.size() != 0) {
      uint32_t label;
      for (int index = 0; index < 40; index++) {
        tensor_map["attr"]->GetItemAt<uint32_t>(&label, {index});
        EXPECT_TRUE(expect_labels[count][index] == label);
      }
      count++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(count == 4);
  }
}

TEST_F(MindDataTestCelebaDataset, TestCelebaRepeat) {
  std::string dir = datasets_root_path_ + "/testCelebAData/";
  uint32_t expect_labels[8][40] = {{0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1},
                                   {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
                                   {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
                                   {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1},
                                   {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1},
                                   {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
                                   {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
                                   {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1}};
  uint32_t count = 0;
  auto op1 = Celeba(16, 2, dir);
  auto op2 = Repeat(2);
  auto tree = Build({op1, op2});
  op1->SetTotalRepeats(2);
  op1->SetNumRepeatsPerEpoch(2);
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
    while (tensor_map.size() != 0) {
      uint32_t label;
      for (int index = 0; index < 40; index++) {
        tensor_map["attr"]->GetItemAt<uint32_t>(&label, {index});
        EXPECT_TRUE(expect_labels[count][index] == label);
      }
      count++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(count == 8);
  }
}

TEST_F(MindDataTestCelebaDataset, TestSubsetRandomSamplerCeleba) {
  std::vector<int64_t> indices({1});
  int64_t num_samples = 0;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<SubsetRandomSamplerRT>(indices, num_samples);
  uint32_t expect_labels[1][40] = {{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1}};
  std::string dir = datasets_root_path_ + "/testCelebAData/";
  uint32_t count = 0;
  auto tree = Build({Celeba(16, 2, dir, std::move(sampler))});
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
    while (tensor_map.size() != 0) {
      uint32_t label;
      for (int index = 0; index < 40; index++) {
        tensor_map["attr"]->GetItemAt<uint32_t>(&label, {index});
        EXPECT_TRUE(expect_labels[count][index] == label);
      }
      count++;
      ASSERT_OK(di.GetNextAsMap(&tensor_map));
    }
    EXPECT_TRUE(count == 1);
  }
}
