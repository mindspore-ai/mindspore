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
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<CelebAOp> Celeba(int32_t num_workers, int32_t rows_per_buffer, int32_t queue_size,
                                 const std::string &dir, std::shared_ptr<SamplerRT> sampler = nullptr,
                                 bool decode = false, const std::string &dataset_type = "all") {
  std::shared_ptr<CelebAOp> so;
  CelebAOp::Builder builder;
  Status rc = builder.SetNumWorkers(num_workers)
                .SetCelebADir(dir)
                .SetRowsPerBuffer(rows_per_buffer)
                .SetOpConnectorSize(queue_size)
                .SetSampler(std::move(sampler))
                .SetDecode(decode)
                .SetUsage(dataset_type).Build(&so);
  return so;
}

class MindDataTestCelebaDataset : public UT::DatasetOpTesting {
protected:
};

TEST_F(MindDataTestCelebaDataset, TestSequentialCeleba) {
  std::string dir = datasets_root_path_ + "/testCelebAData/";
  uint32_t expect_labels[4][40] = {{0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1},
                                   {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1},
                                   {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1},
                                   {0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1}};
  uint32_t count = 0;
  auto tree = Build({Celeba(16, 2, 32, dir)});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tersor_map;
    di.GetNextAsMap(&tersor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tersor_map.size() != 0) {
      uint32_t label;
      for (int index = 0; index < 40; index++) {
        tersor_map["attr"]->GetItemAt<uint32_t>(&label, {index});
        EXPECT_TRUE(expect_labels[count][index] == label);
      }
      count++;
      di.GetNextAsMap(&tersor_map);
    }
    EXPECT_TRUE(count == 4);
  }
}

TEST_F(MindDataTestCelebaDataset, TestCelebaRepeat) {
  std::string dir = datasets_root_path_ + "/testCelebAData/";
  uint32_t expect_labels[8][40] = {{0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1},
                                   {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1},
                                   {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1},
                                   {0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1},
                                   {0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1},
                                   {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1},
                                   {0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1},
                                   {0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1}};
  uint32_t count = 0;
  auto op1 = Celeba(16, 2, 32, dir);
  auto op2 = Repeat(2);
  auto tree = Build({op1, op2});
  op1->set_total_repeats(2);
  op1->set_num_repeats_per_epoch(2);
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tersor_map;
    di.GetNextAsMap(&tersor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tersor_map.size() != 0) {
      uint32_t label;
      for (int index = 0; index < 40; index++) {
        tersor_map["attr"]->GetItemAt<uint32_t>(&label, {index});
        EXPECT_TRUE(expect_labels[count][index] == label);
      }
      count++;
      di.GetNextAsMap(&tersor_map);
    }
    EXPECT_TRUE(count == 8);
  }
}

TEST_F(MindDataTestCelebaDataset, TestSubsetRandomSamplerCeleba) {
  std::vector<int64_t> indices({1});
  int64_t num_samples = 0;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<SubsetRandomSamplerRT>(num_samples, indices);
  uint32_t expect_labels[1][40] = {{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1}};
  std::string dir = datasets_root_path_ + "/testCelebAData/";
  uint32_t count = 0;
  auto tree = Build({Celeba(16, 2, 32, dir, std::move(sampler))});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tersor_map;
    di.GetNextAsMap(&tersor_map);
    EXPECT_TRUE(rc.IsOk());
    while (tersor_map.size() != 0) {
      uint32_t label;
      for (int index = 0; index < 40; index++) {
        tersor_map["attr"]->GetItemAt<uint32_t>(&label, {index});
        EXPECT_TRUE(expect_labels[count][index] == label);
      }
      count++;
      di.GetNextAsMap(&tersor_map);
    }
    EXPECT_TRUE(count == 1);
  }
}
