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

#include "utils/ms_utils.h"
#include "common/common.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
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

namespace common = mindspore::common;
using namespace mindspore::dataset;
using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

std::shared_ptr<BatchOp> Batch(int batch_size = 1, bool drop = false, int rows_per_buf = 2);

std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

Status Create1DTensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements, unsigned char *data = nullptr,
                      DataType::Type data_type = DataType::DE_UINT32);

std::shared_ptr<MnistOp> CreateMnist(int64_t num_wrks, int64_t rows, int64_t conns, std::string path, bool shuf = false,
                                     std::shared_ptr<SamplerRT> sampler = nullptr) {
  std::shared_ptr<MnistOp> so;
  MnistOp::Builder builder;
  Status rc = builder.SetNumWorkers(num_wrks)
                .SetDir(path)
                .SetRowsPerBuffer(rows)
                .SetOpConnectorSize(conns)
                .SetSampler(std::move(sampler))
                .Build(&so);
  return so;
}

class MindDataTestMnistSampler : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestMnistSampler, TestSequentialMnistWithRepeat) {
  // Note: Mnist datasets are not included
  // as part of the build tree.
  // Download datasets and rebuild if data doesn't
  // appear in this dataset
  // Example: python tests/dataset/data/prep_data.py
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  int64_t num_samples = 10;
  int64_t start_index = 0;
  auto seq_sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
  auto op1 = CreateMnist(16, 2, 32, folder_path, false, std::move(seq_sampler));
  auto op2 = Repeat(2);
  op1->set_total_repeats(2);
  op1->set_num_repeats_per_epoch(2);
  auto tree = Build({op1, op2});
  tree->Prepare();
  uint32_t res[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
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
      EXPECT_TRUE(res[i % 10] == label);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 20);
  }
}

TEST_F(MindDataTestMnistSampler, TestSequentialImageFolderWithRepeatBatch) {
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  int64_t num_samples = 10;
  int64_t start_index = 0;
  auto seq_sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
  auto op1 = CreateMnist(16, 2, 32, folder_path, false, std::move(seq_sampler));
  auto op2 = Repeat(2);
  auto op3 = Batch(5);
  op1->set_total_repeats(2);
  op1->set_num_repeats_per_epoch(2);
  auto tree = Build({op1, op2, op3});
  tree->Prepare();
  uint32_t res[4][5] = { {0, 0, 0, 0, 0 },
                         {0, 0, 0, 0, 0 },
                         {0, 0, 0, 0, 0 },
                         {0, 0, 0, 0, 0 } };
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << rc.ToString() << ".";
    EXPECT_TRUE(false);
  } else {
    DatasetIterator di(tree);
    TensorMap tensor_map;
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    uint64_t i = 0;
    while (tensor_map.size() != 0) {
      std::shared_ptr<Tensor> label;
      Create1DTensor(&label, 5, reinterpret_cast<unsigned char *>(res[i % 4]));
      EXPECT_TRUE((*label) == (*tensor_map["label"]));
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << *tensor_map["label"] << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 4);
  }
}
