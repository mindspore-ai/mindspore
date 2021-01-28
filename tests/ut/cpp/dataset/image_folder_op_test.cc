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
#include <iostream>
#include <memory>
#include <string>
#include "common/common.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

std::shared_ptr<BatchOp> Batch(int batch_size = 1, bool drop = false, int rows_per_buf = 2);

std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<ImageFolderOp> ImageFolder(int64_t num_works, int64_t rows, int64_t conns, std::string path,
                                           bool shuf = false, std::shared_ptr<SamplerRT> sampler = nullptr,
                                           std::map<std::string, int32_t> map = {}, bool decode = false) {
  std::shared_ptr<ImageFolderOp> so;
  ImageFolderOp::Builder builder;
  Status rc = builder.SetNumWorkers(num_works)
                .SetImageFolderDir(path)
                .SetRowsPerBuffer(rows)
                .SetOpConnectorSize(conns)
                .SetExtensions({".jpg", ".JPEG"})
                .SetSampler(std::move(sampler))
                .SetClassIndex(map)
                .SetDecode(decode)
                .Build(&so);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Fail to build ImageFolderOp: " << rc.ToString() << "\n";
  }
  return so;
}

Status Create1DTensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements, unsigned char *data = nullptr,
                      DataType::Type data_type = DataType::DE_UINT32) {
  TensorShape shape(std::vector<int64_t>(1, num_elements));
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(shape, DataType(data_type), data, sample_ids));

  return Status::OK();
}

class MindDataTestImageFolderSampler : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestImageFolderSampler, TestSequentialImageFolderWithRepeat) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto op1 = ImageFolder(16, 2, 32, folder_path, false);
  auto op2 = Repeat(2);
  op1->set_total_repeats(2);
  op1->set_num_repeats_per_epoch(2);
  auto tree = Build({op1, op2});
  tree->Prepare();
  int32_t res[] = {0, 1, 2, 3};
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
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(res[(i % 44) / 11] == label);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 88);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestRandomImageFolder) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, true, nullptr)});
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
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 44);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestRandomSamplerImageFolder) {
  int32_t original_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(0);
  int64_t num_samples = 12;
  std::shared_ptr<SamplerRT> sampler = std::make_unique<RandomSamplerRT>(num_samples, true, true);
  int32_t res[] = {2, 2, 2, 3, 2, 3, 2, 3, 1, 2, 2, 1};  // ground truth label
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, std::move(sampler))});
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
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(res[i] == label);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 12);
  }
  GlobalContext::config_manager()->set_seed(original_seed);
}

TEST_F(MindDataTestImageFolderSampler, TestSequentialImageFolderWithRepeatBatch) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto op1 = ImageFolder(16, 2, 32, folder_path, false);
  auto op2 = Repeat(2);
  auto op3 = Batch(11);
  op1->set_total_repeats(2);
  op1->set_num_repeats_per_epoch(2);
  auto tree = Build({op1, op2, op3});
  tree->Prepare();
  int32_t res[4][11] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                        {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                        {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}};
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
    while (tensor_map.size() != 0) {
      std::shared_ptr<Tensor> label;
      Create1DTensor(&label, 11, reinterpret_cast<unsigned char *>(res[i % 4]), DataType::DE_INT32);
      EXPECT_TRUE((*label) == (*tensor_map["label"]));
      MS_LOG(DEBUG) << "row: " << i << " " << tensor_map["image"]->shape() << " (*label):" << (*label)
                    << " *tensor_map[label]: " << *tensor_map["label"] << std::endl;
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 8);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestSubsetRandomSamplerImageFolder) {
  // id range 0 - 10 is label 0, and id range 11 - 21 is label 1
  std::vector<int64_t> indices({0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 11});
  int64_t num_samples = 0;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<SubsetRandomSamplerRT>(num_samples, indices);
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  // Expect 6 samples for label 0 and 1
  int res[2] = {6, 6};
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, std::move(sampler))});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
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
      res[label]--;
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_EQ(res[0], 0);
    EXPECT_EQ(res[1], 0);
    EXPECT_TRUE(i == 12);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestWeightedRandomSamplerImageFolder) {
  // num samples to draw.
  int64_t num_samples = 12;
  int64_t total_samples = 44;
  int64_t samples_per_buffer = 10;
  std::vector<double> weights(total_samples, std::rand() % 100);

  // create sampler with replacement = replacement
  std::shared_ptr<SamplerRT> sampler =
    std::make_shared<WeightedRandomSamplerRT>(num_samples, weights, true, samples_per_buffer);

  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, std::move(sampler))});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
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
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 12);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestImageFolderClassIndex) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  std::map<std::string, int32_t> map;
  map["class3"] = 333;
  map["class1"] = 111;
  map["wrong folder name"] = 1234;  // this is skipped
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, nullptr, map)});
  int64_t res[2] = {111, 333};
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
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(label == res[i / 11]);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 22);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestDistributedSampler) {
  int64_t num_samples = 0;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<DistributedSamplerRT>(num_samples, 11, 10, false);
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto op1 = ImageFolder(16, 2, 32, folder_path, false, std::move(sampler));
  auto op2 = Repeat(4);
  op1->set_total_repeats(4);
  op1->set_num_repeats_per_epoch(4);
  auto tree = Build({op1, op2});
  tree->Prepare();
  Status rc = tree->Launch();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
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
      EXPECT_EQ(i % 4, label);
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 16);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestPKSamplerImageFolder) {
  int64_t num_samples = 0;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<PKSamplerRT>(num_samples, 3, false, 4);
  int32_t res[] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};  // ground truth label
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, std::move(sampler))});
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
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(res[i] == label);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 12);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestImageFolderDecode) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  std::map<std::string, int32_t> map;
  map["class3"] = 333;
  map["class1"] = 111;
  map["wrong folder name"] = 1234;  // this is skipped
  int64_t num_samples = 20;
  int64_t start_index = 0;
  auto seq_sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, std::move(seq_sampler), map, true)});
  int64_t res[2] = {111, 333};
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
    int32_t label = 0;
    while (tensor_map.size() != 0) {
      tensor_map["label"]->GetItemAt<int32_t>(&label, {});
      EXPECT_TRUE(label == res[i / 11]);
      EXPECT_TRUE(tensor_map["image"]->shape() ==
                  TensorShape({2268, 4032, 3}));  // verify shapes are correct after decode
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 20);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestImageFolderSharding1) {
  int64_t num_samples = 5;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<DistributedSamplerRT>(num_samples, 4, 0, false);
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  // numWrks, rows, conns, path, shuffle, sampler, map, numSamples, decode
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, std::move(sampler), {})});
  tree->Prepare();
  Status rc = tree->Launch();
  int32_t labels[5] = {0, 0, 0, 1, 1};
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
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
      EXPECT_EQ(labels[i], label);
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 5);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestImageFolderSharding2) {
  int64_t num_samples = 12;
  std::shared_ptr<SamplerRT> sampler = std::make_shared<DistributedSamplerRT>(num_samples, 4, 3, false);
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  // numWrks, rows, conns, path, shuffle, sampler, map, numSamples, decode
  auto tree = Build({ImageFolder(16, 16, 32, folder_path, false, std::move(sampler), {})});
  tree->Prepare();
  Status rc = tree->Launch();
  uint32_t labels[11] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during tree launch: " << common::SafeCStr(rc.ToString()) << ".";
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
      EXPECT_EQ(labels[i], label);
      MS_LOG(DEBUG) << "row:" << i << "\tlabel:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 11);
  }
}
