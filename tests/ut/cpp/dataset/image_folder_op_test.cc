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
#include "dataset/engine/datasetops/source/image_folder_op.h"
#include "dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/sampler.h"
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
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

std::shared_ptr<BatchOp> Batch(int batch_size = 1, bool drop = false, int rows_per_buf = 2);

std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<ImageFolderOp> ImageFolder(int64_t num_works, int64_t rows, int64_t conns, std::string path,
                                           bool shuf = false, std::unique_ptr<Sampler> sampler = nullptr,
                                           std::map<std::string, int32_t> map = {}, int64_t num_samples = 0,
                                           bool decode = false) {
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
                     .SetNumSamples(num_samples)
                     .Build(&so);
  return so;
}

Status Create1DTensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements, unsigned char *data = nullptr,
                      DataType::Type data_type = DataType::DE_UINT32) {
  TensorShape shape(std::vector<int64_t>(1, num_elements));
  RETURN_IF_NOT_OK(
    Tensor::CreateTensor(sample_ids, TensorImpl::kFlexible, shape, DataType(data_type), data));
  if (data == nullptr) {
    (*sample_ids)->StartAddr();  // allocate memory in case user forgets!
  }
  return Status::OK();
}

class MindDataTestImageFolderSampler : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestImageFolderSampler, TestSequentialImageFolderWithRepeat) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false), Repeat(2)});
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
  std::unique_ptr<Sampler> sampler = std::make_unique<RandomSampler>(true, 12);
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
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false), Repeat(2), Batch(11)});
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
  std::unique_ptr<Sampler> sampler = std::make_unique<SubsetRandomSampler>(indices);
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
  std::unique_ptr<Sampler> sampler =
    std::make_unique<WeightedRandomSampler>(weights, num_samples, true, samples_per_buffer);

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
  std::unique_ptr<Sampler> sampler = std::make_unique<DistributedSampler>(11, 10, false);
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, std::move(sampler)), Repeat(4)});
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
  std::unique_ptr<Sampler> sampler = std::make_unique<PKSampler>(3, false, 4);
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

TEST_F(MindDataTestImageFolderSampler, TestImageFolderNumSamples) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, nullptr, {}, 11), Repeat(2)});
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
      EXPECT_TRUE(0 == label);
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 22);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestImageFolderDecode) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  std::map<std::string, int32_t> map;
  map["class3"] = 333;
  map["class1"] = 111;
  map["wrong folder name"] = 1234;  // this is skipped
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, nullptr, map, 20, true)});
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
      EXPECT_TRUE(
        tensor_map["image"]->shape() == TensorShape({2268, 4032, 3}));  // verify shapes are correct after decode
      MS_LOG(DEBUG) << "row: " << i << "\t" << tensor_map["image"]->shape() << "label:" << label << "\n";
      i++;
      di.GetNextAsMap(&tensor_map);
    }
    EXPECT_TRUE(i == 20);
  }
}

TEST_F(MindDataTestImageFolderSampler, TestImageFolderDatasetSize) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  int64_t num_rows = 0;
  int64_t num_classes = 0;
  ImageFolderOp::CountRowsAndClasses(folder_path, 15, {}, &num_rows, &num_classes);
  EXPECT_TRUE(num_rows == 15 && num_classes == 4);
  ImageFolderOp::CountRowsAndClasses(folder_path, 44, {}, &num_rows, &num_classes);
  EXPECT_TRUE(num_rows == 44 && num_classes == 4);
  ImageFolderOp::CountRowsAndClasses(folder_path, 0, {}, &num_rows, &num_classes);
  EXPECT_TRUE(num_rows == 44 && num_classes == 4);
  ImageFolderOp::CountRowsAndClasses(folder_path, 55, {}, &num_rows, &num_classes);
  EXPECT_TRUE(num_rows == 44 && num_classes == 4);
  ImageFolderOp::CountRowsAndClasses(folder_path, 44, {}, &num_rows, &num_classes, 2, 3);
  EXPECT_TRUE(num_rows == 15 && num_classes == 4);
  ImageFolderOp::CountRowsAndClasses(folder_path, 33, {}, &num_rows, &num_classes, 0, 3);
  EXPECT_TRUE(num_rows == 15 && num_classes == 4);
  ImageFolderOp::CountRowsAndClasses(folder_path, 13, {}, &num_rows, &num_classes, 0, 11);
  EXPECT_TRUE(num_rows == 4 && num_classes == 4);
  ImageFolderOp::CountRowsAndClasses(folder_path, 3, {}, &num_rows, &num_classes, 0, 11);
  EXPECT_TRUE(num_rows == 3 && num_classes == 4);
}

TEST_F(MindDataTestImageFolderSampler, TestImageFolderSharding1) {
  std::unique_ptr<Sampler> sampler = std::make_unique<DistributedSampler>(4, 0, false);
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  // numWrks, rows, conns, path, shuffle, sampler, map, numSamples, decode
  auto tree = Build({ImageFolder(16, 2, 32, folder_path, false, std::move(sampler), {}, 5)});
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
  std::unique_ptr<Sampler> sampler = std::make_unique<DistributedSampler>(4, 3, false);
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  // numWrks, rows, conns, path, shuffle, sampler, map, numSamples, decode
  auto tree = Build({ImageFolder(16, 16, 32, folder_path, false, std::move(sampler), {}, 12)});
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
