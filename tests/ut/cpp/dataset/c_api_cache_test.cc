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
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"

// IR leaf nodes

#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"

using namespace mindspore::dataset;
using namespace mindspore::dataset::api;

// Helper function to get the session id from SESSION_ID env variable
Status GetSessionFromEnv(session_id_type *session_id);

class MindDataTestCacheOp : public UT::DatasetOpTesting {
 public:
  void SetUp() override {
    DatasetOpTesting::SetUp();
    GlobalInit();
  }
};

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheCApiSamplerNull) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true, "127.0.0.1", 50053, 1, 1);
  EXPECT_NE(some_cache, nullptr);

  // Create an ImageFolder Dataset, this folder_path only has 2 images in it
  std::string folder_path = datasets_root_path_ + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, nullptr, {}, {}, some_cache);
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheImageFolderCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create an ImageFolder Dataset, this folder_path only has 2 images in it
  std::string folder_path = datasets_root_path_ + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(), {}, {}, some_cache);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheCocoCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a Coco Dataset, this folder_path has 6 images in it
  std::string folder_path = datasets_root_path_ + "/testCOCO/train/";
  std::string annotation_file_path = datasets_root_path_ + "/testCOCO/annotations/train.json";
  std::shared_ptr<Dataset> ds =
    Coco(folder_path, annotation_file_path, "Detection", false, RandomSampler(), some_cache);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheMnistCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", RandomSampler(false, 10), some_cache);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheCelebaCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a CelebA Dataset, this folder_path has 4 records in it
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path, "all", RandomSampler(false, 10), false, {}, some_cache);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheManifestCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a Manifest Dataset, this file_path has 2 records in it
  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", RandomSampler(), {}, false, some_cache);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheCifar10CApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10), some_cache);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheCifar100CApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a Cifar100 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", RandomSampler(false, 10), some_cache);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheVocCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a VOC Dataset, this folder_path has 9 records in it
  std::string folder_path = datasets_root_path_ + "/testVOC2012/";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, false, RandomSampler(), some_cache);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 18);

  // Manually terminate the pipeline
  iter->Stop();
}
