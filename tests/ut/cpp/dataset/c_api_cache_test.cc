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
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/vision.h"

using namespace mindspore::dataset;

// Helper function to get the session id from SESSION_ID env variable
Status GetSessionFromEnv(session_id_type *session_id);

class MindDataTestCacheOp : public UT::DatasetOpTesting {
 public:
  void SetUp() override { DatasetOpTesting::SetUp(); }
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
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for ImageFolderNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheCApiNestedCache) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create an ImageFolder Dataset, this folder_path only has 2 images in it
  std::string folder_path = datasets_root_path_ + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, {}, some_cache);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  EXPECT_NE(decode_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({decode_op}, {}, {}, {"image"}, some_cache);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now in the cache_error_pass would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheImageFolderCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create an ImageFolder Dataset, this folder_path only has 2 images in it
  std::string folder_path = datasets_root_path_ + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, {}, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
    Coco(folder_path, annotation_file_path, "Detection", false, std::make_shared<RandomSampler>(), some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10), some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds =
    CelebA(folder_path, "all", std::make_shared<RandomSampler>(false, 10), false, {}, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds = Manifest(file_path, "train", std::make_shared<RandomSampler>(), {}, false, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10), some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", std::make_shared<RandomSampler>(false, 10), some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, false, std::make_shared<RandomSampler>(), some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 18);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheAlbumCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"image", "label", "id"};
  // Create a Album Dataset, 7 records in it
  std::shared_ptr<Dataset> ds =
    Album(folder_path, schema_file, column_names, false, std::make_shared<RandomSampler>(), some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 14);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheRandomDataCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();

  schema->add_column("image", mindspore::DataType::kNumberTypeUInt8, {2});
  schema->add_column("label", mindspore::DataType::kNumberTypeUInt8, {1});
  std::shared_ptr<Dataset> ds = RandomData(8, schema, {}, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 16);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheTFRecordCApi1) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a TFRecord Dataset, this file_path has 3 records in it
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::string schema_path = datasets_root_path_ + "/test_tf_file_3_images2/datasetSchema.json";
  std::shared_ptr<Dataset> ds =
    TFRecord({file_path}, schema_path, {"image"}, 0, ShuffleMode::kFalse, 1, 0, false, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheTFRecordCApi2) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a TFRecord Dataset, this file_path has 3 records in it
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::string schema_path = datasets_root_path_ + "/test_tf_file_3_images2/datasetSchema.json";

  // In this one, the TFRecord dataset will be given sharding configuration, however since a cache is
  // used, the tree prepare should undo the sharding configuration and instead, a distributed
  // sampler will be chosen with the same shard config.
  // With only 3 records shard into 3, we expect only 1 record returned for this shard
  // However, the sharding will be done by the sampler, not by the TFRecord leaf node
  // In this case, it is a row-based sharding, not the file-based sharding that would happen if
  // there was not any cache.
  std::shared_ptr<Dataset> ds =
    TFRecord({file_path}, schema_path, {"image"}, 0, ShuffleMode::kFalse, 3, 0, false, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheTFRecordCApi3) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a TFRecord Dataset, this file_path has 3 records in it
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::string schema_path = datasets_root_path_ + "/test_tf_file_3_images2/datasetSchema.json";

  // In this one, a num_samples argument is given.
  // In this case, a sequential sampler would be chosen with the same num_samples argument.
  // The samples will be selected by the sequential sampler, not by the TFRecord leaf node.
  std::shared_ptr<Dataset> ds =
    TFRecord({file_path}, schema_path, {"image"}, 2, ShuffleMode::kFalse, 1, 0, false, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheTextfileCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a TextFile Dataset, this file_path has 3 records in it
  std::string file_path = datasets_root_path_ + "/testTextFileDataset/1.txt";

  // In this one, a num_samples=2 argument is given.
  // In this case, a sequential sampler would be chosen with the same num_samples argument.
  // The samples will be selected by the sequential sampler, not by the TextFile leaf node.
  std::shared_ptr<Dataset> ds = TextFile({file_path}, 2, ShuffleMode::kGlobal, 1, 0, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheCsvCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a CSV Dataset, this file_path has 3 records in it
  std::string file_path = datasets_root_path_ + "/testCSV/1.csv";
  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};

  // In this one, a num_samples=2 argument is given.
  // In this case, a sequential sampler would be chosen with the same num_samples argument.
  // The samples will be selected by the sequential sampler, not by the CSV leaf node.
  std::shared_ptr<Dataset> ds = CSV({file_path}, ',', {}, column_names, 2, ShuffleMode::kFalse, 1, 0, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheClueCApi) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create a CLUE Dataset, this file_path has 3 records in it
  std::string file_path = datasets_root_path_ + "/testCLUE/afqmc/train.json";
  std::string task = "AFQMC";
  std::string usage = "train";

  // In this one, a num_samples=2 argument is given.
  // In this case, a sequential sampler would be chosen with the same num_samples argument.
  // The samples will be selected by the sequential sampler, not by the CLUE leaf node.
  std::shared_ptr<Dataset> ds = CLUE({file_path}, task, usage, 2, ShuffleMode::kFalse, 1, 0, some_cache);
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
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCApiCacheShare1) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create an ImageFolder Dataset, this folder_path only has 2 images in it
  std::string folder_path = datasets_root_path_ + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, {}, some_cache);
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, {}, some_cache);
  EXPECT_NE(ds2, nullptr);

  // Create and launch the Execution Tree for ds1
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter1->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter1->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  // Manually terminate the pipeline
  iter1->Stop();

  // Create and launch the Execution Tree for ds2
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);
  // Iterate the dataset and get each row
  iter2->GetNextRow(&row);

  i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter2->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter2->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCApiCacheShare2) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create an ImageFolder Dataset, this folder_path only has 2 images in it
  std::string folder_path = datasets_root_path_ + "/testImageNetData/train/";
  // The first pipeline is ImageFolder with RandomSampler, the second pipeline is ImageFolder with SequentialSampler
  // Since sampler does not influence the data in the source, these two pipelines can share a common cache.
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(), {}, {}, some_cache);
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 =
    ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(), {}, {}, some_cache);
  EXPECT_NE(ds2, nullptr);

  // Create and launch the Execution Tree for ds1
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter1->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter1->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  // Manually terminate the pipeline
  iter1->Stop();

  // Create and launch the Execution Tree for ds2
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);
  // Iterate the dataset and get each row
  iter2->GetNextRow(&row);

  i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter2->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter2->Stop();
}

TEST_F(MindDataTestCacheOp, DISABLED_TestCApiCacheShareFailure1) {
  session_id_type env_session;
  Status s = GetSessionFromEnv(&env_session);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, true);
  EXPECT_NE(some_cache, nullptr);

  // Create an ImageFolder Dataset, this folder_path only has 2 images in it
  std::string folder_path = datasets_root_path_ + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(), {}, {}, some_cache);
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, {}, some_cache);
  EXPECT_NE(ds2, nullptr);

  // Create and launch the Execution Tree for ds1
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter1->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter1->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  // Manually terminate the pipeline
  iter1->Stop();

  // Re-use a cache for the second pipeline would fail
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_EQ(iter2, nullptr);
}
