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
#include <string>
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/cache/cache_client.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/data_schema.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::dataset::CacheClient;
using mindspore::dataset::TaskGroup;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

// Helper function to get the session id from SESSION_ID env variable
Status GetSessionFromEnv(session_id_type *session_id) {
  RETURN_UNEXPECTED_IF_NULL(session_id);
  if (const char *session_env = std::getenv("SESSION_ID")) {
    std::string session_id_str(session_env);
    try {
      *session_id = std::stoul(session_id_str);
    } catch (const std::exception &e) {
      std::string err_msg = "Invalid numeric value for session id in env var: " + session_id_str;
      return Status(StatusCode::kMDSyntaxError, err_msg);
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Test case requires a session id to be provided via SESSION_ID environment variable.");
  }
  return Status::OK();
}

class MindDataTestCacheOp : public UT::DatasetOpTesting {
 public:
  void SetUp() override {
    DatasetOpTesting::SetUp();
    GlobalInit();
  }
};

TEST_F(MindDataTestCacheOp, DISABLED_TestCacheServer) {
  Status rc;
  CacheClient::Builder builder;
  session_id_type env_session;
  rc = GetSessionFromEnv(&env_session);
  ASSERT_TRUE(rc.IsOk());

  // use arbitrary session of 1, size of 0, spilling// is true
  builder.SetSessionId(env_session).SetCacheMemSz(0).SetSpill(true);
  std::shared_ptr<CacheClient> myClient;
  rc = builder.Build(&myClient);
  ASSERT_TRUE(rc.IsOk());
  // cksum value of 1 for CreateCache here...normally you do not directly create a cache and the cksum arg is generated.
  rc = myClient->CreateCache(1, true);
  ASSERT_TRUE(rc.IsOk());
  std::cout << *myClient << std::endl;

  // Create a schema using the C api's
  int32_t rank = 0;  // not used
  std::unique_ptr<DataSchema> testSchema = std::make_unique<DataSchema>();
  // 2 columns. First column is an "image" 640,480,3
  TensorShape c1Shape({640, 480, 3});
  ColDescriptor c1("image", DataType(DataType::DE_INT8), TensorImpl::kFlexible,
                   rank,  // not used
                   &c1Shape);
  // Column 2 will just be a scalar label number
  TensorShape c2Shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor c2("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, rank, &c2Shape);

  testSchema->AddColumn(c1);
  testSchema->AddColumn(c2);

  std::unordered_map<std::string, int32_t> map;
  rc = testSchema->GetColumnNameMap(&map);
  ASSERT_TRUE(rc.IsOk());

  // Test the CacheSchema api
  rc = myClient->CacheSchema(map);
  ASSERT_TRUE(rc.IsOk());

  // Create a tensor, take a snapshot and restore it back, and compare.
  std::shared_ptr<Tensor> t;
  Tensor::CreateEmpty(TensorShape({2, 3}), DataType(DataType::DE_UINT64), &t);
  t->SetItemAt<uint64_t>({0, 0}, 1);
  t->SetItemAt<uint64_t>({0, 1}, 2);
  t->SetItemAt<uint64_t>({0, 2}, 3);
  t->SetItemAt<uint64_t>({1, 0}, 4);
  t->SetItemAt<uint64_t>({1, 1}, 5);
  t->SetItemAt<uint64_t>({1, 2}, 6);
  std::cout << *t << std::endl;
  TensorTable tbl;
  TensorRow row;
  row.push_back(t);
  int64_t row_id;
  rc = myClient->WriteRow(row, &row_id);
  ASSERT_TRUE(rc.IsOk());

  // Switch off build phase.
  rc = myClient->BuildPhaseDone();
  ASSERT_TRUE(rc.IsOk());

  // Now restore from cache.
  row.clear();
  rc = myClient->GetRows({row_id}, &tbl);
  row = tbl.front();
  ASSERT_TRUE(rc.IsOk());
  auto r = row.front();
  std::cout << *r << std::endl;
  // Compare
  bool cmp = (*t == *r);
  ASSERT_TRUE(cmp);

  // Get back the schema and verify
  std::unordered_map<std::string, int32_t> map_out;
  rc = myClient->FetchSchema(&map_out);
  ASSERT_TRUE(rc.IsOk());
  cmp = (map_out == map);
  ASSERT_TRUE(cmp);

  rc = myClient->DestroyCache();
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestCacheOp, DISABLED_TestConcurrencyRequest) {
  // Clear the rc of the master thread if any
  (void)TaskManager::GetMasterThreadRc();
  TaskGroup vg;
  Status rc;

  session_id_type env_session;
  rc = GetSessionFromEnv(&env_session);
  ASSERT_TRUE(rc.IsOk());

  // use arbitrary session of 1, size 1, spilling is true
  CacheClient::Builder builder;
  // use arbitrary session of 1, size of 0, spilling// is true
  builder.SetSessionId(env_session).SetCacheMemSz(1).SetSpill(true);
  std::shared_ptr<CacheClient> myClient;
  rc = builder.Build(&myClient);
  ASSERT_TRUE(rc.IsOk());
  // cksum value of 1 for CreateCache here...normally you do not directly create a cache and the cksum arg is generated.
  rc = myClient->CreateCache(1, true);
  ASSERT_TRUE(rc.IsOk());
  std::cout << *myClient << std::endl;
  std::shared_ptr<Tensor> t;
  Tensor::CreateEmpty(TensorShape({2, 3}), DataType(DataType::DE_UINT64), &t);
  t->SetItemAt<uint64_t>({0, 0}, 1);
  t->SetItemAt<uint64_t>({0, 1}, 2);
  t->SetItemAt<uint64_t>({0, 2}, 3);
  t->SetItemAt<uint64_t>({1, 0}, 4);
  t->SetItemAt<uint64_t>({1, 1}, 5);
  t->SetItemAt<uint64_t>({1, 2}, 6);
  TensorTable tbl;
  TensorRow row;
  row.push_back(t);
  // Cache tensor row t 5000 times using 10 threads.
  for (auto k = 0; k < 10; ++k) {
    Status vg_rc = vg.CreateAsyncTask("Test agent", [&myClient, &row]() -> Status {
      TaskManager::FindMe()->Post();
      for (auto i = 0; i < 500; i++) {
        RETURN_IF_NOT_OK(myClient->WriteRow(row));
      }
      return Status::OK();
    });
    ASSERT_TRUE(vg_rc.IsOk());
  }
  ASSERT_TRUE(vg.join_all().IsOk());
  ASSERT_TRUE(vg.GetTaskErrorIfAny().IsOk());
  rc = myClient->BuildPhaseDone();
  ASSERT_TRUE(rc.IsOk());
  // Get statistics from the server.
  CacheServiceStat stat{};
  rc = myClient->GetStat(&stat);
  ASSERT_TRUE(rc.IsOk());
  std::cout << stat.min_row_id << ":" << stat.max_row_id << ":" << stat.num_mem_cached << ":" << stat.num_disk_cached
            << "\n";
  // Expect there are 5000 rows there.
  EXPECT_EQ(5000, stat.max_row_id - stat.min_row_id + 1);
  // Get them all back using row id and compare with tensor t.
  for (auto i = stat.min_row_id; i <= stat.max_row_id; ++i) {
    tbl.clear();
    row.clear();
    rc = myClient->GetRows({i}, &tbl);
    ASSERT_TRUE(rc.IsOk());
    row = tbl.front();
    auto r = row.front();
    bool cmp = (*t == *r);
    ASSERT_TRUE(cmp);
  }
  rc = myClient->DestroyCache();
  ASSERT_TRUE(rc.IsOk());
}

// Simple test with a repeated cache op over random data producer
//
//     RepeatOp
//        |
//     CacheOp
//        |
//   RandomDataOp
//
TEST_F(MindDataTestCacheOp, DISABLED_TestRandomDataCache1) {
  // Clear the rc of the master thread if any
  (void)TaskManager::GetMasterThreadRc();
  Status rc;
  int32_t rank = 0;  // not used

  session_id_type env_session;
  rc = GetSessionFromEnv(&env_session);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "UT test TestRandomDataCache1";
  // Start with an empty execution tree
  auto myTree = std::make_shared<ExecutionTree>();

  // Create a schema using the C api's
  std::unique_ptr<DataSchema> testSchema = std::make_unique<DataSchema>();

  // 2 columns. First column is an "image" 640,480,3
  TensorShape c1Shape({640, 480, 3});
  ColDescriptor c1("image", DataType(DataType::DE_INT8), TensorImpl::kFlexible,
                   rank,  // not used
                   &c1Shape);

  // Column 2 will just be a scalar label number
  TensorShape c2Shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor c2("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, rank, &c2Shape);

  testSchema->AddColumn(c1);
  testSchema->AddColumn(c2);

  // RandomDataOp
  std::shared_ptr<RandomDataOp> myRandomDataOp;
  rc = RandomDataOp::Builder()
         .SetRowsPerBuffer(4)
         .SetNumWorkers(4)
         .SetDataSchema(std::move(testSchema))
         .SetTotalRows(50)  // 50 samples for now
         .Build(&myRandomDataOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myRandomDataOp);
  ASSERT_TRUE(rc.IsOk());

  // CacheOp
  // size of 0, spilling is true
  CacheClient::Builder builder;
  builder.SetSessionId(env_session).SetCacheMemSz(0).SetSpill(true);
  std::shared_ptr<CacheClient> myClient;
  rc = builder.Build(&myClient);
  ASSERT_TRUE(rc.IsOk());
  std::shared_ptr<CacheOp> myCacheOp;

  int64_t num_samples = 0;
  int64_t start_index = 0;
  auto seq_sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
  rc = CacheOp::Builder()
         .SetNumWorkers(5)
         .SetClient(myClient)
         .SetRowsPerBuffer(4)
         .SetSampler(std::move(seq_sampler))
         .Build(&myCacheOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myCacheOp);
  ASSERT_TRUE(rc.IsOk());

  // RepeatOp
  uint32_t numRepeats = 4;
  std::shared_ptr<RepeatOp> myRepeatOp;
  rc = RepeatOp::Builder(numRepeats).Build(&myRepeatOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myRepeatOp);
  ASSERT_TRUE(rc.IsOk());

  // Assign tree relations and root
  myCacheOp->set_total_repeats(numRepeats);
  myCacheOp->set_num_repeats_per_epoch(numRepeats);
  rc = myRepeatOp->AddChild(myCacheOp);
  ASSERT_TRUE(rc.IsOk());
  // Always set to 1 under a CacheOp because we read from it only once. The CacheOp is the one that repeats.
  myRandomDataOp->set_total_repeats(1);
  myRandomDataOp->set_num_repeats_per_epoch(1);
  rc = myCacheOp->AddChild(myRandomDataOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssignRoot(myRepeatOp);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration";
  rc = myTree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  // quick check to see what tree looks like
  std::ostringstream ss;
  ss << *myTree;  // some funny const error if I try to write directly to ms log stream
  MS_LOG(INFO) << "Here's the tree:\n" << ss.str();

  std::cout << *myClient << std::endl;

  rc = myTree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  ASSERT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    // Don't display these rows, just count them
    MS_LOG(INFO) << "Row fetched #: " << rowCount;
    rc = dI.FetchNextTensorRow(&tensorList);
    ASSERT_TRUE(rc.IsOk());
    rowCount++;
  }
  ASSERT_EQ(rowCount, 200);
  rc = myClient->DestroyCache();
  ASSERT_TRUE(rc.IsOk());
}

//// Simple test with a repeated cache op over random data producer.
//// This one will exceed memory and require a spill.
////
////     RepeatOp
////        |
////     CacheOp
////        |
////   RandomDataOp
////
TEST_F(MindDataTestCacheOp, DISABLED_TestRandomDataCacheSpill) {
  // Clear the rc of the master thread if any
  (void)TaskManager::GetMasterThreadRc();
  Status rc;
  int32_t rank = 0;  // not used
  MS_LOG(INFO) << "UT test TestRandomDataCacheSpill";

  session_id_type env_session;
  rc = GetSessionFromEnv(&env_session);
  ASSERT_TRUE(rc.IsOk());

  // Start with an empty execution tree
  auto myTree = std::make_shared<ExecutionTree>();

  // Create a schema using the C api's
  std::unique_ptr<DataSchema> testSchema = std::make_unique<DataSchema>();

  // 2 columns. First column is an "image" 640,480,3
  TensorShape c1Shape({640, 480, 3});
  ColDescriptor c1("image", DataType(DataType::DE_INT8), TensorImpl::kFlexible,
                   rank,  // not used
                   &c1Shape);

  // Column 2 will just be a scalar label number
  TensorShape c2Shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor c2("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, rank, &c2Shape);

  testSchema->AddColumn(c1);
  testSchema->AddColumn(c2);

  // RandomDataOp
  std::shared_ptr<RandomDataOp> myRandomDataOp;
  rc = RandomDataOp::Builder()
         .SetRowsPerBuffer(2)
         .SetNumWorkers(4)
         .SetDataSchema(std::move(testSchema))
         .SetTotalRows(10)
         .Build(&myRandomDataOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myRandomDataOp);
  ASSERT_TRUE(rc.IsOk());

  // CacheOp
  int64_t num_samples = 0;
  int64_t start_index = 0;
  auto seq_sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);
  CacheClient::Builder builder;
  builder.SetSessionId(env_session).SetCacheMemSz(4).SetSpill(true);
  std::shared_ptr<CacheClient> myClient;
  rc = builder.Build(&myClient);
  ASSERT_TRUE(rc.IsOk());
  std::shared_ptr<CacheOp> myCacheOp;
  rc = CacheOp::Builder()
         .SetNumWorkers(4)
         .SetClient(myClient)
         .SetRowsPerBuffer(3)
         .SetSampler(std::move(seq_sampler))
         .Build(&myCacheOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myCacheOp);
  ASSERT_TRUE(rc.IsOk());

  // RepeatOp
  uint32_t numRepeats = 4;
  std::shared_ptr<RepeatOp> myRepeatOp;
  rc = RepeatOp::Builder(numRepeats).Build(&myRepeatOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myRepeatOp);
  ASSERT_TRUE(rc.IsOk());

  // Assign tree relations and root
  myCacheOp->set_total_repeats(numRepeats);
  myCacheOp->set_num_repeats_per_epoch(numRepeats);
  rc = myRepeatOp->AddChild(myCacheOp);
  ASSERT_TRUE(rc.IsOk());
  // Always set to 1 under a CacheOp because we read from it only once. The CacheOp is the one that repeats.
  myRandomDataOp->set_total_repeats(1);
  myRandomDataOp->set_num_repeats_per_epoch(1);
  rc = myCacheOp->AddChild(myRandomDataOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssignRoot(myRepeatOp);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration";
  rc = myTree->Prepare();
  ASSERT_TRUE(rc.IsOk());

  std::cout << *myClient << std::endl;

  rc = myTree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  ASSERT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    // Don't display these rows, just count them
    MS_LOG(INFO) << "Row fetched #: " << rowCount;
    rc = dI.FetchNextTensorRow(&tensorList);
    ASSERT_TRUE(rc.IsOk());
    rowCount++;
  }
  ASSERT_EQ(rowCount, 40);
  rc = myClient->DestroyCache();
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestCacheOp, DISABLED_TestImageFolderCacheMerge) {
  // Clear the rc of the master thread if any
  (void)TaskManager::GetMasterThreadRc();
  Status rc;
  int64_t num_samples = 0;
  int64_t start_index = 0;

  session_id_type env_session;
  rc = GetSessionFromEnv(&env_session);
  ASSERT_TRUE(rc.IsOk());

  auto seq_sampler = std::make_shared<SequentialSamplerRT>(num_samples, start_index);

  CacheClient::Builder ccbuilder;
  ccbuilder.SetSessionId(env_session).SetCacheMemSz(0).SetSpill(true);
  std::shared_ptr<CacheClient> myClient;
  rc = ccbuilder.Build(&myClient);
  ASSERT_TRUE(rc.IsOk());

  std::shared_ptr<CacheLookupOp> myLookupOp;
  rc = CacheLookupOp::Builder().SetNumWorkers(4).SetClient(myClient).SetSampler(seq_sampler).Build(&myLookupOp);
  std::shared_ptr<CacheMergeOp> myMergeOp;
  rc = CacheMergeOp::Builder().SetNumWorkers(4).SetClient(myClient).Build(&myMergeOp);

  std::shared_ptr<ImageFolderOp> so;
  ImageFolderOp::Builder builder;
  builder.SetOpConnectorSize(3)
    .SetNumWorkers(3)
    .SetRowsPerBuffer(2)
    .SetExtensions({".jpg", ".JPEG"})
    .SetRecursive(true)
    .SetImageFolderDir(datasets_root_path_ + "/testPK/data");
  rc = builder.Build(&so);
  so->SetSampler(myLookupOp);
  ASSERT_TRUE(rc.IsOk());

  // RepeatOp
  uint32_t numRepeats = 4;
  std::shared_ptr<RepeatOp> myRepeatOp;
  rc = RepeatOp::Builder(numRepeats).Build(&myRepeatOp);
  ASSERT_TRUE(rc.IsOk());

  auto myTree = std::make_shared<ExecutionTree>();
  rc = myTree->AssociateNode(so);
  ASSERT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myLookupOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssociateNode(myMergeOp);
  ASSERT_TRUE(rc.IsOk());

  rc = myTree->AssociateNode(myRepeatOp);
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->AssignRoot(myRepeatOp);
  ASSERT_TRUE(rc.IsOk());

  myMergeOp->set_total_repeats(numRepeats);
  myMergeOp->set_num_repeats_per_epoch(numRepeats);
  rc = myRepeatOp->AddChild(myMergeOp);
  ASSERT_TRUE(rc.IsOk());
  myLookupOp->set_total_repeats(numRepeats);
  myLookupOp->set_num_repeats_per_epoch(numRepeats);
  rc = myMergeOp->AddChild(myLookupOp);
  ASSERT_TRUE(rc.IsOk());
  so->set_total_repeats(numRepeats);
  so->set_num_repeats_per_epoch(numRepeats);
  rc = myMergeOp->AddChild(so);
  ASSERT_TRUE(rc.IsOk());

  rc = myTree->Prepare();
  ASSERT_TRUE(rc.IsOk());
  rc = myTree->Launch();
  ASSERT_TRUE(rc.IsOk());
  // Start the loop of reading tensors from our pipeline
  DatasetIterator dI(myTree);
  TensorRow tensorList;
  rc = dI.FetchNextTensorRow(&tensorList);
  ASSERT_TRUE(rc.IsOk());
  int rowCount = 0;
  while (!tensorList.empty()) {
    rc = dI.FetchNextTensorRow(&tensorList);
    ASSERT_TRUE(rc.IsOk());
    if (rc.IsError()) {
      std::cout << rc << std::endl;
      break;
    }
    rowCount++;
  }
  ASSERT_EQ(rowCount, 176);
  std::cout << "Row count : " << rowCount << std::endl;
  rc = myClient->DestroyCache();
  ASSERT_TRUE(rc.IsOk());
}
