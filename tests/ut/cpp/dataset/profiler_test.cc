/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <chrono>
#include <thread>
#include "common/common.h"
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;

namespace mindspore {
namespace dataset {
namespace test {
class MindDataTestProfiler : public UT::DatasetOpTesting {
 protected:
  MindDataTestProfiler() {}
  Status DeleteFiles(int file_id = 0) {
    std::shared_ptr<ProfilingManager> profiler_manager = GlobalContext::profiling_manager();
    std::string pipeline_file = "./pipeline_profiling_" + std::to_string(file_id) + ".json";
    std::string cpu_util_file = "./minddata_cpu_utilization_" + std::to_string(file_id) + ".json";
    std::string dataset_iterator_file = "./dataset_iterator_profiling_" + std::to_string(file_id) + ".txt";
    if (remove(pipeline_file.c_str()) == 0 && remove(cpu_util_file.c_str()) == 0 &&
        remove(dataset_iterator_file.c_str()) == 0) {
      return Status::OK();
    } else {
      RETURN_STATUS_UNEXPECTED("Error deleting profiler files");
    }
  }
  std::shared_ptr<Dataset> set_dataset(int32_t op_input) {
    std::string folder_path = datasets_root_path_ + "/testPK/data/";
    int64_t num_samples = 20;
    std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, num_samples));
    EXPECT_NE(ds, nullptr);

    ds = ds->Shuffle(op_input);
    EXPECT_NE(ds, nullptr);

    // Create objects for the tensor ops
    std::shared_ptr<TensorTransform> one_hot = std::make_shared<transforms::OneHot>(op_input);
    EXPECT_NE(one_hot, nullptr);

    // Create a Map operation, this will automatically add a project after map
    ds = ds->Map({one_hot}, {"label"}, {"label"});
    EXPECT_NE(ds, nullptr);

    ds = ds->Project({"label"});

    ds = ds->Take(op_input);
    EXPECT_NE(ds, nullptr);

    ds = ds->Batch(op_input, true);
    EXPECT_NE(ds, nullptr);

    int repeat_num = 10;
    ds = ds->Repeat(repeat_num);
    EXPECT_NE(ds, nullptr);

    return ds;
  }
};

/// Feature: MindData Profiling Support
/// Description: Test MindData Profiling with profiling enabled for pipeline with ImageFolder
/// Expectation: Profiling files are created.
TEST_F(MindDataTestProfiler, TestProfilerManager1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestProfilerManager1.";

  // Enable profiler and check
  common::SetEnv("RANK_ID", "1");
  std::shared_ptr<ProfilingManager> profiler_manager = GlobalContext::profiling_manager();
  EXPECT_OK(profiler_manager->Init());
  EXPECT_OK(profiler_manager->Start());
  EXPECT_TRUE(profiler_manager->IsProfilingEnable());

  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  ds = ds->Shuffle(4);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot, nullptr);

  // Create a Map operation, this will automatically add a project after map
  ds = ds->Map({one_hot}, {"label"}, {"label"});
  EXPECT_NE(ds, nullptr);

  ds = ds->Take(4);
  EXPECT_NE(ds, nullptr);

  ds = ds->Batch(2, true);
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 2);
  // Manually terminate the pipeline
  iter->Stop();

  // Stop MindData Profiling and save output files to current working directory
  EXPECT_OK(profiler_manager->Stop());
  EXPECT_FALSE(profiler_manager->IsProfilingEnable());
  EXPECT_OK(profiler_manager->Save("."));

  // File_id is expected to equal RANK_ID
  EXPECT_OK(DeleteFiles(1));
}

/// Feature: MindData Profiling Support
/// Description: Test MindData Profiling with profiling enabled for pipeline with Mnist
/// Expectation: Profiling files are created.
TEST_F(MindDataTestProfiler, TestProfilerManager2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestProfilerManager2.";

  // Enable profiler and check
  common::SetEnv("RANK_ID", "2");
  std::shared_ptr<ProfilingManager> profiler_manager = GlobalContext::profiling_manager();
  EXPECT_OK(profiler_manager->Init());
  EXPECT_OK(profiler_manager->Start());
  EXPECT_TRUE(profiler_manager->IsProfilingEnable());

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  ds = ds->Skip(1);
  EXPECT_NE(ds, nullptr);

  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  ds = ds->Batch(2, false);
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 2);
  // Manually terminate the pipeline
  iter->Stop();

  // Stop MindData Profiling and save output files to current working directory
  EXPECT_OK(profiler_manager->Stop());
  EXPECT_FALSE(profiler_manager->IsProfilingEnable());
  EXPECT_OK(profiler_manager->Save("."));

  // File_id is expected to equal RANK_ID
  EXPECT_OK(DeleteFiles(2));
}

/// Feature: MindData Profiling Support
/// Description: Test MindData Profiling GetByEpoch Methods
/// Expectation: Results are successfully outputted.
TEST_F(MindDataTestProfiler, TestProfilerManagerByEpoch) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestProfilerManagerByEpoch.";

  // Enable profiler and check
  common::SetEnv("RANK_ID", "2");
  GlobalContext::config_manager()->set_monitor_sampling_interval(10);
  std::shared_ptr<ProfilingManager> profiler_manager = GlobalContext::profiling_manager();
  EXPECT_OK(profiler_manager->Init());
  EXPECT_OK(profiler_manager->Start());
  EXPECT_TRUE(profiler_manager->IsProfilingEnable());

  std::shared_ptr<Dataset> ds = set_dataset(20);

  std::shared_ptr<Iterator> iter = ds->CreateIterator(3);
  EXPECT_NE(iter, nullptr);

  std::vector<uint8_t> cpu_result;
  std::vector<uint16_t> op_result;
  std::vector<int32_t> connector_result;
  std::vector<int32_t> time_result;
  float_t queue_result;

  // Note: These Get* calls fail since epoch number cannot be 0.
  EXPECT_ERROR(profiler_manager->GetUserCpuUtilByEpoch(0, &cpu_result));
  EXPECT_ERROR(profiler_manager->GetBatchTimeByEpoch(0, &time_result));

  std::vector<mindspore::MSTensor> row;
  for (int i = 0; i < 3; i++) {
    // Iterate the dataset and get each row
    ASSERT_OK(iter->GetNextRow(&row));
    while (row.size() != 0) {
      ASSERT_OK(iter->GetNextRow(&row));
    }
  }
  // Check iteration failure after finishing the num_epochs
  EXPECT_ERROR(iter->GetNextRow(&row));
  // Manually terminate the pipeline
  iter->Stop();

  for (int i = 1; i < 4; i++) {
    ASSERT_OK(profiler_manager->GetUserCpuUtilByEpoch(i, &cpu_result));
    ASSERT_OK(profiler_manager->GetUserCpuUtilByEpoch(i - 1, i, &op_result));
    ASSERT_OK(profiler_manager->GetSysCpuUtilByEpoch(i, &cpu_result));
    ASSERT_OK(profiler_manager->GetSysCpuUtilByEpoch(i - 1, i, &op_result));
    // Epoch is 1 for each iteration and 20 steps for each epoch, so the output size are expected to be 20
    ASSERT_OK(profiler_manager->GetBatchTimeByEpoch(i, &time_result));
    EXPECT_EQ(time_result.size(), 10);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetPipelineTimeByEpoch(i, &time_result));
    EXPECT_EQ(time_result.size(), 10);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetPushTimeByEpoch(i, &time_result));
    EXPECT_EQ(time_result.size(), 10);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetConnectorSizeByEpoch(i, &connector_result));
    EXPECT_EQ(connector_result.size(), 10);
    connector_result.clear();
    ASSERT_OK(profiler_manager->GetConnectorCapacityByEpoch(i, &connector_result));
    EXPECT_EQ(connector_result.size(), 10);
    connector_result.clear();
    ASSERT_OK(profiler_manager->GetConnectorSizeByEpoch(i - 1, i, &connector_result));
    EXPECT_GT(connector_result.size(), 0);  // Connector size is expected to be greater than 0
    connector_result.clear();
    ASSERT_OK(profiler_manager->GetEmptyQueueFrequencyByEpoch(i, &queue_result));
    EXPECT_GE(queue_result, 0);
    EXPECT_LE(queue_result, 1);
  }
  ASSERT_ERROR(profiler_manager->GetUserCpuUtilByEpoch(4, &cpu_result));  // Check there is no epoch 4

  int num = profiler_manager->GetNumOfProfiledEpochs();
  EXPECT_EQ(num, 3);

  // Stop MindData Profiling and save output files to current working directory
  EXPECT_OK(profiler_manager->Stop());
  EXPECT_FALSE(profiler_manager->IsProfilingEnable());
  EXPECT_OK(profiler_manager->Save("."));

  // File_id is expected to equal RANK_ID
  EXPECT_OK(DeleteFiles(2));
}

/// Feature: MindData Profiling Support
/// Description: Test MindData Profiling GetByStep Methods
/// Expectation: Results are successfully outputted.
TEST_F(MindDataTestProfiler, TestProfilerManagerByStep) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestProfilerManagerByStep.";
  // Enable profiler and check
  common::SetEnv("RANK_ID", "2");
  GlobalContext::config_manager()->set_monitor_sampling_interval(10);
  std::shared_ptr<ProfilingManager> profiler_manager = GlobalContext::profiling_manager();
  EXPECT_OK(profiler_manager->Init());
  EXPECT_OK(profiler_manager->Start());
  EXPECT_TRUE(profiler_manager->IsProfilingEnable());

  std::shared_ptr<Dataset> ds = set_dataset(20);

  std::shared_ptr<Iterator> iter = ds->CreateIterator(3);
  EXPECT_NE(iter, nullptr);

  std::vector<uint8_t> cpu_result;
  std::vector<uint16_t> op_result;
  std::vector<int32_t> connector_result;
  std::vector<int32_t> time_result;
  float_t queue_result;

  uint64_t i = 0;
  ASSERT_ERROR(
    profiler_manager->GetUserCpuUtilByStep(i, i, &cpu_result));  // Fail in TimeIntervalForStepRange for start_step = 0
  ASSERT_ERROR(profiler_manager->GetBatchTimeByStep(
    i, i + 2, &time_result));  // Fail in GetRecordEntryFieldValue for end_step > total_steps
  ASSERT_ERROR(profiler_manager->GetPipelineTimeByStep(
    i + 2, i, &time_result));  // Fail in GetRecordEntryFieldValue for start_step > total_steps
  ASSERT_ERROR(profiler_manager->GetPushTimeByStep(
    i + 1, i, &time_result));  // Fail in GetRecordEntryFieldValue for start_step > end_steps

  std::vector<mindspore::MSTensor> row;
  for (int i = 0; i < 3; i++) {
    // Iterate the dataset and get each row
    ASSERT_OK(iter->GetNextRow(&row));
    while (row.size() != 0) {
      ASSERT_OK(iter->GetNextRow(&row));
    }
  }
  // Manually terminate the pipeline
  iter->Stop();

  // There are 3 epochs and 10 samplers for each epoch, 3x10=30 steps in total
  for (int i = 1; i < 31; i++) {
    ASSERT_OK(profiler_manager->GetUserCpuUtilByStep(i, i, &cpu_result));
    ASSERT_OK(profiler_manager->GetSysCpuUtilByStep(i, i, &cpu_result));
    // Step is 1 for each iteration, so the output size is expected to be 1
    ASSERT_OK(profiler_manager->GetBatchTimeByStep(i, i, &time_result));
    EXPECT_EQ(time_result.size(), 1);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetPipelineTimeByStep(i, i, &time_result));
    EXPECT_EQ(time_result.size(), 1);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetPushTimeByStep(i, i, &time_result));
    EXPECT_EQ(time_result.size(), 1);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetConnectorSizeByStep(i, i, &connector_result));
    EXPECT_EQ(connector_result.size(), 1);
    connector_result.clear();
    ASSERT_OK(profiler_manager->GetConnectorCapacityByStep(i, i, &connector_result));
    EXPECT_EQ(connector_result.size(), 1);
    connector_result.clear();
    ASSERT_OK(profiler_manager->GetEmptyQueueFrequencyByStep(i, i, &queue_result));
    EXPECT_GE(queue_result, 0);
    EXPECT_LE(queue_result, 1);
  }
  // Iterate by op_id
  for (int i = 0; i < 8; i++) {
    ASSERT_OK(profiler_manager->GetUserCpuUtilByStep(i, i+1, i+1, &op_result));
    ASSERT_OK(profiler_manager->GetSysCpuUtilByStep(i, i+1, i+1, &op_result));
    ASSERT_OK(profiler_manager->GetConnectorSizeByStep(i, i+1, i+1, &connector_result));
    EXPECT_GT(connector_result.size(), 0);  // Connector size is expected to be greater than 0
    connector_result.clear();
  }
  ASSERT_ERROR(profiler_manager->GetUserCpuUtilByStep(8, 9, 9, &op_result));  // Check there is no op_id=8

  int num = profiler_manager->GetNumOfProfiledEpochs();
  EXPECT_EQ(num, 3);

  // Stop MindData Profiling and save output files to current working directory
  EXPECT_OK(profiler_manager->Stop());
  EXPECT_FALSE(profiler_manager->IsProfilingEnable());
  EXPECT_OK(profiler_manager->Save("."));

  // File_id is expected to equal RANK_ID
  EXPECT_OK(DeleteFiles(2));
}

/// Feature: MindData Profiling Support
/// Description: Test MindData Profiling GetByTime Methods
/// Expectation: Results are successfully outputted.
TEST_F(MindDataTestProfiler, TestProfilerManagerByTime) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestProfilerManagerByTime.";

  // Enable profiler and check
  common::SetEnv("RANK_ID", "2");
  GlobalContext::config_manager()->set_monitor_sampling_interval(10);
  std::shared_ptr<ProfilingManager> profiler_manager = GlobalContext::profiling_manager();
  EXPECT_OK(profiler_manager->Init());
  EXPECT_OK(profiler_manager->Start());
  EXPECT_TRUE(profiler_manager->IsProfilingEnable());

  std::shared_ptr<Dataset> ds = set_dataset(20);

  std::shared_ptr<Iterator> iter = ds->CreateIterator(5);
  EXPECT_NE(iter, nullptr);

  std::vector<uint8_t> cpu_result;
  std::vector<uint16_t> op_result;
  std::vector<int32_t> connector_result;
  std::vector<int32_t> time_result;
  float_t queue_result;
  std::vector<uint64_t> ts = {};

  std::vector<mindspore::MSTensor> row;
  for (int i = 0; i < 5; i++) {
    ts.push_back(ProfilingTime::GetCurMilliSecond());
    // Iterate the dataset and get each row
    ASSERT_OK(iter->GetNextRow(&row));
    while (row.size() != 0) {
      ASSERT_OK(iter->GetNextRow(&row));
    }
  }
  ts.push_back(ProfilingTime::GetCurMilliSecond());
  // Manually terminate the pipeline
  iter->Stop();

  for (int i = 1; i < 6; i++) {
    uint64_t start_ts = ts[i - 1];
    uint64_t end_ts = ts[i];
    ASSERT_OK(profiler_manager->GetUserCpuUtilByTime(start_ts, end_ts, &cpu_result));
    ASSERT_OK(profiler_manager->GetUserCpuUtilByTime(i - 1, start_ts, end_ts, &op_result));
    ASSERT_OK(profiler_manager->GetSysCpuUtilByTime(start_ts, end_ts, &cpu_result));
    ASSERT_OK(profiler_manager->GetSysCpuUtilByTime(i - 1, start_ts, end_ts, &op_result));
    ASSERT_OK(profiler_manager->GetBatchTimeByTime(start_ts, end_ts, &time_result));
    EXPECT_GT(time_result.size(), 0);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetPipelineTimeByTime(start_ts, end_ts, &time_result));
    EXPECT_GT(time_result.size(), 0);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetPushTimeByTime(start_ts, end_ts, &time_result));
    EXPECT_GT(time_result.size(), 0);
    time_result.clear();
    ASSERT_OK(profiler_manager->GetConnectorSizeByTime(start_ts, end_ts, &connector_result));
    EXPECT_GT(connector_result.size(), 0);
    connector_result.clear();
    ASSERT_OK(profiler_manager->GetConnectorCapacityByTime(start_ts, end_ts, &connector_result));
    EXPECT_GT(connector_result.size(), 0);
    connector_result.clear();
    ASSERT_OK(profiler_manager->GetConnectorSizeByTime(i - 1, start_ts, end_ts, &connector_result));
    EXPECT_GT(connector_result.size(), 0);  // Connector size is expected to be greater than 0
    connector_result.clear();
    ASSERT_OK(profiler_manager->GetEmptyQueueFrequencyByTime(start_ts, end_ts, &queue_result));
    EXPECT_GE(queue_result, 0);
    EXPECT_LE(queue_result, 1);
  }
  int num = profiler_manager->GetNumOfProfiledEpochs();
  EXPECT_EQ(num, 5);

  // Stop MindData Profiling and save output files to current working directory
  EXPECT_OK(profiler_manager->Stop());
  EXPECT_FALSE(profiler_manager->IsProfilingEnable());
  EXPECT_OK(profiler_manager->Save("."));

  // File_id is expected to equal RANK_ID
  EXPECT_OK(DeleteFiles(2));
}
}  // namespace test
}  // namespace dataset
}  // namespace mindspore
