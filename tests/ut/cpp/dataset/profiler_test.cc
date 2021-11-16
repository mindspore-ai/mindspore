/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::MsLogLevel::INFO;

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
  ds = ds->Map({one_hot}, {"label"}, {"label"}, {"label"});
  EXPECT_NE(ds, nullptr);

  ds = ds->Take(4);
  EXPECT_NE(ds, nullptr);

  ds = ds->Batch(2, true);
  EXPECT_NE(ds, nullptr);

  // No columns are specified, use all columns
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

  // No columns are specified, use all columns
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
}  // namespace test
}  // namespace dataset
}  // namespace mindspore
