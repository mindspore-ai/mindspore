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
#include <iostream>
#include <memory>

#include "./prof_reporter.h"
#include "common/common_test.h"
#include "device/ascend/profiling/profiling_manager.h"
#include "./common.h"
#define private public
#include "device/ascend/profiling/plugin_impl.h"
#undef private
#include "device/ascend/profiling/profiling_engine_impl.h"

namespace mindspore {
namespace device {
namespace ascend {
class stubReporter : public Reporter {
 public:
  stubReporter() = default;
  ~stubReporter() = default;

  int Report(const Msprof::Engine::ReporterData *data) override;
  int Flush() override;
};

int stubReporter::Report(const Msprof::Engine::ReporterData *data) { return 0; }

int stubReporter::Flush() { return 0; }

class TestAscendProfiling : public UT::Common {
 public:
  TestAscendProfiling() {}
};

TEST_F(TestAscendProfiling, test_profiling_GetJobId) {
  auto job_id = ProfilingManager::GetInstance().GetJobId();
  printf("get job_id:%ld\n", job_id);
}

int test_profiling_start() {
  (void)setenv("PROFILING_MODE", "true", 1);
  (void)setenv("PROFILING_OPTIONS", "training_trace:task_trace", 1);
  auto ret = ProfilingManager::GetInstance().StartupProfiling(0);
  (void)unsetenv("PROFILING_MODE");
  (void)unsetenv("PROFILING_OPTIONS");
  return ret;
}

TEST_F(TestAscendProfiling, test_profiling_start) {
  auto ret = test_profiling_start();
  ASSERT_EQ(ret, true);
}

int test_profiling_stop() {
  (void)setenv("PROFILING_MODE", "true", 1);
  auto engine = std::make_shared<ProfilingEngineImpl>();
  auto report = std::make_shared<stubReporter>();
  auto plug = engine->CreatePlugin();
  plug->Init(report.get());
  auto ret = ProfilingManager::GetInstance().StopProfiling();
  plug->UnInit();
  engine->ReleasePlugin(plug);
  (void)unsetenv("PROFILING_OPTIONS");
  return ret;
}

TEST_F(TestAscendProfiling, test_profiling_stop) {
  auto ret = test_profiling_stop();
  ASSERT_EQ(ret, true);
}

int test_profiling_rpt() {
  (void)setenv("PROFILING_MODE", "true", 1);
  std::map<uint32_t, std::string> op_taskId_map;
  op_taskId_map[1] = "add";
  op_taskId_map[2] = "mul";
  auto engine = std::make_shared<ProfilingEngineImpl>();
  auto report = std::make_shared<stubReporter>();
  auto plug = engine->CreatePlugin();
  plug->Init(report.get());
  ProfilingManager::GetInstance().ReportProfilingData(op_taskId_map);
  plug->UnInit();
  engine->ReleasePlugin(plug);
  (void)unsetenv("PROFILING_OPTIONS");
  return 0;
}

TEST_F(TestAscendProfiling, test_profiling_rpt) {
  auto ret = test_profiling_rpt();
  ASSERT_EQ(ret, false);
}

int test_profiling_rpt_abnormal() {
  std::map<uint32_t, std::string> op_taskId_map;
  ProfilingManager::GetInstance().ReportProfilingData(op_taskId_map);
  (void)setenv("PROFILING_MODE", "true", 1);
  ProfilingManager::GetInstance().ReportProfilingData(op_taskId_map);
  op_taskId_map[1] = "add";
  op_taskId_map[2] = "mul";
  ProfilingManager::GetInstance().ReportProfilingData(op_taskId_map);
  (void)unsetenv("PROFILING_OPTIONS");
  return 0;
}

TEST_F(TestAscendProfiling, test_profiling_rpt_abnormal) {
  auto ret = test_profiling_rpt_abnormal();
  ASSERT_EQ(ret, false);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
