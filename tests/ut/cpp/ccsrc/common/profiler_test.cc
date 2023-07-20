/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <random>
#include "common/common_test.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace runtime {
class TestProfiler : public UT::Common {
 public:
  TestProfiler() = default;
  virtual ~TestProfiler() = default;

  void SetUp() override {}
  void TearDown() override {}
};

std::vector<ProfilerDataPtr> GenProfilerData(bool shuffle) {
  std::vector<ProfilerDataPtr> profiler_data_vec{
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 0ull, 10ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 5ull, 15ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 10ull, 15ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 15ull, 20ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 25ull, 30ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test2", false, 20ull, 30ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test2", false, 35ull, 60ull),
    std::make_shared<ProfilerData>(ProfilerStage::kDefault, 0ull, 40ull),
    std::make_shared<ProfilerData>(ProfilerStage::kDefault, 20ull, 50ull)};
  if (shuffle) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(profiler_data_vec.begin(), profiler_data_vec.end(), g);
  }
  return profiler_data_vec;
}

/// Feature: test profiler.
/// Description: test profiler structure and interface.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestProfiler, test_profiler) {
  auto &instance = ProfilerAnalyzer::GetInstance();
  EXPECT_EQ(false, instance.profiler_enable());

  // Test profiler enabled.
  instance.set_profiler_enable(true);
  EXPECT_EQ(true, instance.profiler_enable());

  // Test profiler data line.
  const auto &serial_data = GenProfilerData(false);
  instance.StartStep();
  for (const auto &data : serial_data) {
    instance.RecordData(data);
  }
  auto &data = instance.data();
  EXPECT_EQ(serial_data.size(), data.size());
  instance.set_step_time(100ull);
  instance.EndStep();
  EXPECT_EQ(0, data.size());
  const auto &data_line = instance.data_line();
  EXPECT_EQ(1, data_line.size());
  instance.Clear();

  EXPECT_EQ(0, data.size());
  auto &module_infos = instance.module_infos();
  EXPECT_EQ(0, module_infos.size());
  auto &stage_infos = instance.stage_infos();
  EXPECT_EQ(0, stage_infos.size());

  // Prepare data.
  const auto &shuffled_data = GenProfilerData(true);
  const int shuffled_data_size = shuffled_data.size();

  ProfilerDataSpan shuffled_data_list(shuffled_data.begin(), shuffled_data.end());
  // Test ProcessModuleSummaryData.
  instance.ProcessModuleSummaryData(shuffled_data_list);
  // Assert result.
  EXPECT_EQ(1, module_infos.size());
  EXPECT_EQ(0, stage_infos.size());
  auto &module_info_ptr = module_infos.at(ProfilerModule::kDefault);
  auto &module_info_statistics_info_ptr = module_info_ptr->module_statistics_info_;
  EXPECT_EQ(shuffled_data_size - 2, module_info_statistics_info_ptr->count_);
  EXPECT_EQ(60ull, module_info_statistics_info_ptr->total_time_);

  // Test profiler detail.
  instance.StartStep();
  for (const auto &data : shuffled_data_list) {
    instance.RecordData(data);
  }
  EXPECT_EQ(shuffled_data_size, data.size());
  EXPECT_EQ(1, module_infos.size());

  // Do statistics.
  instance.set_step_time(100ull);
  instance.EndStep();
  instance.Clear();

  EXPECT_EQ(0, module_infos.size());
  EXPECT_EQ(0, stage_infos.size());
  EXPECT_EQ(2, instance.step());

  // Assert json data.
  const auto &json_infos = instance.json_infos();
  EXPECT_EQ(18, json_infos.size());
  int32_t pid = getpid();
  for (const auto &obj : json_infos) {
    for (const auto &[k, v] : obj.items()) {
      if (k == "ph") {
        EXPECT_EQ("X", v);
      } else if (k == "pid") {
        EXPECT_EQ(pid, std::stoi(v.get<std::string>()));
      }
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
