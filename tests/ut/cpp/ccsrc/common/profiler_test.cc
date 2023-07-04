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

#include "common/common_test.h"

#include "include/common/profiler.h"

#include <random>

namespace mindspore {
namespace runtime {
class TestProfiler : public UT::Common {
 public:
  TestProfiler() = default;
  virtual ~TestProfiler() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test profiler.
/// Description: test profiler structure and interface.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestProfiler, test_profiler) {
  auto &instance = ProfilerAnalyzer::GetInstance();
  EXPECT_EQ(false, instance.profiler_enable());

  instance.set_profiler_enable(true);
  instance.StartStep();
  auto &data = instance.data();
  EXPECT_EQ(0, data.size());

  auto &module_infos = instance.module_infos();
  EXPECT_EQ(0, module_infos.size());

  auto &stage_infos = instance.stage_infos();
  EXPECT_EQ(0, stage_infos.size());

  // Prepare data.
  std::vector<ProfilerDataPtr> stageDefaultRange{
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 0ull, 10ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 5ull, 15ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 10ull, 15ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 15ull, 20ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test1", false, 25ull, 30ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test2", false, 20ull, 30ull),
    std::make_shared<ProfilerData>(ProfilerModule::kDefault, ProfilerEvent::kDefault, "op_test2", false, 35ull, 60ull),
    std::make_shared<ProfilerData>(ProfilerStage::kDefault, 0ull, 40ull),
    std::make_shared<ProfilerData>(ProfilerStage::kDefault, 20ull, 50ull)};
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(stageDefaultRange.begin(), stageDefaultRange.end(), g);

  const int stage_default_range_size = stageDefaultRange.size();
  for (auto &range : stageDefaultRange) {
    instance.RecordData(range);
  }
  EXPECT_EQ(stage_default_range_size, data.size());
  EXPECT_EQ(0, module_infos.size());
  EXPECT_EQ(0, stage_infos.size());

  // Do statistics.
  instance.set_step_time(100ull);
  instance.EndStep();

  // Assert result.
  EXPECT_EQ(2, module_infos.size());
  auto &module_info_ptr = module_infos.at(ProfilerModule::kDefault);
  auto &module_info_statistics_info_ptr = module_info_ptr->module_statistics_info_;
  EXPECT_EQ(stage_default_range_size - 2, module_info_statistics_info_ptr->count_);
  EXPECT_EQ(60ull, module_info_statistics_info_ptr->total_time_);

  auto &module_info_event_infos = module_info_ptr->event_infos_;
  EXPECT_EQ(1, module_info_event_infos.size());
  auto &profiler_event_info_ptr = module_info_event_infos[ProfilerEvent::kDefault];
  auto event_statistics_info_ptr = profiler_event_info_ptr->event_statistics_info_;
  EXPECT_EQ(stage_default_range_size - 2, event_statistics_info_ptr->count_);
  EXPECT_EQ(70ull, event_statistics_info_ptr->total_time_);

  EXPECT_EQ(2, stage_infos.size());
}
}  // namespace runtime
}  // namespace mindspore
