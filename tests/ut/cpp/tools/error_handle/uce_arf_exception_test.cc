/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include "common/common_test.h"
#include "common/mockcpp.h"
#include "include/utils/callback.h"
#include "include/utils/python_adapter.h"
#include "plugin/ascend/res_manager/error_manager/collective_comm_monitor.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "tools/error_handler/error_config.h"
#include "tools/error_handler/error_handler.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
class TestErrorHandler : public UT::Common {
 public:
  TestErrorHandler() = default;
  virtual ~TestErrorHandler() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test UCE Exception instance.
/// Description:Test all interfaces in the instance.
/// Expectation: The return value of the normal interface is as expected, and the exception branch is expected to catch
/// the exception.
TEST_F(TestErrorHandler, test_interface) {
  auto &error_handler = tools::ErrorHandler::GetInstance();

  EXPECT_EQ(tools::TftConfig::GetInstance()->IsEnableUCE(), false);
  EXPECT_EQ(error_handler.HasThrownError(), false);
  EXPECT_EQ(error_handler.GetForceStopFlag(), false);
  EXPECT_EQ(error_handler.GetUceFlag(), false);
  EXPECT_EQ(error_handler.IsRebootNode(), false);
  EXPECT_EQ(error_handler.IsArf(), false);

  // test arf/uce/ttp basic interface
  EXPECT_NO_THROW(error_handler.SetIsArf(true));
  EXPECT_NO_THROW(error_handler.SetRebootNode(true));
  EXPECT_EQ(error_handler.IsRebootNode(), true);
  EXPECT_EQ(error_handler.IsArf(), true);
  EXPECT_EQ(error_handler.HasThrownError(), true);

  EXPECT_NO_THROW(error_handler.SetForceStopFlag(true));
  EXPECT_EQ(error_handler.GetForceStopFlag(), true);

  EXPECT_NO_THROW(error_handler.SetRebootType("arf"));
  EXPECT_EQ(error_handler.GetRebootType(), "arf");

  EXPECT_EQ(error_handler.GetSuspectRemoteFlag(), false);
  EXPECT_EQ(error_handler.HasThrownError(), true);
}
}  // namespace mindspore

using mindspore::device::ascend::aclrtCreateEventExWithFlag_;
using mindspore::device::ascend::aclrtCreateEventExWithFlagFunObj;
using mindspore::device::ascend::aclrtDestroyEvent_;
using mindspore::device::ascend::aclrtDestroyEventFunObj;
using mindspore::device::ascend::HcclWatchDogHandler;
using mindspore::device::ascend::HcclWatchDogManager;

struct WatchdogInitCase {
  const char *name;
  const char *watchdog;
  bool status_save;
  size_t group_count;
  bool expect_handler;
};

class WatchdogTestRuntime {
 public:
  WatchdogTestRuntime() : create_event_(aclrtCreateEventExWithFlag_), destroy_event_(aclrtDestroyEvent_) {
    current_runtime_ = this;
    aclrtCreateEventExWithFlag_ = [this](aclrtEvent *event, uint32_t) {
      *event = reinterpret_cast<aclrtEvent>(&event_token_);
      ++created_events_;
      return ACL_SUCCESS;
    };
    aclrtDestroyEvent_ = [this](aclrtEvent) {
      ++destroyed_events_;
      return ACL_SUCCESS;
    };
    MOCKER_CPP(&HcclWatchDogHandler::Initialize, bool (*)(HcclWatchDogHandler *))
      .stubs()
      .will(invoke(InitializeHandler));
  }

  ~WatchdogTestRuntime() {
    HcclWatchDogManager::GetInstance().DestroyHandler();
    GlobalMockObject::reset();
    aclrtCreateEventExWithFlag_ = std::move(create_event_);
    aclrtDestroyEvent_ = std::move(destroy_event_);
    current_runtime_ = nullptr;
  }

  const std::vector<std::string> &initialized_groups() const { return initialized_groups_; }
  size_t created_events() const { return created_events_; }
  size_t destroyed_events() const { return destroyed_events_; }

 private:
  static bool InitializeHandler(HcclWatchDogHandler *handler) {
    EXPECT_EQ(handler->rank_id(), 0U);
    current_runtime_->initialized_groups_.push_back(handler->group_name());
    return true;
  }

  inline static WatchdogTestRuntime *current_runtime_{nullptr};
  aclrtCreateEventExWithFlagFunObj create_event_;
  aclrtDestroyEventFunObj destroy_event_;
  std::vector<std::string> initialized_groups_;
  size_t created_events_{0};
  size_t destroyed_events_{0};
  uint32_t event_token_{0};
};

static void CheckWatchdogRegistration(const WatchdogInitCase &test_case) {
  ASSERT_EQ(mindspore::common::SetEnv("MS_SIMULATION_LEVEL", ""), 0);
  auto python_scope = mindspore::python_adapter::set_python_scoped();
  py::gil_scoped_acquire gil;
  py::dict configs;
  if (!std::string(test_case.watchdog).empty()) {
    configs[py::str(mindspore::kWatchdog)] = py::str(test_case.watchdog);
  }
  if (test_case.status_save) {
    configs[py::str(mindspore::kStatusRecord)] = py::str("1");
  }
  mindspore::tools::TftConfig::GetInstance()->RegisterConfig(configs);
  auto context = mindspore::MsContext::GetInstance();
  ASSERT_NE(context, nullptr);
  ASSERT_TRUE(context->get_param<bool>(mindspore::MS_CTX_ENABLE_HCCL_WATCHDOG));

  WatchdogTestRuntime runtime;
  auto register_watchdog = GET_COMMON_CALLBACK(RegisterHcclWatchDog, void, uint32_t, uint32_t, const std::string &,
                                               HcclComm, const std::vector<uint32_t> &);
  ASSERT_NE(register_watchdog, nullptr);
  auto &manager = HcclWatchDogManager::GetInstance();
  const std::vector<uint32_t> group_ranks{0, 1};
  std::vector<uint32_t> comm_tokens(test_case.group_count, 0);
  std::vector<std::string> expected_groups;
  for (size_t group_index = 0; group_index < test_case.group_count; ++group_index) {
    const auto group_name = "watchdog_init_group_" + std::to_string(group_index);
    register_watchdog(0, 0, group_name, reinterpret_cast<HcclComm>(&comm_tokens[group_index]), group_ranks);
    if (test_case.expect_handler) {
      expected_groups.push_back(group_name);
    }
    EXPECT_EQ(runtime.initialized_groups(), expected_groups);
    EXPECT_EQ(HcclWatchDogManager::CheckStatusSaveEnable(), test_case.status_save);
  }
  EXPECT_EQ(runtime.created_events(), 2U * expected_groups.size());
  manager.DestroyHandler();
  EXPECT_EQ(runtime.destroyed_events(), runtime.created_events());
  for (size_t group_index = 0; group_index < test_case.group_count; ++group_index) {
    EXPECT_FALSE(manager.InitHandler("watchdog_init_group_" + std::to_string(group_index)));
  }
  GlobalMockObject::verify();
}

class TestHcclWatchdogInitDeathTest : public testing::TestWithParam<WatchdogInitCase> {};

/*
 * Feature: HCCL watchdog initialization before the first kernel launch.
 * Description: Register groups with default, enabled, disabled and status-save-only configs in isolated processes.
 * Expectation: Enabled groups get handlers, disabled groups do not, and cleanup releases every handler and event.
 */
TEST_P(TestHcclWatchdogInitDeathTest, RegistrationBeforeFirstKernel) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
    {
      CheckWatchdogRegistration(GetParam());
      std::_Exit(::testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
    },
    ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

static std::string WatchdogInitCaseName(const testing::TestParamInfo<WatchdogInitCase> &info) {
  return info.param.name;
}

INSTANTIATE_TEST_CASE_P(Configurations, TestHcclWatchdogInitDeathTest,
                        testing::Values(WatchdogInitCase{"DefaultEnabled", "", false, 1, true},
                                        WatchdogInitCase{"ExplicitEnabled", "1", false, 1, true},
                                        WatchdogInitCase{"ExplicitDisabled", "0", false, 1, false},
                                        WatchdogInitCase{"StatusSaveOnly", "0", true, 1, true},
                                        WatchdogInitCase{"MultipleGroups", "1", false, 2, true}),
                        WatchdogInitCaseName);
