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
#include "common/backend_common_test.h"
#define private public
#include "plugin/device/ascend/optimizer/ascend_pass_control.h"
#undef private

namespace {
constexpr char kMsAscendFusionSwitch[] = "MS_DEV_ASCEND_FUSION_SWITCH";
}  // namespace
namespace mindspore {
namespace opt {
class PlantPass : public Pass {
 public:
  PlantPass() : Pass("plant") {}
  ~PlantPass() override = default;
  bool GetRunStatus() const { return is_run_; }
  bool Run(const FuncGraphPtr &) override {
    is_run_ = true;
    return true;
  }

  bool is_run_ = false;
};

class PlantPatternPass : public PatternProcessPass {
 public:
  PlantPatternPass() : PatternProcessPass("plant_pattern") { is_run_ = false; }
  ~PlantPatternPass() override = default;
  bool GetRunStatus() const { return is_run_; }

  const BaseRef DefinePattern() const override { return BaseRef({std::make_shared<Var>()}); }
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override {
    is_run_ = true;
    return nullptr;
  }

  inline static bool is_run_ = false;
};

class TestPass : public PassWithSwitch {
 public:
  TestPass() : PassWithSwitch("test") {
    PassSwitchManager::GetInstance().RegistLicPass(name(), OptPassEnum::DereluFusion);
  }
  ~TestPass() override = default;
  bool GetRunStatus() const { return is_run_; }
  bool RunPass(const FuncGraphPtr &) override {
    is_run_ = true;
    return true;
  }

  bool is_run_ = false;
};

class TestPatternPass : public PatternProcessPassWithSwitch {
 public:
  TestPatternPass() : PatternProcessPassWithSwitch("test_pattern") {
    PassSwitchManager::GetInstance().RegistLicPass(name(), OptPassEnum::DereluFusion);
    is_run_ = false;
  }
  ~TestPatternPass() override = default;
  bool GetRunStatus() const { return is_run_; }

  const BaseRef DefinePattern() const override { return BaseRef({std::make_shared<Var>()}); }
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override {
    is_run_ = true;
    return nullptr;
  }

  inline static bool is_run_ = false;
};

class TestAscendPassControl : public UT::Common {
 public:
  TestAscendPassControl() = default;

  void TearDown() override {
    (void)unsetenv(kMsAscendFusionSwitch);
    PassSwitchManager::GetInstance().SetSwitchFromEnv();
  }
};

/// Feature: Pass Switch
/// Description: no MS_DEV_ASCEND_FUSION_SWITCH set and run pass
/// Expectation: switch pass run, plant pass run
TEST_F(TestAscendPassControl, test_no_env_for_pass) {
  (void)unsetenv(kMsAscendFusionSwitch);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  pass.Run(nullptr);
  ASSERT_TRUE(pass.GetRunStatus());
  PlantPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(nullptr);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: set MS_DEV_ASCEND_FUSION_SWITCH as "on" and run pass
/// Expectation: switch pass run, plant pass run
TEST_F(TestAscendPassControl, test_env_on_for_pass_0) {
  (void)setenv(kMsAscendFusionSwitch, "on", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  pass.Run(nullptr);
  ASSERT_TRUE(pass.GetRunStatus());
  PlantPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(nullptr);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: set invalid MS_DEV_ASCEND_FUSION_SWITCH and run pass
/// Expectation: switch pass run, plant pass run
TEST_F(TestAscendPassControl, test_env_on_for_pass_1) {
  (void)setenv(kMsAscendFusionSwitch, "invalidxxxxxxxx", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  pass.Run(nullptr);
  ASSERT_TRUE(pass.GetRunStatus());
  PlantPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(nullptr);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: set MS_DEV_ASCEND_FUSION_SWITCH as "0" and run pass
/// Expectation: switch pass dont run, plant pass run
TEST_F(TestAscendPassControl, test_env_off_for_pass_0) {
  (void)setenv(kMsAscendFusionSwitch, "0", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  pass.Run(nullptr);
  ASSERT_FALSE(pass.GetRunStatus());
  PlantPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(nullptr);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: set MS_DEV_ASCEND_FUSION_SWITCH as "off" and run pass
/// Expectation: switch pass dont run, plant pass run
TEST_F(TestAscendPassControl, test_env_off_for_pass_1) {
  (void)setenv(kMsAscendFusionSwitch, "off", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  pass.Run(nullptr);
  ASSERT_FALSE(pass.GetRunStatus());
  PlantPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(nullptr);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: set MS_DEV_ASCEND_FUSION_SWITCH as "OFF" and run pass
/// Expectation: switch pass dont run, plant pass run
TEST_F(TestAscendPassControl, test_env_off_for_pass_2) {
  (void)setenv(kMsAscendFusionSwitch, "OFF", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  pass.Run(nullptr);
  ASSERT_FALSE(pass.GetRunStatus());
  PlantPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(nullptr);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: no MS_DEV_ASCEND_FUSION_SWITCH set and run pattern pass
/// Expectation: switch pass run, plant pass run
TEST_F(TestAscendPassControl, test_no_env_for_pattern_pass) {
  (void)unsetenv(kMsAscendFusionSwitch);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPatternPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  CNodePtr node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>("test"))});
  pass.Run(graph, node);
  ASSERT_TRUE(pass.GetRunStatus());
  PlantPatternPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(graph, node);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: MS_DEV_ASCEND_FUSION_SWITCH set on and run pattern pass
/// Expectation: switch pass run, plant pass run
TEST_F(TestAscendPassControl, test_env_on_for_pattern_pass) {
  (void)setenv(kMsAscendFusionSwitch, "on", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPatternPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  CNodePtr node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>("test"))});
  pass.Run(graph, node);
  ASSERT_TRUE(pass.GetRunStatus());
  PlantPatternPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(graph, node);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: MS_DEV_ASCEND_FUSION_SWITCH set invalid and run pattern pass
/// Expectation: switch pass run, plant pass run
TEST_F(TestAscendPassControl, test_env_invalid_for_pattern_pass) {
  (void)setenv(kMsAscendFusionSwitch, "invalid_xxasdasdasfsldjmg", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPatternPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  CNodePtr node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>("test"))});
  pass.Run(graph, node);
  ASSERT_TRUE(pass.GetRunStatus());
  PlantPatternPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(graph, node);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: MS_DEV_ASCEND_FUSION_SWITCH set off and run pattern pass
/// Expectation: switch pass dont run, plant pass run
TEST_F(TestAscendPassControl, test_env_off_for_pattern_pass_0) {
  (void)setenv(kMsAscendFusionSwitch, "off", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPatternPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  CNodePtr node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>("test"))});
  pass.Run(graph, node);
  ASSERT_FALSE(pass.GetRunStatus());
  PlantPatternPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(graph, node);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: MS_DEV_ASCEND_FUSION_SWITCH set OFF and run pattern pass
/// Expectation: switch pass dont run, plant pass run
TEST_F(TestAscendPassControl, test_env_off_for_pattern_pass_1) {
  (void)setenv(kMsAscendFusionSwitch, "OFF", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPatternPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  CNodePtr node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>("test"))});
  pass.Run(graph, node);
  ASSERT_FALSE(pass.GetRunStatus());
  PlantPatternPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(graph, node);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}

/// Feature: Pass Switch
/// Description: MS_DEV_ASCEND_FUSION_SWITCH set 0 and run pattern pass
/// Expectation: switch pass dont run, plant pass run
TEST_F(TestAscendPassControl, test_env_off_for_pattern_pass_2) {
  (void)setenv(kMsAscendFusionSwitch, "0", 1);
  PassSwitchManager::GetInstance().SetSwitchFromEnv();
  TestPatternPass pass;
  ASSERT_FALSE(pass.GetRunStatus());
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  CNodePtr node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>("test"))});
  pass.Run(graph, node);
  ASSERT_FALSE(pass.GetRunStatus());
  PlantPatternPass plant_pass;
  ASSERT_FALSE(plant_pass.GetRunStatus());
  plant_pass.Run(graph, node);
  ASSERT_TRUE(plant_pass.GetRunStatus());
}
}  // namespace opt
}  // namespace mindspore
