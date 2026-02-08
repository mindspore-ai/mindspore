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

#include "common/common_test.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#define protected public
#define private public
#include "include/backend/common/custom_pass/custom_pass_plugin_manager.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
namespace {

class TestPass : public Pass {
 public:
  explicit TestPass(const std::string &name) : Pass(name) {}
  bool Run(const FuncGraphPtr &) override { return true; }
};

class TestCustomPassPlugin : public CustomPassPlugin {
 public:
  TestCustomPassPlugin(std::string plugin_name, std::vector<std::string> passes,
                       std::unordered_set<std::string> failing_creations = {})
      : plugin_name_(std::move(plugin_name)),
        available_passes_(std::move(passes)),
        failing_creations_(std::move(failing_creations)) {}

  std::string GetPluginName() const override { return plugin_name_; }

  std::vector<std::string> GetAvailablePassNames() const override { return available_passes_; }

  std::shared_ptr<Pass> CreatePass(const std::string &pass_name) const override {
    if (std::find(available_passes_.begin(), available_passes_.end(), pass_name) == available_passes_.end()) {
      return nullptr;
    }
    if (failing_creations_.find(pass_name) != failing_creations_.end()) {
      return nullptr;
    }
    return std::make_shared<TestPass>(pass_name);
  }

 private:
  std::string plugin_name_;
  std::vector<std::string> available_passes_;
  std::unordered_set<std::string> failing_creations_;
};

class CustomPassPluginManagerUt : public UT::Common {
 protected:
  void SetUp() override {
    UT::Common::SetUp();
    manager_ = &CustomPassPluginManager::GetInstance();
    ResetManagerState();
  }

  void TearDown() override {
    ResetManagerState();
    UT::Common::TearDown();
  }

  void ResetManagerState() {
    if (manager_ == nullptr) {
      return;
    }
    manager_->UnloadAllPlugins();
    manager_->pass_execution_order_.clear();
    manager_->plugins_.clear();
    manager_->next_registration_order_ = 0;
  }

  void InsertPlugin(const std::shared_ptr<TestCustomPassPlugin> &plugin) {
    auto info = std::make_unique<CustomPassPluginManager::PluginInfo>(
      plugin->GetPluginName(), "/tmp/" + plugin->GetPluginName(), plugin, nullptr, nullptr);
    manager_->plugins_[plugin->GetPluginName()] = std::move(info);
  }

  CustomPassPluginManager *manager_ = nullptr;
};

/// Feature: Plugin loading functionality
/// Description: Test loading a plugin with invalid file path
/// Expectation: LoadPlugin returns false and internal state remains clean
TEST_F(CustomPassPluginManagerUt, LoadPluginInvalidPath) {
  EXPECT_FALSE(manager_->LoadPlugin("/invalid/path/libfake.so", "FakePass"));
  EXPECT_TRUE(manager_->plugins_.empty());
  EXPECT_TRUE(manager_->pass_execution_order_.empty());
}

/// Feature: Plugin reference counting and lifecycle management
/// Description: Test unloading plugin with outstanding shared_ptr references
/// Expectation: Plugin remains loaded until all references are released
TEST_F(CustomPassPluginManagerUt, UnloadPluginRespectsOutstandingReferences) {
  auto plugin = std::make_shared<TestCustomPassPlugin>("TestPlugin", std::vector<std::string>{"PassA"});
  InsertPlugin(plugin);
  plugin.reset();
  manager_->RegisterPassExecution("PassA", "TestPlugin", "cpu", "stage1");

  auto external_ref = manager_->GetPlugin("TestPlugin");
  ASSERT_NE(external_ref, nullptr);
  EXPECT_TRUE(manager_->IsPluginLoaded("TestPlugin"));
  EXPECT_EQ(manager_->pass_execution_order_.size(), 1UL);

  manager_->UnloadPlugin("TestPlugin");
  EXPECT_TRUE(manager_->IsPluginLoaded("TestPlugin"));
  EXPECT_EQ(manager_->pass_execution_order_.size(), 1UL);

  external_ref.reset();
  manager_->UnloadPlugin("TestPlugin");
  EXPECT_FALSE(manager_->IsPluginLoaded("TestPlugin"));
  EXPECT_TRUE(manager_->pass_execution_order_.empty());
}

/// Feature: Device-specific pass registration
/// Description: Test registering passes to optimizer with device filtering
/// Expectation: Only passes matching the target device are registered
TEST_F(CustomPassPluginManagerUt, RegisterPassesToOptimizerFiltersByDevice) {
  auto plugin = std::make_shared<TestCustomPassPlugin>("DevicePlugin", std::vector<std::string>{"CpuPass", "GpuPass"});
  InsertPlugin(plugin);
  manager_->RegisterPassExecution("CpuPass", "DevicePlugin", "cpu", "pre");
  manager_->RegisterPassExecution("GpuPass", "DevicePlugin", "gpu", "pre");

  auto optimizer = std::make_shared<GraphOptimizer>("test_optimizer");
  manager_->RegisterPassesToOptimizer(optimizer, "cpu");

  ASSERT_EQ(optimizer->pass_managers_.size(), 1UL);
  const auto &ordered_manager = optimizer->pass_managers_.front();
  ASSERT_NE(ordered_manager, nullptr);
  ASSERT_EQ(ordered_manager->Passes().size(), 1UL);
  EXPECT_EQ(ordered_manager->Passes().front()->name(), "CpuPass");
}

/// Feature: Bulk plugin unloading with reference protection
/// Description: Test UnloadAllPlugins behavior with plugins still in use
/// Expectation: In-use plugins remain loaded until references are released
TEST_F(CustomPassPluginManagerUt, UnloadAllPluginsHonorsInUsePlugins) {
  auto plugin = std::make_shared<TestCustomPassPlugin>("SharedPlugin", std::vector<std::string>{"PassA"});
  InsertPlugin(plugin);
  plugin.reset();
  manager_->RegisterPassExecution("PassA", "SharedPlugin", "cpu", "stage1");

  auto external_ref = manager_->GetPlugin("SharedPlugin");
  ASSERT_NE(external_ref, nullptr);

  manager_->UnloadAllPlugins();
  EXPECT_TRUE(manager_->IsPluginLoaded("SharedPlugin"));
  EXPECT_EQ(manager_->pass_execution_order_.size(), 1UL);

  external_ref.reset();
  manager_->UnloadAllPlugins();
  EXPECT_FALSE(manager_->IsPluginLoaded("SharedPlugin"));
  EXPECT_TRUE(manager_->pass_execution_order_.empty());
}

/// Feature: Pass execution order and metadata tracking
/// Description: Test registration order and metadata preservation during pass execution setup
/// Expectation: Pass execution order and metadata are correctly tracked and preserved
TEST_F(CustomPassPluginManagerUt, RegisterPassExecutionTracksOrderAndMetadata) {
  manager_->RegisterPassExecution("PassA", "PluginA", "cpu", "stage1");
  manager_->RegisterPassExecution("PassB", "PluginB", "gpu", "stage2");

  ASSERT_EQ(manager_->pass_execution_order_.size(), 2UL);
  EXPECT_EQ(manager_->pass_execution_order_[0].pass_name, "PassA");
  EXPECT_EQ(manager_->pass_execution_order_[0].plugin_name, "PluginA");
  EXPECT_EQ(manager_->pass_execution_order_[0].device, "cpu");
  EXPECT_EQ(manager_->pass_execution_order_[0].registration_order, 0UL);

  EXPECT_EQ(manager_->pass_execution_order_[1].pass_name, "PassB");
  EXPECT_EQ(manager_->pass_execution_order_[1].plugin_name, "PluginB");
  EXPECT_EQ(manager_->pass_execution_order_[1].device, "gpu");
  EXPECT_EQ(manager_->pass_execution_order_[1].registration_order, 1UL);
}

/// Feature: Plugin state query functions
/// Description: Test GetAllPlugins and GetLoadedPluginNames reflect current manager state
/// Expectation: Query functions return accurate plugin information matching internal state
TEST_F(CustomPassPluginManagerUt, GetAllPluginsAndNamesReflectState) {
  auto plugin_a = std::make_shared<TestCustomPassPlugin>("PluginA", std::vector<std::string>{"PassA"});
  auto plugin_b = std::make_shared<TestCustomPassPlugin>("PluginB", std::vector<std::string>{"PassB"});
  InsertPlugin(plugin_a);
  InsertPlugin(plugin_b);

  auto plugins = manager_->GetAllPlugins();
  ASSERT_EQ(plugins.size(), 2UL);
  std::unordered_set<std::string> names;
  for (const auto &p : plugins) {
    names.insert(p->GetPluginName());
  }
  EXPECT_TRUE(names.count("PluginA") == 1);
  EXPECT_TRUE(names.count("PluginB") == 1);

  auto loaded_names = manager_->GetLoadedPluginNames();
  std::unordered_set<std::string> loaded(loaded_names.begin(), loaded_names.end());
  EXPECT_TRUE(loaded.count("PluginA") == 1);
  EXPECT_TRUE(loaded.count("PluginB") == 1);
}

/// Feature: Pass registration with missing plugin handling
/// Description: Test pass registration behavior when referenced plugin is not loaded
/// Expectation: Optimizer registration gracefully skips passes from missing plugins
TEST_F(CustomPassPluginManagerUt, RegisterPassesSkipsMissingPlugin) {
  manager_->RegisterPassExecution("OrphanPass", "MissingPlugin", "cpu", "stage1");

  auto optimizer = std::make_shared<GraphOptimizer>("test_optimizer");
  manager_->RegisterPassesToOptimizer(optimizer, "cpu");

  if (!optimizer->pass_managers_.empty()) {
    EXPECT_TRUE(optimizer->pass_managers_.front()->Passes().empty());
  }
}

/// Feature: Pass registration with creation failure handling
/// Description: Test pass registration behavior when pass creation fails
/// Expectation: Failed pass creation is skipped while successful passes are registered
TEST_F(CustomPassPluginManagerUt, RegisterPassesSkipsFailedPassCreation) {
  auto plugin = std::make_shared<TestCustomPassPlugin>("FailPlugin", std::vector<std::string>{"Good", "Bad"},
                                                       std::unordered_set<std::string>{"Bad"});
  InsertPlugin(plugin);
  manager_->RegisterPassExecution("Good", "FailPlugin", "cpu", "stage1");
  manager_->RegisterPassExecution("Bad", "FailPlugin", "cpu", "stage1");

  auto optimizer = std::make_shared<GraphOptimizer>("test_optimizer");
  manager_->RegisterPassesToOptimizer(optimizer, "cpu");

  ASSERT_EQ(optimizer->pass_managers_.size(), 1UL);
  const auto &ordered_manager = optimizer->pass_managers_.front();
  ASSERT_NE(ordered_manager, nullptr);
  ASSERT_EQ(ordered_manager->Passes().size(), 1UL);
  EXPECT_EQ(ordered_manager->Passes().front()->name(), "Good");
}

/// Feature: Plugin unloading edge case handling
/// Description: Test unloading a plugin that was never loaded
/// Expectation: UnloadPlugin is a no-op when plugin does not exist
TEST_F(CustomPassPluginManagerUt, UnloadPluginNoOpWhenNotLoaded) {
  manager_->UnloadPlugin("NonExistingPlugin");
  EXPECT_TRUE(manager_->plugins_.empty());
  EXPECT_TRUE(manager_->pass_execution_order_.empty());
}

}  // namespace
}  // namespace opt
}  // namespace mindspore
