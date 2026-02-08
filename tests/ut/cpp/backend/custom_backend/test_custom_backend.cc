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

#include "common/graph_optimizer_test_framework.h"
#define protected public
#define private public
#include "include/backend/backend_manager/backend_manager.h"
#undef private
#undef protected

namespace mindspore {
namespace backend {
namespace {
// mock BackendBase
class TestBackend : public BackendBase {
 public:
  BackendGraphId Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) { return 0; }

  RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
    return RunningStatus::kRunningSuccess;
  }
};

class CustomBackendManagerUt : public UT::Common {
 protected:
  void SetUp() override {
    UT::Common::SetUp();
    manager_ = &BackendManager::GetInstance();
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
    manager_->Clear();
    manager_->backend_load_handle_.clear();
    manager_->backend_creators_.clear();
  }

  void AddBackend(const std::string &backend_name) {
    BackendCreator func = []() { return std::make_shared<TestBackend>(); };
    manager_->Register(backend_name, std::move(func));
  }

  BackendManager *manager_ = nullptr;
};

/// Feature: Plugin loading functionality
/// Description: Test loading a plugin with invalid file path
/// Expectation: LoadPlugin returns false and internal state remains clean
TEST_F(CustomBackendManagerUt, LoadPluginInvalidPath) {
  EXPECT_FALSE(manager_->LoadBackend("FakeBackend", "/invalid/path/libfake.so"));
  EXPECT_TRUE(manager_->backend_load_handle_.empty());
}

/// Feature: test register backends
/// Description: Test AddBackend twice with same name
/// Expectation: AddBackend throws an exception
TEST_F(CustomBackendManagerUt, AddCustomBackend) {
  AddBackend("test_bakcend");
  EXPECT_THROW(AddBackend("test_bakcend"), std::runtime_error);

  AddBackend("ms_backend");
  EXPECT_THROW(manager_->LoadBackend("ms_backend"), std::runtime_error);

  EXPECT_EQ(manager_->backend_creators_.size(), 2UL);
  EXPECT_EQ(manager_->backend_load_handle_.size(), 0UL);
}

/// Feature: test GetOrCreateBackend function
/// Description: Test GetOrCreateBackend under different scenarios
/// Expectation: GetOrCreateBackend throws an exception
TEST_F(CustomBackendManagerUt, GetOrCreateBackend) {
  AddBackend("test_bakcend");
  AddBackend("ms_backend");

  EXPECT_NE(manager_->GetOrCreateBackend(static_cast<BackendType>(0)), nullptr);
  EXPECT_EQ(manager_->backend_creators_.size(), 2UL);
  EXPECT_EQ(manager_->backend_load_handle_.size(), 0UL);

  EXPECT_NE(manager_->GetOrCreateBackend(static_cast<BackendType>(3)), nullptr);
  EXPECT_EQ(manager_->backend_creators_.size(), 2UL);
  EXPECT_EQ(manager_->backend_load_handle_.size(), 0UL);

  EXPECT_THROW(manager_->GetOrCreateBackend(static_cast<BackendType>(1)), std::runtime_error);
  EXPECT_EQ(manager_->backend_creators_.size(), 2UL);
  EXPECT_EQ(manager_->backend_load_handle_.size(), 0UL);

  EXPECT_NE(manager_->GetOrCreateBackend(static_cast<BackendType>(0)), nullptr);
  EXPECT_EQ(manager_->backend_creators_.size(), 2UL);
  EXPECT_EQ(manager_->backend_load_handle_.size(), 0UL);

  EXPECT_NE(manager_->GetOrCreateBackend(static_cast<BackendType>(3)), nullptr);
  EXPECT_EQ(manager_->backend_creators_.size(), 2UL);
  EXPECT_EQ(manager_->backend_load_handle_.size(), 0UL);
}

/// Feature: Test register a backend and run it
/// Description: Register a custom backend and run it
/// Expectation: run successful
TEST_F(CustomBackendManagerUt, RegisterAndRunCustomBackend) {
  AddBackend("ms_backend");

  EXPECT_NE(manager_->GetOrCreateBackend(static_cast<BackendType>(0)), nullptr);
  auto kernel_graph = std::make_shared<session::KernelGraph>();
  BackendJitConfig backend_jit_config;
  auto build_ret = manager_->Build(kernel_graph, backend_jit_config, "ms_backend");
  EXPECT_EQ(build_ret.first, static_cast<BackendType>(0));
  EXPECT_EQ(static_cast<int>(build_ret.second), 0);
}

/// Feature: Plugin unloading edge case handling
/// Description: Test unloading a plugin that was never loaded
/// Expectation: UnloadPlugin is a no-op when plugin does not exist
TEST_F(CustomBackendManagerUt, UnloadPlugin) {
  AddBackend("test_bakcend");
  manager_->UnloadBackend();
  manager_->Clear();
  EXPECT_EQ(manager_->backend_creators_.size(), 0UL);
  EXPECT_EQ(manager_->backend_load_handle_.size(), 0UL);
}

}  // namespace
}  // namespace backend
}  // namespace mindspore
