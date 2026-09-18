/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "ir/primitive.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
// Stands in for the op plugin kernel mod whose initialization failed: it reports the failure and is never usable.
class FailingInitKernelMod : public kernel::KernelMod {
 public:
  std::vector<KernelAttr> GetOpSupport() override { return {}; }
  bool Init(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) override { return false; }
};

// Serves the CustomOpPlugin kernel mod and reports that the op has no builtin kernel, which is the situation the
// op plugin fallback exists for.
class PluginAndNoBuiltinKernelExecutor : public device::KernelExecutor {
 public:
  kernel::KernelModPtr CreateKernelMod(const std::string &op_name) const override {
    if (op_name == "CustomOpPlugin") {
      plugin_mod_requested_ = true;
      return std::make_shared<FailingInitKernelMod>();
    }
    return nullptr;
  }

  bool plugin_mod_requested() const { return plugin_mod_requested_; }
  void reset() const { plugin_mod_requested_ = false; }

 private:
  mutable bool plugin_mod_requested_{false};
};

class TestDeviceContext : public device::DeviceContext {
 public:
  TestDeviceContext() : device::DeviceContext(device::DeviceContextKey{device::DeviceType::kCPU, 0}) {}
  ~TestDeviceContext() override = default;

  void Initialize() override {}
  void Destroy() override {}
};
}  // namespace

class PyBoostUtilsCreateKernelModTest : public testing::Test {
 protected:
  void SetUp() override {
    // The op plugin branch is only taken when MS_OP_PLUGIN_PATH is visible, and it is read once per process.
    common::SetEnv("MS_OP_PLUGIN_PATH", "/tmp/ms_ut_op_plugin");
  }
};

/// Feature: PyBoost kernel mod caching.
/// Description: The op plugin kernel mod fails to initialize and the op has no builtin kernel, so creating the
///   kernel mod raises; the same op is then dispatched again.
/// Expectation: Every dispatch reports the failure instead of reusing the kernel mod that never initialized.
TEST_F(PyBoostUtilsCreateKernelModTest, FailedPluginInitIsNotCached) {
  auto executor = std::make_shared<PluginAndNoBuiltinKernelExecutor>();
  TestDeviceContext device_context;
  device_context.SetKernelExecutor(executor);
  auto prim = std::make_shared<Primitive>("PyBoostUtilsUtOpWithoutBuiltinKernel");
  std::vector<KernelTensor *> inputs;
  std::vector<KernelTensor *> outputs;

  ASSERT_THROW(PyBoostUtils::CreateKernelMod(prim, &device_context, inputs, outputs, false), std::exception);

  executor->reset();
  ASSERT_THROW(PyBoostUtils::CreateKernelMod(prim, &device_context, inputs, outputs, false), std::exception);
  EXPECT_TRUE(executor->plugin_mod_requested())
    << "the op plugin branch was not taken again, the kernel mod cache served the kernel mod whose init failed";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
