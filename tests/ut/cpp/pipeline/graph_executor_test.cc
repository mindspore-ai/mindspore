/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "utils/log_adapter.h"
#include "frontend/jit/ps/action.h"
#include "include/frontend/jit/ps/executor/graph_executor_py.h"
#include "frontend/jit/ps/pipeline.h"

namespace mindspore {
namespace pipeline {
class TestGraphExecutor : public UT::Common {
 public:
  TestGraphExecutor() {}
};

/// Feature: Test jit_config
/// Description: Test set jit_level = o0
/// Expectation: success
TEST_F(TestGraphExecutor, DISABLED_test_jit_config_with_jit_level_equal_o0) {
  py::dict obj = python_adapter::CallPyFn("gtest_input.pipeline.graph_executor_test", "get_jit_config_o0");
  pipeline::GraphExecutorPy::GetInstance()->SetJitConfig(obj);

  auto jit_level = pipeline::GetJitLevel();
  ASSERT_TRUE(jit_level == "O0");

  auto actions = VmPipeline(std::make_shared<pipeline::Resource>());
  bool ret = false;
  for (auto action : actions) {
    if (action.first == "combine_like_graphs") {
      ret = true;
    }
  }
  ASSERT_TRUE(ret == false);
}

/// Feature: Test jit_config
/// Description: Test set jit_level = o1
/// Expectation: success
TEST_F(TestGraphExecutor, test_jit_config_with_jit_level_equal_o1) {
  py::dict obj = python_adapter::CallPyFn("gtest_input.pipeline.graph_executor_test", "get_jit_config_o1");
  pipeline::GraphExecutorPy::GetInstance()->SetJitConfig(obj);

  auto jit_level = pipeline::GetJitLevel();
  ASSERT_TRUE(jit_level == "O1");
}

/// Feature: Test jit_config
/// Description: Test jit_level with unused config
/// Expectation: success
TEST_F(TestGraphExecutor, test_jit_config_with_unused_config) {
  py::dict obj = python_adapter::CallPyFn("gtest_input.pipeline.graph_executor_test", "get_unused_config");
  pipeline::GraphExecutorPy::GetInstance()->SetJitConfig(obj);
  auto jit_level = pipeline::GetJitLevel();
  ASSERT_TRUE(jit_level == "");
}
}  // namespace pipeline
}  // namespace mindspore
