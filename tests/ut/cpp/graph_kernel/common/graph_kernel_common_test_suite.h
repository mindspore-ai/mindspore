/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef TESTS_UT_CPP_GRAPH_KERNEL_COMMON_GRAPH_KERNEL_TEST_SUITE_H_
#define TESTS_UT_CPP_GRAPH_KERNEL_COMMON_GRAPH_KERNEL_TEST_SUITE_H_

#include "common/common_test.h"
#include "common/graph_optimizer_test_framework.h"

namespace mindspore::graphkernel::test {
using mindspore::test::ConstructGraph;
using mindspore::test::RunPass;

class GraphKernelCommonTestSuite : public UT::Common {
 public:
  GraphKernelCommonTestSuite(){};
  virtual ~GraphKernelCommonTestSuite() = default;
};
}  // namespace mindspore::graphkernel::test
#endif  // TESTS_UT_CPP_GRAPH_KERNEL_COMMON_GRAPH_KERNEL_TEST_SUITE_H_
