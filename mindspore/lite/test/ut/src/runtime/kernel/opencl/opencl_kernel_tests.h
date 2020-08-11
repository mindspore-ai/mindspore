/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"

#ifndef TESTS_UT_OPENCL_KERNLE_TESTS_H
#define TESTS_UT_OPENCL_KERNLE_TESTS_H

namespace mindspore {

class TestOpenCLKernel : public mindspore::CommonTest {
 public:
  TestOpenCLKernel() {}
};

}  // namespace mindspore
#endif  // TESTS_UT_OPENCL_KERNLE_TESTS_H
