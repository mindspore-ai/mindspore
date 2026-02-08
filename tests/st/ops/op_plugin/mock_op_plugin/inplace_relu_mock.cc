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
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "custom_kernel_input_info.h"

extern "C" {

// Mock implementation of the inplace_relu operator.
// Test inplace op.
int InplaceReLU(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                void *extra) {
  std::cout << "op_plugin mock: InplaceReLU called" << std::endl;
  constexpr int expected_nparam = 2;
  if (nparam != expected_nparam || params == nullptr || ndims == nullptr || shapes == nullptr) {
    std::cout << "Invalid parameters for inplace_relu operator" << std::endl;
    return -1;
  }

  float *x = static_cast<float *>(params[0]);
  if (ndims[0] > 2 || (ndims[0] == 2 && shapes[0][1] != 1)) {
    std::cout << "Only support 1d or 2d (1 column) input for mock inplace_relu operator" << std::endl;
    return -1;
  }
  int x_dim = shapes[0][0];
  for (int i = 0; i < x_dim; ++i) {
    x[i] = std::max(0.0f, x[i]);
  }

  return 0;
}

}  // extern "C"
