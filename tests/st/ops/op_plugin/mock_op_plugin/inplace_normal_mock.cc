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
#include <iostream>

#include "custom_kernel_input_info.h"

extern "C" {

// Mock implementation of the inplace_normal operator, it is called by mint.randn().
// Test the case when the input is tuple scalar
int InplaceNormal(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                  void *extra) {
  std::cout << "op_plugin mock: InplaceNormal called" << std::endl;
  constexpr int expected_nparam = 6;
  if (nparam != expected_nparam || params == nullptr || ndims == nullptr || shapes == nullptr) {
    std::cout << "Invalid parameters for InplaceNormal operator" << std::endl;
    return -1;
  }

  float *x = static_cast<float *>(params[0]);
  int x_ndim = ndims[0];
  int64_t numel = 1;
  for (int i = 0; i < x_ndim; ++i) {
    numel *= shapes[0][i];
  }
  for (size_t i = 0; i < numel; ++i) {
    x[i] = i;  // implemented as iota for simple validation
  }

  return 0;
}

}  // extern "C"
