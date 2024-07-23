/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <string.h>
#include <cstdint>
#include <vector>
#include "custom_aot_extra.h"

extern "C" std::vector<int64_t> CustomAddInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
  const int64_t kDynRankSize = -2;
  if (shapes[0][0] == kDynRankSize) {
    return std::vector<int64_t>{shapes[0][0]};
  }
  return std::vector<int64_t>{shapes[0][0], shapes[0][1]};
}

extern "C" int CustomAdd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
  constexpr int OUTPUT_INDEX = 2;
  constexpr int TOTAL_PARAM_NUM = 3;

  // Users can add any check on their need. If check fails, user can return any value larger than 0 to safely exit.
  // Return value not equal to 0 will cause MindSpore to stop computing and safely exit.

  // This is to check if the num of parameters the same as what the user wants.
  // In this case, there are two inputs and one output, so the nparam should be 3.
  if (nparam != TOTAL_PARAM_NUM) {
    return 1;
  }

  // This is to check if the type of parameters the same as what the user wants.
  for (int i = 0; i < nparam; i++) {
    if (strcmp(dtypes[i], "float32") != 0) {
      return 2;
    }
  }

  // input1's index is 0, input2's index is 1 and output's index is 2
  float *input1 = static_cast<float *>(params[0]);
  float *input2 = static_cast<float *>(params[1]);
  float *output = static_cast<float *>(params[2]);
  int size = 1;

  // Cumprod of output's shape to compute elements' num
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++) {
    size *= shapes[OUTPUT_INDEX][i];
  }

  // Do the computation
  // Add
  for (int i = 0; i < size; i++) {
    output[i] = input1[i] + input2[i];
  }

  // When return 0, MindSpore will continue to run if this kernel could launch successfully.
  return 0;
}
