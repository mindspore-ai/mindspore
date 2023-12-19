/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_UTIL_H_
#define MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_UTIL_H_

#include <vector>
#include <string>
#include "utils/ms_context.h"
#include "ir/anf.h"

namespace mindspore::transform {
typedef enum : int8_t {
  KEEP_DTYPE = 0,
  ALLOW_FP32_DOWN_PRECISION = 1,
  FORCE_FP16 = 2,
  FORCE_HF32 = 3,
} aclCubeMathType;

class OpApiUtil {
 public:
  static aclCubeMathType GetCubeMathType(bool use_hf32 = false);

  static void GetValidKernelBuildInfo(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                      std::vector<std::string> *output_formats,
                                      std::vector<std::string> *input_reshape_types,
                                      std::vector<std::string> *output_reshape_types);
};
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_ACL_IR_OP_API_UTIL_H_
