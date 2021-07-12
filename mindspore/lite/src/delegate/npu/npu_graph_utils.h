/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_GRAPH_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_GRAPH_UTILS_H_

#include <vector>
#include "include/ms_tensor.h"
#include "src/delegate/npu/op/npu_op.h"
namespace mindspore {
class NPUGraphUtils {
 public:
  static std::vector<mindspore::MSTensor> GetGraphInTensors(std::vector<NPUOp *> ops);

  static std::vector<mindspore::MSTensor> GetGraphOutTensors(std::vector<NPUOp *> ops);
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_NPU_GRAPH_UTILS_H_
