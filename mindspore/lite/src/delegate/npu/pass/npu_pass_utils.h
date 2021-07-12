/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_PASS_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_PASS_UTILS_H_
#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include "src/delegate/npu/op/npu_op.h"
#include "src/delegate/npu/op/transpose_npu.h"

namespace mindspore {
extern std::unordered_map<schema::PrimitiveType, std::set<int>> nodes2const_index;
class NPUPassUtils {
 public:
  static NPUOp *CreateNchw2NhwcOp(const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name);

  static NPUOp *CreateNhwc2NchwOp(const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name);

  static void UpdateOp(NPUOp *op, const std::vector<NPUOp *> &in_ops, const std::vector<NPUOp *> &out_ops,
                       const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors);

  static void UpdateNH2NCTransNodePreOp(NPUOp *pre_op, NPUOp *trans_op, NPUOp *op);

  static void UpdateNC2NHTransNodePreOp(NPUOp *pre_op, const std::vector<NPUOp *> &trans_ops,
                                        const std::vector<NPUOp *> &ops);

  static void UpdateNH2NCTransNodePostOp(NPUOp *trans_op, NPUOp *post_op);

  static void UpdateNC2NHTransNodePostOp(NPUOp *op, NPUOp *trans_op, NPUOp *post_op);

  static void UpdateNC2NHPostOpInTensors(NPUOp *op, NPUOp *trans_op, NPUOp *post_op);

  static bool IsNhwc2Nchw(NPUOp *op);

  static bool IsNchw2Nhwc(NPUOp *op);
  static NPUOp *OpInputFromOp(NPUOp *op, mindspore::MSTensor in_tensor);
  static std::vector<mindspore::MSTensor> GetNonConstInputs(NPUOp *op);
  static bool Scale4dCase(NPUOp *op);
  static void AssistDataNHWC2NCHW(int *data, size_t unit_size);
  static int MaskDataNHWC2NCHW(int mask);
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_PASS_NPU_PASS_UTILS_H_
