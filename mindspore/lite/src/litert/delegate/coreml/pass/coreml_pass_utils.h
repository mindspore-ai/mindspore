/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_PASS_COREML_PASS_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_PASS_COREML_PASS_UTILS_H_
#include <vector>
#include <set>
#include <string>
#include <unordered_map>
#include "src/litert/delegate/coreml/op/coreml_op.h"
#include "src/litert/delegate/coreml/op/transpose_coreml.h"

namespace mindspore::lite {
class CoreMLPassUtils {
 public:
  static CoreMLOp *CreateNchw2NhwcOp(const std::vector<mindspore::MSTensor> &in_tensors,
                                     const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name);

  static CoreMLOp *CreateNhwc2NchwOp(const std::vector<mindspore::MSTensor> &in_tensors,
                                     const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name);

  static void UpdateOp(CoreMLOp *op, const std::vector<CoreMLOp *> &in_ops, const std::vector<CoreMLOp *> &out_ops,
                       const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors);

  static void UpdateNH2NCTransNodePreOp(CoreMLOp *pre_op, CoreMLOp *trans_op, CoreMLOp *op);

  static void UpdateNC2NHTransNodePreOp(CoreMLOp *pre_op, const std::vector<CoreMLOp *> &trans_ops,
                                        const std::vector<CoreMLOp *> &ops);

  static void UpdateNH2NCTransNodePostOp(CoreMLOp *trans_op, CoreMLOp *post_op);

  static void UpdateNC2NHTransNodePostOp(CoreMLOp *op, CoreMLOp *trans_op, CoreMLOp *post_op,
                                         const mindspore::MSTensor &org_in_tensor);

  static bool IsNhwc2Nchw(CoreMLOp *op);

  static bool IsNchw2Nhwc(CoreMLOp *op);
  static CoreMLOp *OpInputFromOp(CoreMLOp *op, mindspore::MSTensor in_tensor);
  static std::vector<mindspore::MSTensor> GetNonConstInputs(CoreMLOp *op);
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_PASS_COREML_PASS_UTILS_H_
