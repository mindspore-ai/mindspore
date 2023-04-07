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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_UNPACK_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_UNPACK_NPU_H_
#include <vector>
#include <string>
#include "include/graph/op/all_ops.h"
#include "src/litert/delegate/npu/op/npu_op.h"

namespace mindspore {
namespace lite {
class UnpackNPUOp : public NPUOp {
 public:
  UnpackNPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
              const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : NPUOp(primitive, in_tensors, out_tensors, name) {}

  ~UnpackNPUOp();

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override {
    return RET_OK;
  }

  int Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
           const std::vector<mindspore::MSTensor> &out_tensors) override;

  int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensor,
                   const std::vector<ge::Operator *> &npu_inputs) override;

  int HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) override;

  ge::Operator *GetNPUOp() override;

 private:
  hiai::op::Unpack *unpack_ = nullptr;
  int axis_ = 0;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_UNPACK_NPU_H_
