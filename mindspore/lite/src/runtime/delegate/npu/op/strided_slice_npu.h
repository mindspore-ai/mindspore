/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_STRIDED_SLICE_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_STRIDED_SLICE_NPU_H_
#include <vector>
#include <string>
#include "include/graph/op/all_ops.h"
#include "src/runtime/delegate/npu/op/npu_op.h"

namespace mindspore {
constexpr int ONNX_INPUT_SIZE = 5;
constexpr int MIN_INPUT_SIZE = 4;
constexpr int BEGIN_INDEX = 1;
constexpr int END_INDEX = 2;
constexpr int STRIDE_INDEX = 3;
constexpr int ONNX_STRIDE_INDEX = 4;

class StridedSliceNPUOp : public NPUOp {
 public:
  StridedSliceNPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                    const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : NPUOp(primitive, in_tensors, out_tensors, name) {}

  ~StridedSliceNPUOp() override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

  int Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
           const std::vector<mindspore::MSTensor> &out_tensors) override;

  int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors,
                   const std::vector<ge::Operator *> &npu_inputs) override;

  ge::Operator *GetNPUOp() override;

  int HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) override;

 private:
  hiai::op::StridedSlice *strided_slice_ = nullptr;
  hiai::op::CastT *in_cast_ = nullptr;
  hiai::op::CastT *out_cast_ = nullptr;
  bool need_cast_ = false;
  int begins_mask_ = 0;
  int ends_mask_ = 0;
  int ellipsis_mask_ = 0;
  int new_axis_mask_ = 0;
  int shrink_axis_mask_ = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_STRIDED_SLICE_NPU_H_
