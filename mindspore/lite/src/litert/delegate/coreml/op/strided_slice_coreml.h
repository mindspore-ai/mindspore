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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_STRIDED_SLICE_COREML_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_STRIDED_SLICE_COREML_H_

#include <vector>
#include <string>
#include "src/litert/delegate/coreml/op/coreml_op.h"
namespace mindspore::lite {
constexpr int ONNX_INPUT_SIZE = 5;
constexpr int MIN_INPUT_SIZE = 4;
constexpr int BEGIN_INDEX = 1;
constexpr int END_INDEX = 2;
constexpr int STRIDE_INDEX = 3;
constexpr int ONNX_STRIDE_INDEX = 4;

class StridedSliceCoreMLOp : public CoreMLOp {
 public:
  StridedSliceCoreMLOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : CoreMLOp(primitive, in_tensors, out_tensors, name) {}

  ~StridedSliceCoreMLOp() override;

  int IsSupport() override;

  int InitParams() override;

  int BuildLayer() override;

  int HandleAxis() override;

 private:
  const schema::StridedSlice *strided_slice_prim_;
  int *begins_idx_ = nullptr;
  bool *begins_mask_ = nullptr;
  int *ends_idx_ = nullptr;
  bool *ends_mask_ = nullptr;
  int *strides_ = nullptr;
  bool *squeeze_mask_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_STRIDED_SLICE_COREML_H_
