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
 * See the License for the specific language governing shapeissions and
 * limitations under the License.
 */

#include "src/litert/delegate/coreml/op/reshape_coreml.h"
namespace mindspore::lite {
int ReshapeCoreMLOp::IsSupport() {
  MS_CHECK_GE(in_tensors_.size(), kInputSize1, RET_NOT_SUPPORT);
  return RET_OK;
}

int ReshapeCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  auto shape_tensor = in_tensors_.at(1);
  if (shape_tensor.IsConst()) {
    auto shape_dim = shape_tensor.ElementNum();
    auto shape_data = reinterpret_cast<const int *>(shape_tensor.Data().get());
    auto shape_param = op_->mutable_reshapestatic();
    for (int i = 0; i < shape_dim; i++) {
      shape_param->add_targetshape(shape_data[i]);
    }
  } else {
    op_->mutable_reshapedynamic();
  }
  return RET_OK;
}

void ReshapeCoreMLOp::SetMLOpInOut() {
  MS_ASSERT(op_ != nullptr);
  op_->add_input(in_tensors_[0].Name());
  if (!in_tensors_[1].IsConst()) {
    op_->add_input(in_tensors_[1].Name());
  }
  op_->add_output(out_tensors_[0].Name());
}
}  // namespace mindspore::lite
