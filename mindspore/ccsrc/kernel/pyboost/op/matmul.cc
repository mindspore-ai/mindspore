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

#include "kernel/pyboost/op/matmul.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void Matmul::CastInput() {}

void Matmul::InferOutput(const tensor::TensorPtr &x, const tensor::TensorPtr &y) {
  auto eval_impl = abstract::GetPrimitiveInferImpl(primitive_);
  if (!eval_impl.has_value()) {
    MS_LOG(EXCEPTION) << "Not found infer func for Matmul";
  }
  std::vector<AbstractBasePtr> input_abs = {x->ToAbstract(), y->ToAbstract()};
  auto output_abs = eval_impl->InferShapeAndType(nullptr, primitive_, input_abs);

  std::vector<TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(output_abs, &outputs);
  if (outputs.empty()) {
    MS_LOG(EXCEPTION) << "Cannot create output tensor for Matmul";
  }
  output_ = outputs[0];
}

tensor::TensorPtr Matmul::Call(const tensor::TensorPtr &x, const tensor::TensorPtr &y) {
  // TODO: kernel_mod->launch
  return mindspore::tensor::TensorPtr();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore