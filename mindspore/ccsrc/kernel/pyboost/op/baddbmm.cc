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

#include "kernel/pyboost/op/baddbmm.h"
#include "kernel/pyboost/py_boost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void Baddbmm::CastInput() {
  // todo
}

void Baddbmm::InferOutput(const tensor::TensorPtr &input, const tensor::TensorPtr &batch1,
                          const tensor::TensorPtr &batch2, const ScalarPtr &beta, const ScalarPtr &alpha) {
  // todo: DoInfer and get AbstractBasePtr.
  // output_abstract_ = Infer();
  // same shape with input

  auto create_tensor = [](const TypeId &type, const ShapeVector &shape_vector) {
    auto output_tensor = std::make_shared<tensor::Tensor>(type, shape_vector);
    output_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
    MS_LOG(DEBUG) << "Create output tensor " << output_tensor->ToString();
    return output_tensor;
  };

  MS_EXCEPTION_IF_NULL(input);
  output_ = create_tensor(input->data_type(), input->shape());
}

tensor::TensorPtr Baddbmm::Call(const tensor::TensorPtr &input, const tensor::TensorPtr &batch1,
                                const tensor::TensorPtr &batch2, const ScalarPtr &beta, const ScalarPtr &alpha) {
  // TODO: For cpu/gpu, split and run Mul/Add/BatchMatmul.
  return nullptr;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
