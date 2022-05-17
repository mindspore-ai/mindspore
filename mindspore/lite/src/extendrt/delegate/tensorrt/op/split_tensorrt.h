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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_SPLIT_TENSORRT_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_SPLIT_TENSORRT_H_
#include <string>
#include <vector>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
class SplitTensorRT : public TensorRTOp {
 public:
  SplitTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {
    auto split_op = primitive->value_as_Split();
    axis_ = split_op->axis();
    axis_ = axis_ < 0 ? axis_ + in_tensors_[0].Shape().size() : axis_;

    output_num_ = split_op->output_num();

    auto size_splits_ptr = split_op->size_splits();
    if (size_splits_ptr != nullptr) {
      size_splits_.resize(size_splits_ptr->size());
      std::copy(size_splits_ptr->begin(), size_splits_ptr->end(), size_splits_.begin());
    }
  }

  ~SplitTensorRT() override = default;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  int64_t axis_;
  int64_t output_num_;
  std::vector<int64_t> size_splits_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_SPLIT_TENSORRT_H_
