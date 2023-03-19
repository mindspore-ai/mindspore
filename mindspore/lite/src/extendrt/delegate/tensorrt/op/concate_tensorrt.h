/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CONCATE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CONCATE_TENSORRT_H_
#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "ops/concat.h"
#include "ops/stack.h"

namespace mindspore::lite {
class ConcateTensorRT : public TensorRTOp {
 public:
  ConcateTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                  const std::vector<TensorInfo> &out_tensors, const std::string &name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {
    if (type_ == ops::kNameConcat) {
      axis_ = AsOps<ops::Concat>()->get_axis();
    } else {
      axis_ = AsOps<ops::Stack>()->get_axis();
    }
  }

  ~ConcateTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  int PreProcessInputs(TensorRTContext *ctx, nvinfer1::ITensor *trt_input_tensors[]);
  int CheckParams(TensorRTContext *ctx);

  Format out_format_{Format::NCHW};
  bool same_format_{true};
  int axis_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_CONCATE_TENSORRT_H_
