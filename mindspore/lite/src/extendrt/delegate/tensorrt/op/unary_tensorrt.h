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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_UNARY_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_UNARY_TENSORRT_H_
#include <string>
#include <vector>
#include <map>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"

#include "ops/fusion/exp_fusion.h"
#include "ops/sqrt.h"
#include "ops/abs.h"
#include "ops/log.h"
#include "ops/neg.h"
#include "ops/sin.h"
#include "ops/cos.h"
#include "ops/ceil.h"
#include "ops/floor.h"
#include "ops/logical_not.h"

namespace mindspore::lite {
class UnaryTensorRT : public TensorRTOp {
 public:
  UnaryTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~UnaryTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;

 private:
  std::map<std::string, nvinfer1::UnaryOperation> unary_ops_ = {
    {ops::kNameSqrt, nvinfer1::UnaryOperation::kSQRT},
    {ops::kNameAbs, nvinfer1::UnaryOperation::kABS},
    {ops::kNameNeg, nvinfer1::UnaryOperation::kNEG},
    {ops::kNameLog, nvinfer1::UnaryOperation::kLOG},
    {ops::kNameSin, nvinfer1::UnaryOperation::kSIN},
    {ops::kNameCos, nvinfer1::UnaryOperation::kCOS},
    {ops::kNameCeil, nvinfer1::UnaryOperation::kCEIL},
    {ops::kNameFloor, nvinfer1::UnaryOperation::kFLOOR},
    {ops::kNameExpFusion, nvinfer1::UnaryOperation::kEXP},
#if TRT_VERSION_GE(7, 2)
    {ops::kNameLogicalNot, nvinfer1::UnaryOperation::kNOT},
#endif
  };
  nvinfer1::UnaryOperation unary_op_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_UNARY_TENSORRT_H_
