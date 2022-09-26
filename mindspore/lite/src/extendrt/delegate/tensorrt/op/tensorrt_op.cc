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

#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include <unordered_map>
#include "src/extendrt/delegate/tensorrt/tensorrt_runtime.h"

namespace mindspore::lite {
TensorRTOp::TensorRTOp(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                       const std::vector<TensorInfo> &out_tensors, std::string name)
    : base_operator_(base_operator), in_tensors_(in_tensors), out_tensors_(out_tensors), op_name_(std::move(name)) {
  MS_EXCEPTION_IF_NULL(base_operator);

  this->type_ = base_operator->name();
  auto primitive_c = base_operator->GetPrim();
  if (primitive_c != nullptr) {
    return;
  }
}

const BaseOperatorPtr &TensorRTOp::GetBaseOperator() { return this->base_operator_; }

std::string TensorRTOp::GetOpName() { return this->op_name_; }

std::vector<TensorInfo> &TensorRTOp::inputs() { return this->in_tensors_; }

std::vector<TensorInfo> &TensorRTOp::outputs() { return this->out_tensors_; }

ITensorHelper TensorRTOp::input(TensorRTContext *ctx, size_t i) {
  auto in_ms_tensor = in_tensors_[i];
  ITensorHelper in_trt_tensor = ctx->MsName2Tensor(in_ms_tensor.Name());

  if (!GetSupportInputBool() && in_ms_tensor.DataType() == DataType::kNumberTypeBool) {
    ITensorHelper in_trt_tensor_cast = ctx->MsName2Tensor(in_ms_tensor.Name() + "_to_int32");
    if (in_trt_tensor_cast.trt_tensor_ == nullptr) {
      auto cast_trt_tensor =
        TRTTensorCast(ctx, in_trt_tensor.trt_tensor_, nvinfer1::DataType::kINT32, in_ms_tensor.Name() + "_cast_int32");
      in_trt_tensor_cast = ITensorHelper{cast_trt_tensor, in_ms_tensor.format(), true};
      ctx->RegisterTensor(in_trt_tensor_cast, in_ms_tensor.Name() + "_to_int32");
    }
    return in_trt_tensor_cast;
  }
  return in_trt_tensor;
}

ITensorHelper TensorRTOp::output(TensorRTContext *ctx, size_t i) { return ctx->MsName2Tensor(out_tensors_[i].Name()); }

const std::string &TensorRTOp::type() const { return this->type_; }

schema::QuantType TensorRTOp::GetQuantType() const { return this->quant_type_; }

void TensorRTOp::set_in_ops(const std::vector<TensorRTOp *> &in_ops) { this->in_ops_ = in_ops; }

void TensorRTOp::set_out_ops(const std::vector<TensorRTOp *> &out_ops) { this->out_ops_ = out_ops; }

const std::vector<TensorRTOp *> &TensorRTOp::in_ops() const { return this->in_ops_; }

const std::vector<TensorRTOp *> &TensorRTOp::out_ops() const { return this->out_ops_; }

void TensorRTOp::SetRuntime(TensorRTRuntime *runtime) {
  this->runtime_ = runtime;
  device_id_ = runtime_->GetDeviceID();
}

bool TensorRTOp::HasConst() const {
  return std::any_of(in_tensors_.begin(), in_tensors_.end(),
                     [](const TensorInfo &tensor) { return tensor.Data() != nullptr && tensor.IsConst(); });
}

int TensorRTOp::ReadyInputsNumber(TensorRTContext *ctx) const {
  return std::count_if(in_tensors_.begin(), in_tensors_.end(),
                       [&](const TensorInfo &tensor) { return ctx->HasTensor(tensor.Name()); });
}

bool TensorRTOp::IsShapeKnown() { return true; }

bool TensorRTOp::IsDynamicInput(TensorRTContext *ctx, size_t k) {
  nvinfer1::Dims dims = input(ctx, k).trt_tensor_->getDimensions();
  return std::any_of(dims.d, dims.d + dims.nbDims, [](int d) { return d == -1; });
}

int TensorRTOp::Prepare(void **network_tensor_bindings, nvinfer1::ICudaEngine *engine) {
  if (op_binding_tensor_.size() != 0) {
    MS_LOG(ERROR) << "need special op Prepare for " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

DynamicShapeParams TensorRTOp::GetDynamicShapeParams() const { return this->dynamic_shape_params_; }

int TensorRTOp::SetInt8DynamicRange(TensorRTContext *ctx) {
  // setting param layer_ forcely
  if (this->layer_ == nullptr) {
    MS_LOG(ERROR) << op_name_ << " layer is nullptr.";
    return RET_ERROR;
  }
  if (in_tensors_.empty() || out_tensors_.empty()) {
    MS_LOG(ERROR) << "input or output tensor empty.";
    return RET_ERROR;
  }
  return RET_OK;
}

int TensorRTOp::SetTransposeDynamicRange() {
  if (this->transpose_layer_ == nullptr) {
    MS_LOG(INFO) << op_name_ << " transpose_layer is nullptr.";
    return RET_OK;
  }
  return RET_OK;
}

bool TensorRTOp::GetSupportInputBool() { return this->support_input_bool_; }

void TensorRTOp::SetSupportInputBool(bool support_input_bool) { this->support_input_bool_ = support_input_bool; }
}  // namespace mindspore::lite
