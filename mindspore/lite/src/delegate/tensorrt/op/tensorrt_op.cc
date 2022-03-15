/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/tensorrt/op/tensorrt_op.h"
#include "src/delegate/tensorrt/tensorrt_runtime.h"
#include <unordered_map>

namespace mindspore::lite {
const schema::Primitive *TensorRTOp::GetPrimitive() { return this->op_primitive_; }

void TensorRTOp::AddInnerInTensors(ITensorHelper tensor) { this->tensorrt_in_tensors_.push_back(tensor); }

void TensorRTOp::AddInnerOutTensors(ITensorHelper tensor) { this->tensorrt_out_tensors_.push_back(tensor); }

std::vector<ITensorHelper> &TensorRTOp::GetInnerOutTensor() { return this->tensorrt_out_tensors_; }

std::vector<ITensorHelper> &TensorRTOp::GetInnerInTensors() { return this->tensorrt_in_tensors_; }

std::string TensorRTOp::GetOpName() { return this->op_name_; }

std::vector<mindspore::MSTensor> &TensorRTOp::inputs() { return this->in_tensors_; }

std::vector<mindspore::MSTensor> &TensorRTOp::outputs() { return this->out_tensors_; }

schema::PrimitiveType TensorRTOp::type() const { return this->type_; }

schema::QuantType TensorRTOp::GetQuantType() const { return this->quant_type_; }

void TensorRTOp::set_in_ops(const std::vector<TensorRTOp *> &in_ops) { this->in_ops_ = in_ops; }

void TensorRTOp::set_out_ops(const std::vector<TensorRTOp *> &out_ops) { this->out_ops_ = out_ops; }

const std::vector<TensorRTOp *> &TensorRTOp::in_ops() const { return this->in_ops_; }

const std::vector<TensorRTOp *> &TensorRTOp::out_ops() const { return this->out_ops_; }

void TensorRTOp::SetRuntime(TensorRTRuntime *runtime) { this->runtime_ = runtime; }

bool TensorRTOp::IsShapeKnown() {
  if (this->in_tensors_.size() == 1 && this->in_tensors_[0].Shape().size() == 0) {
    return false;
  }
  return true;
}

int TensorRTOp::Prepare(void **network_tensor_bindings, nvinfer1::ICudaEngine *engine) {
  if (op_binding_tensor_.size() != 0) {
    MS_LOG(ERROR) << "need special op Prepare for " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

DynamicShapeParams TensorRTOp::GetDynamicShapeParams() const { return this->dynamic_shape_params_; }

int TensorRTOp::SetInt8DynamicRange() {
  // setting param layer_ forcely
  if (this->layer_ == nullptr) {
    MS_LOG(ERROR) << op_name_ << " layer is nullptr.";
    return RET_ERROR;
  }
  if (in_tensors_.empty() || out_tensors_.empty()) {
    MS_LOG(ERROR) << "input or output tensor empty.";
    return RET_ERROR;
  }
  if (quant_type_ != schema::QuantType_QUANT_ALL) {
    MS_LOG(DEBUG) << "op " << op_name_ << " not quantized.";
    return RET_OK;
  }

  if (in_tensors_[0].QuantParams().empty() || out_tensors_[0].QuantParams().empty()) {
    MS_LOG(WARNING) << op_name_ << " quant param is empty.";
    MS_LOG(WARNING) << "in_tensor quant param size: " << in_tensors_[0].QuantParams().size()
                    << " ,out_tensor quant param size: " << out_tensors_[0].QuantParams().size();
  }
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto tensor = in_tensors_.at(i);
    if (!tensor.IsConst()) {
      tensorrt_in_tensors_.at(i).trt_tensor_->setDynamicRange(tensor.QuantParams().at(0).min,
                                                              tensor.QuantParams().at(0).max);
      // Don't set the presion on non-computation layers as they don't support int8.
      if (this->layer_->getType() != nvinfer1::LayerType::kCONSTANT &&
          this->layer_->getType() != nvinfer1::LayerType::kCONCATENATION &&
          this->layer_->getType() != nvinfer1::LayerType::kSHAPE) {
        this->layer_->setPrecision(nvinfer1::DataType::kINT8);
      }
    }
  }
  for (size_t i = 0; i < out_tensors_.size(); i++) {
    auto tensor = out_tensors_.at(0);
    tensorrt_out_tensors_.at(i).trt_tensor_->setDynamicRange(tensor.QuantParams().at(0).min,
                                                             tensor.QuantParams().at(0).max);
    // set output type of execution tensors.
    if (this->layer_->getOutput(i)->isExecutionTensor()) {
      this->layer_->setOutputType(i, nvinfer1::DataType::kINT8);
    }
  }
  return SetTransposeDynamicRange();
}

int TensorRTOp::SetTransposeDynamicRange() {
  if (this->transpose_layer_ == nullptr) {
    MS_LOG(INFO) << op_name_ << " transpose_layer is nullptr.";
    return RET_OK;
  }
  if (!in_tensors_[0].QuantParams().empty() && !out_tensors_[0].QuantParams().empty()) {
    this->transpose_layer_->getInput(0)->setDynamicRange(in_tensors_.front().QuantParams().at(0).min,
                                                         in_tensors_.front().QuantParams().at(0).max);
    this->transpose_layer_->getOutput(0)->setDynamicRange(in_tensors_.front().QuantParams().at(0).min,
                                                          in_tensors_.front().QuantParams().at(0).max);
    this->transpose_layer_->setOutputType(0, nvinfer1::DataType::kINT8);
    this->transpose_layer_->setPrecision(nvinfer1::DataType::kINT8);
  }
  return RET_OK;
}
}  // namespace mindspore::lite
