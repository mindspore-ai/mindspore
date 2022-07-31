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

#include "src/litert/delegate/coreml/op/matmul_coreml.h"
namespace mindspore::lite {
int MatMulCoreMLOp::IsSupport() {
  MS_CHECK_GE(in_tensors_.size(), kInputSize1, RET_NOT_SUPPORT);
  if (in_tensors_.size() > kInputSize1 && !in_tensors_.at(SECOND_INPUT).IsConst()) {
    MS_LOG(WARNING) << "Bias for CoreML matmul is supported only when the second input is a constant.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int MatMulCoreMLOp::InitParams() {
  matmul_prim_ = op_primitive_->value_as_MatMulFusion();
  if (matmul_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  matmul_param_ = op_->mutable_batchedmatmul();
  matmul_param_->set_transposea(matmul_prim_->transpose_a());
  matmul_param_->set_transposeb(matmul_prim_->transpose_b());
  if (in_tensors_.at(SECOND_INPUT).IsConst()) {
    if (matmul_prim_->transpose_b()) {
      auto ret = ConstMatMulWithTransB();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Build MatMul layer with const input and true TransposeB failed for op: " << name_;
        return RET_ERROR;
      }
    } else {
      // CoreML will automatically transpose the const input even though transposeB is false.
      auto ret = ConstMatMulWithoutTransB();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Build MatMul layer with const input and false TransposeB failed for op: " << name_;
        return RET_ERROR;
      }
    }
  }
  auto act_type = matmul_prim_->activation_type();
  if (act_type != schema::ActivationType_NO_ACTIVATION) {
    auto ret = SetActivation(act_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set matmul activation failed for op: " << name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int MatMulCoreMLOp::ConstMatMulWithTransB() {
  MS_ASSERT(matmul_param_ != nullptr);
  auto input_b = in_tensors_.at(SECOND_INPUT);
  auto dim_b = input_b.Shape().size();
  int64_t in_channel =
    matmul_prim_->transpose_b() ? input_b.Shape()[dim_b - DIMENSION_1D] : input_b.Shape()[dim_b - DIMENSION_2D];
  int64_t out_channel =
    matmul_prim_->transpose_b() ? input_b.Shape()[dim_b - DIMENSION_2D] : input_b.Shape()[dim_b - DIMENSION_1D];
  matmul_param_->set_weightmatrixfirstdimension(in_channel);
  matmul_param_->set_weightmatrixseconddimension(out_channel);
  auto org_weight = input_b.Data().get();
  if (input_b.DataType() == DataType::kNumberTypeFloat32) {
    auto *ml_weight_container = matmul_param_->mutable_weights()->mutable_floatvalue();
    ml_weight_container->Resize(input_b.ElementNum(), 0);
    auto *ml_weight = reinterpret_cast<void *>(ml_weight_container->mutable_data());
    memcpy(ml_weight, org_weight, input_b.DataSize());
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight tensor for CoreML convolution.";
    return RET_ERROR;
  }
  if (in_tensors_.size() > kInputSize1) {
    auto bias_tensor = in_tensors_.at(THIRD_INPUT);
    auto org_bias = bias_tensor.Data().get();
    matmul_param_->set_hasbias(true);
    if (bias_tensor.DataType() == DataType::kNumberTypeFloat32) {
      auto *ml_bias_container = matmul_param_->mutable_bias()->mutable_floatvalue();
      ml_bias_container->Resize(bias_tensor.ElementNum(), 0);
      auto *ml_bias = reinterpret_cast<void *>(ml_bias_container->mutable_data());
      memcpy(ml_bias, org_bias, bias_tensor.DataSize());
    } else {
      MS_LOG(ERROR) << "Unsupported data type of bias tensor for CoreML convolution.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int MatMulCoreMLOp::ConstMatMulWithoutTransB() {
  MS_ASSERT(matmul_param_ != nullptr);
  auto ret = SetConstInput(in_tensors_[SECOND_INPUT]);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set const input failed for op: " << name_;
    return RET_ERROR;
  }
  if (in_tensors_.size() > kInputSize1) {
    // when the second input is not const anymore, the bias param will be invalid.
    bias_op_ = std::make_unique<CoreML::Specification::NeuralNetworkLayer>();
    if (bias_op_ == nullptr) {
      MS_LOG(ERROR) << "New CoreML op " << name_ << "_bias failed.";
      return RET_ERROR;
    }
    bias_op_->set_name("CoreML_" + name_ + "_bias");
    (void)bias_op_->mutable_addbroadcastable();
    ret = SetConstInput(in_tensors_[THIRD_INPUT]);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set const input failed for op: " << name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void MatMulCoreMLOp::SetMLOpInOut() {
  MS_ASSERT(op_ != nullptr);
  op_->add_input(in_tensors_.at(FIRST_INPUT).Name());
  auto input_b_name = in_tensors_.at(SECOND_INPUT).Name();
  auto output_name = out_tensors_.at(0).Name();
  if (!in_tensors_.at(SECOND_INPUT).IsConst()) {
    op_->add_input(input_b_name);
  } else if (!const_ops_.empty()) {
    const_ops_[input_b_name]->add_output(input_b_name);
    op_->add_input(input_b_name);
    if (bias_op_ != nullptr) {
      std::string bias_name = op_->name() + "_bias_0";
      op_->add_output(bias_name);
      bias_op_->add_input(bias_name);
      auto input_c_name = in_tensors_.at(THIRD_INPUT).Name();
      const_ops_[input_c_name]->add_output(input_c_name);
      bias_op_->add_input(input_c_name);
    }
  }
  if (act_op_ != nullptr) {
    std::string act_name = op_->name() + "_act_0";
    if (bias_op_ != nullptr) {
      bias_op_->add_output(act_name);
    } else {
      op_->add_output(act_name);
    }
    act_op_->add_input(act_name);
    act_op_->add_output(output_name);
    return;
  }
  if (bias_op_ != nullptr) {
    bias_op_->add_output(output_name);
  } else {
    op_->add_output(output_name);
  }
  return;
}

std::vector<CoreML::Specification::NeuralNetworkLayer *> MatMulCoreMLOp::GetLayers() {
  MS_ASSERT(op_ != nullptr);
  std::vector<CoreML::Specification::NeuralNetworkLayer *> ret_ops;
  for (auto it = const_ops_.begin(); it != const_ops_.end(); it++) {
    ret_ops.push_back(it->second.release());
  }
  ret_ops.push_back(op_.release());
  if (bias_op_ != nullptr) {
    ret_ops.push_back(bias_op_.release());
  }
  if (act_op_ != nullptr) {
    ret_ops.push_back(act_op_.release());
  }
  return ret_ops;
}
}  // namespace mindspore::lite
