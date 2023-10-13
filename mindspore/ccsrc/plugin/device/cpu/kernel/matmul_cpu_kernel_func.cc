/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifdef USE_SELF_DEVELOP
#include <algorithm>
#include <utility>
#include "nnacl/kernel.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/cpu/kernel/nnacl/matmul_parameter.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/matmul_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "mindspore/core/ops/mat_mul.h"
#include "plugin/device/cpu/kernel/matmul_cpu_kernel_func.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel/init_exec_env.h"
#include "nnacl/kernel/matmul_struct.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMatMulInputsNum = 2;
constexpr size_t kMatMulWithBiasAddInputsNum = 3;
constexpr size_t kBiasAddInputIndex = kMatMulWithBiasAddInputsNum - 1;
constexpr size_t kMatMulOutputsNum = 1;
constexpr size_t kIndexOffset = 2;
constexpr size_t kRankMin = 2;
}  // namespace

MatMulCpuKernelFunc::~MatMulCpuKernelFunc() {
  if (in_ != nullptr) {
    for (size_t i = 0; i < in_size_; i++) {
      delete in_[i];
    }
    free(in_);
    in_ = nullptr;
  }
  if (out_ != nullptr) {
    for (size_t i = 0; i < out_size_; i++) {
      delete out_[i];
    }
    free(out_);
    out_ = nullptr;
  }
  delete op_parameter_;
  if (kernel_ != nullptr) {
    kernel_->Release(kernel_);

    free(kernel_);
    kernel_ = nullptr;
  }
}

void MatMulCpuKernelFunc::InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MatMul>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast MatMul ops failed!";
  }
  trans_a_ = kernel_ptr->get_transpose_a();
  trans_b_ = kernel_ptr->get_transpose_b();

  in_size_ = inputs.size();
  out_size_ = outputs.size();
  if (kernel_ptr->HasAttr(kAttrWithBiasAdd)) {
    with_bias_add_ = GetValue<bool>(kernel_ptr->GetAttr(kAttrWithBiasAdd));
  }
  if (kernel_ptr->HasAttr(kAttrWithRelu)) {
    with_relu_ = GetValue<bool>(kernel_ptr->GetAttr(kAttrWithRelu));
  }
  in_ = reinterpret_cast<TensorC **>(malloc(in_size_ * sizeof(TensorC *)));
  if (in_ == nullptr) {
    MS_LOG(EXCEPTION) << "malloc in_ for matmul kernel failed.";
  }

  out_ = reinterpret_cast<TensorC **>(malloc(out_size_ * sizeof(TensorC *)));
  if (out_ == nullptr) {
    MS_LOG(EXCEPTION) << "malloc out_ for matmul kernel failed.";
  }
  for (size_t i = 0; i < in_size_; i++) {
    in_[i] = new TensorC;
    auto inputs_shape_i = inputs[i]->GetShapeVector();
    in_[i]->shape_size_ = inputs_shape_i.size();
    for (size_t j = 0; j < inputs_shape_i.size(); j++) {
      in_[i]->shape_[j] = inputs_shape_i[j];
    }
    in_[i]->data_type_ = inputs[i]->GetDtype();
  }
  for (size_t i = 0; i < out_size_; i++) {
    out_[i] = new TensorC;
    auto outputs_shape_i = outputs[i]->GetShapeVector();
    out_[i]->shape_size_ = outputs_shape_i.size();
    for (size_t j = 0; j < outputs_shape_i.size(); j++) {
      out_[i]->shape_[j] = outputs_shape_i[j];
    }
    out_[i]->data_type_ = outputs[i]->GetDtype();
  }
  op_parameter_ = new OpParameter;
  int thread_num = GetActorMgrInnerThreadPool()->GetKernelThreadNum();
  if (thread_num == 0) {
    thread_num = 1;
  }
  auto matmul_param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  matmul_param->b_transpose_ = trans_b_;
  matmul_param->a_transpose_ = trans_a_;
  matmul_param->has_bias_ = with_bias_add_;
  if (with_relu_) {
    matmul_param->act_type_ = ActType_Relu;
  } else {
    matmul_param->act_type_ = ActType_No;
  }
  op_parameter_->thread_num_ = thread_num;
  op_parameter_->type_ = PrimType_MatMulFusion;
  op_parameter_->is_train_session_ = false;
  kernel_ = CreateKernel(op_parameter_, in_, in_size_, out_, out_size_, out_[0]->data_type_, exec_env_);
  if (kernel_ == nullptr) {
    MS_LOG(EXCEPTION) << "NNACL create matmul kernel failed.";
  }
}

bool MatMulCpuKernelFunc::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (kernel_ == nullptr) {
    return false;
  }
  for (size_t i = 0; i < in_size_; i++) {
    in_[i]->data_ = inputs[i]->addr;
  }
  for (size_t i = 0; i < out_size_; i++) {
    out_[i]->data_ = outputs[i]->addr;
  }
  int ret = kernel_->Prepare(kernel_);
  if (ret != 0) {
    MS_LOG(ERROR) << "NNACL matmul/fc prepare failed. ret: " << ret;
    return ret;
  }
  ret = kernel_->Resize(kernel_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "NNACL matmul/fc resize failed. ret: " << ret;
    return ret;
  }
  ret = kernel_->Compute(kernel_);
  if (ret != 0) {
    MS_LOG(ERROR) << "NNACL compute failed. ret: " << ret;
    return false;
  }
  return true;
}

int MatMulCpuKernelFunc::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (kernel_ == nullptr) {
    return 0;
  }

  MatmulStruct *matmul = reinterpret_cast<MatmulStruct *>(kernel_);
  matmul->is_sharing_pack_ = false;
  matmul->infer_shape_ = true;
  matmul->a_const_ = false;
  matmul->b_const_ = false;
  matmul->bias_need_repack_ = true;
  matmul = reinterpret_cast<MatmulStruct *>(kernel_);
  matmul->model_thread_nr_ = kernel_->thread_nr_;

  for (size_t i = 0; i < in_size_; i++) {
    auto inputs_shape_i = inputs[i]->GetShapeVector();
    in_[i]->shape_size_ = inputs_shape_i.size();
    for (size_t j = 0; j < inputs_shape_i.size(); j++) {
      in_[i]->shape_[j] = inputs_shape_i[j];
    }
    in_[i]->data_type_ = inputs[i]->GetDtype();
  }
  for (size_t i = 0; i < out_size_; i++) {
    auto outputs_shape_i = outputs[i]->GetShapeVector();
    out_[i]->shape_size_ = outputs_shape_i.size();
    for (size_t j = 0; j < outputs_shape_i.size(); j++) {
      out_[i]->shape_[j] = outputs_shape_i[j];
    }
    out_[i]->data_type_ = outputs[i]->GetDtype();
  }
  return 0;
}
}  // namespace kernel
}  // namespace mindspore
#endif
