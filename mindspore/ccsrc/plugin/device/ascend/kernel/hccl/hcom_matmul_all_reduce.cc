/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "hccl/hccl_types.h"
#include "plugin/device/ascend/kernel/hccl/hcom_matmul_all_reduce.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "runtime/rt.h"

namespace mindspore {
namespace kernel {
bool HcomMatMulAllReduceKernel::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kMatMulAllReduceInputNum) {
    MS_LOG(EXCEPTION) << "Input number of MatMulAllReduce should be 2, but got " << inputs.size();
  }
  if (outputs.size() != kMatMulAllReduceOutputNum) {
    MS_LOG(EXCEPTION) << "Output number of MatMulAllReduce should be 1, but got " << outputs.size();
  }

  if (!HcclKernel::Init(inputs, outputs)) {
    MS_LOG(ERROR) << "Call HcclKernel::Init failed.";
    return false;
  }

  if (!HcomUtil::GetHcomAttr<bool>(primitive_, kAttrNameTransposeA, &transpose_a_)) {
    return false;
  }
  if (!HcomUtil::GetHcomAttr<bool>(primitive_, kAttrNameTransposeB, &transpose_b_)) {
    return false;
  }

  if (GetHcclDataType() != HCCL_DATA_TYPE_FP16 && GetHcclDataType() != HCCL_DATA_TYPE_BFP16) {
    MS_LOG(EXCEPTION) << "MatMulAllReduce only support data type fp16 or bf16.";
  }
  lcoc_dtype_ = (GetHcclDataType() == HCCL_DATA_TYPE_FP16) ? Lcal::CoCDataTypeDesc::FP16FP16_FP32_FP16
                                                           : Lcal::CoCDataTypeDesc::BF16BF16_FP32_BF16;

  // Dynamic load lcoc symbols.
  auto get_lcoc_func = DlsymFuncObj(CreateLcocForOp, lowlatency_comm_lib_handle_);
  lcoc_ptr_ = get_lcoc_func(group_);
  MS_EXCEPTION_IF_NULL(lcoc_ptr_);

  set_param_for_lcoc_func_ = DlsymFuncObj(SetParamForLcoc, lowlatency_comm_lib_handle_);
  MS_EXCEPTION_IF_NULL(set_param_for_lcoc_func_);

  get_lcoc_workspace_func_ = DlsymFuncObj(GetLcocWorkspaceSize, lowlatency_comm_lib_handle_);
  MS_EXCEPTION_IF_NULL(get_lcoc_workspace_func_);

  matmul_all_reduce_func_ = DlsymFuncObj(MatmulAllReduce, lowlatency_comm_lib_handle_);
  MS_EXCEPTION_IF_NULL(matmul_all_reduce_func_);
  return true;
}

int HcomMatMulAllReduceKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = HcclKernel::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "Resize failed";
    return ret;
  }

  // The dimensions of left and right matrices.
  matmul_info_.m = transpose_a_ ? hccl_kernel_input_shape_list_[0][1] : hccl_kernel_input_shape_list_[0][0];
  matmul_info_.k = transpose_a_ ? hccl_kernel_input_shape_list_[0][0] : hccl_kernel_input_shape_list_[0][1];
  matmul_info_.n = transpose_b_ ? hccl_kernel_input_shape_list_[1][0] : hccl_kernel_input_shape_list_[1][1];
  matmul_info_.transA = transpose_a_;
  matmul_info_.transB = transpose_b_;

  MS_LOG(INFO) << matmul_info_.m << " " << matmul_info_.k << " " << matmul_info_.n << " " << matmul_info_.transA << " "
               << matmul_info_.transB << " " << lcoc_dtype_ << " " << op_type_ << lcoc_type_;
  param_desc_.dataTypeDesc = lcoc_dtype_;
  param_desc_.mmInfo = matmul_info_;
  param_desc_.quantInfo = quant_info_;
  param_desc_.op = op_type_;
  set_param_for_lcoc_func_(lcoc_ptr_, lcoc_type_, tiling_, param_desc_);

  workspace_size_list_.clear();
  workspace_size_list_.push_back(get_lcoc_workspace_func_(lcoc_ptr_));
  return KRET_OK;
}

bool HcomMatMulAllReduceKernel::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "MatMulAllReduce launch";
  if (inputs.empty() || outputs.empty() || workspace.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid AllReduce input, output or data type size (" << inputs.size() << ", " << outputs.size()
                  << ", " << workspace.size() << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(inputs[1]);
  MS_EXCEPTION_IF_NULL(workspace[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  Lcal::CoCInputPkg coc_input_args = {
    inputs[0]->device_ptr(), inputs[1]->device_ptr(), nullptr, nullptr, nullptr, nullptr, nullptr};
  Lcal::CoCOutputPkg coc_output_args = {outputs[0]->device_ptr(), nullptr};
  auto lccl_result =
    matmul_all_reduce_func_(lcoc_ptr_, coc_input_args, coc_output_args, workspace[0]->device_ptr(), stream_ptr);
  if (lccl_result != Lcal::LCAL_SUCCESS) {
    MS_LOG(EXCEPTION) << "LCOC MatmulAllReduce failed.";
  }

  return true;
}
}  // namespace kernel
}  // namespace mindspore
