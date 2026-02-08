/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/hccl/hcom_matmul_reduce_scatter.h"
#include "hccl/hccl_types.h"
#include "mindspore/ops/infer/ops_func_impl/matmul_reduce_scatter.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "runtime/rt.h"
#include "mindspore/core/include/utils/anf_utils.h"

namespace mindspore {
namespace kernel {
bool HcomMatMulReduceScatterKernel::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (outputs.size() != kMatMulReduceScatterOutputNum) {
    MS_LOG(EXCEPTION) << "Output number of MatMulReduceScatter should be 1, but got " << outputs.size();
  }

  std::vector<KernelTensor *> real_inputs(inputs.begin(), inputs.begin() + kSizeTwo);

  if (!HcclKernel::Init(real_inputs, outputs)) {
    MS_LOG(ERROR) << "Call HcclKernel::Init failed.";
    return false;
  }

#ifdef ENABLE_INTERNAL_KERNELS
  if (use_lccl_) {
    transpose_a_ = inputs[mindspore::ops::kMatmulReduceScatterInputTransInputIndex]->GetValueWithCheck<bool>();
    transpose_b_ = inputs[mindspore::ops::kMatmulReduceScatterInputTransX2Index]->GetValueWithCheck<bool>();

    if (GetHcclDataType() != HCCL_DATA_TYPE_FP16 && GetHcclDataType() != HCCL_DATA_TYPE_BFP16) {
      MS_LOG(EXCEPTION) << "MatMulReduceScatter only support data type fp16 or bf16.";
    }

    lcoc_dtype_ = (GetHcclDataType() == HCCL_DATA_TYPE_FP16) ? Lcal_ms::CoCDataTypeDesc::FP16FP16_FP32_FP16
                                                             : Lcal_ms::CoCDataTypeDesc::BF16BF16_FP32_BF16;

    // Dynamic load lcoc symbols.
    auto get_lcoc_func = DlsymFuncObj(CreateLcocForOp, lowlatency_comm_lib_handle_);
    lcoc_ptr_ = get_lcoc_func(group_);
    MS_EXCEPTION_IF_NULL(lcoc_ptr_);
    set_param_for_lcoc_func_ = DlsymFuncObj(SetParamForLcoc, lowlatency_comm_lib_handle_);
    MS_EXCEPTION_IF_NULL(set_param_for_lcoc_func_);
    get_lcoc_workspace_func_ = DlsymFuncObj(GetLcocWorkspaceSize, lowlatency_comm_lib_handle_);
    MS_EXCEPTION_IF_NULL(get_lcoc_workspace_func_);
    matmul_reduce_scatter_func_ = DlsymFuncObj(MatmulReduceScatter, lowlatency_comm_lib_handle_);
    MS_EXCEPTION_IF_NULL(matmul_reduce_scatter_func_);
  }
#endif
  return true;
}

int HcomMatMulReduceScatterKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  std::vector<KernelTensor *> real_inputs(inputs.begin(), inputs.begin() + kSizeTwo);
  int ret = HcclKernel::Resize(real_inputs, outputs);
  if (ret != KRET_OK) {
    MS_LOG(ERROR) << "Resize failed";
    return ret;
  }
#ifdef ENABLE_INTERNAL_KERNELS
  if (use_lccl_) {
    // The dimensions of left and right matrices.
    matmul_info_.m = transpose_a_ ? hccl_kernel_input_shape_list_[0][1] : hccl_kernel_input_shape_list_[0][0];
    matmul_info_.k = transpose_a_ ? hccl_kernel_input_shape_list_[0][0] : hccl_kernel_input_shape_list_[0][1];
    matmul_info_.n = transpose_b_ ? hccl_kernel_input_shape_list_[1][0] : hccl_kernel_input_shape_list_[1][1];
    matmul_info_.transA = transpose_a_;
    matmul_info_.transB = transpose_b_;

    MS_LOG(INFO) << matmul_info_.m << " " << matmul_info_.k << " " << matmul_info_.n << " " << matmul_info_.transA
                 << " " << matmul_info_.transB << " " << lcoc_dtype_ << " " << op_type_ << lcoc_type_;
    param_desc_.dataTypeDesc = lcoc_dtype_;
    param_desc_.mmInfo = matmul_info_;
    param_desc_.quantInfo = quant_info_;
    param_desc_.op = op_type_;
    set_param_for_lcoc_func_(lcoc_ptr_, lcoc_type_, tiling_, param_desc_);

    workspace_size_list_.clear();
    workspace_size_list_.push_back(get_lcoc_workspace_func_(lcoc_ptr_));
  }
#endif
  return KRET_OK;
}

bool HcomMatMulReduceScatterKernel::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "MatMulReduceScatter launch";
#ifdef ENABLE_INTERNAL_KERNELS
  if (use_lccl_) {
    if (outputs.empty() || workspace.empty() || hccl_data_type_list_.empty()) {
      MS_LOG(ERROR) << "Invalid ReduceScatter output or data type size (" << outputs.size() << ", " << workspace.size()
                    << ", " << hccl_data_type_list_.size() << ").";
      return false;
    }
    MS_EXCEPTION_IF_NULL(inputs[0]);
    MS_EXCEPTION_IF_NULL(inputs[1]);
    MS_EXCEPTION_IF_NULL(workspace[0]);
    MS_EXCEPTION_IF_NULL(outputs[0]);
    MS_EXCEPTION_IF_NULL(stream_ptr);

    Lcal_ms::CoCInputPkg coc_input_args = {
      inputs[0]->device_ptr(), inputs[1]->device_ptr(), nullptr, nullptr, nullptr, nullptr, nullptr};
    Lcal_ms::CoCOutputPkg coc_output_args = {outputs[0]->device_ptr(), nullptr};
    auto lccl_result =
      matmul_reduce_scatter_func_(lcoc_ptr_, coc_input_args, coc_output_args, workspace[0]->device_ptr(), stream_ptr);
    if (lccl_result != Lcal_ms::LCAL_SUCCESS) {
      MS_LOG(EXCEPTION) << "LCOC MatmulReduceScatter failed.";
    }
  }
#endif
  return true;
}
}  // namespace kernel
}  // namespace mindspore
