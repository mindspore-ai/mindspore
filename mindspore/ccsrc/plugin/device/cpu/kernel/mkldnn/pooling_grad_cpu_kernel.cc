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

#include "plugin/device/cpu/kernel/mkldnn/pooling_grad_cpu_kernel.h"

#include <algorithm>
#include <functional>
#include <unordered_map>

#include "utils/ms_utils.h"
#include "utils/profile.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kAvgPooling3DGradInputsNum = 1;
constexpr size_t kPoolingGradInputsNum = 3;
constexpr size_t kPoolingGradOutputsNum = 1;
constexpr size_t kPoolingGradWorkSpaceNum = 2;
constexpr size_t kGradIndex = 2;
}  // namespace

void PoolingGradCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  DeprecatedNativeCpuKernelMod::InitInputOutputSize(kernel_node);
  if (algorithm_ == dnnl::algorithm::pooling_max) {
    size_t work_space = GetSize(workspace_desc_);
    size_t dst_space =
      std::accumulate(dst_shape_.begin(), dst_shape_.end(), size_t(1), std::multiplies<size_t>()) * sizeof(float);
    workspace_size_list_.push_back(work_space);
    workspace_size_list_.push_back(dst_space);
  }
}

void PoolingGradCpuKernelMod::InitPoolingGradFields(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  PrimitivePtr prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr(CEIL_MODE)) {
    ValuePtr ceil_mode = prim->GetAttr(CEIL_MODE);
    ceil_mode_ = (ceil_mode->isa<BoolImm>() && GetValue<bool>(ceil_mode)) ||
                 (ceil_mode->isa<Int64Imm>() && GetValue<int64_t>(ceil_mode) == 1);
  }
  if (kernel_name_ == kAvgPoolGradOpName || kernel_name_ == kAvgPool3DGradOpName) {
    algorithm_ = dnnl::algorithm::pooling_avg;
    if (prim->HasAttr(COUNT_INCLUDE_PAD) && GetValue<bool>(prim->GetAttr(COUNT_INCLUDE_PAD))) {
      algorithm_ = dnnl::algorithm::pooling_avg_include_padding;
    }
    if (prim->HasAttr(DIVISOR_OVERRIDE) && GetValue<int64_t>(prim->GetAttr(DIVISOR_OVERRIDE)) != 0) {
      divisor_override_ = GetValue<int64_t>(prim->GetAttr(DIVISOR_OVERRIDE));
    }
  }
  grad_index_ = kernel_name_ == kAvgPool3DGradOpName ? 0 : kGradIndex;
  dst_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, grad_index_);
}

void PoolingGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  InitPoolingGradFields(kernel_node);
  std::vector<size_t> src_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  const size_t src_dim = src_shape.size();
  if (src_dim != SHAPE_4D && src_dim != SHAPE_5D) {
    MS_LOG(EXCEPTION) << "PoolingGrad only supports 4D/5D input, but got " << src_dim << "D";
  }
  src_desc_ = GetDefaultMemDesc(src_shape);
  dst_desc_ = GetDefaultMemDesc(dst_shape_);
  const auto format = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, FORMAT);
  if (src_dim == SHAPE_4D && format != NCHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 4D input with NCHW format, but got format " << format;
  }
  if (src_dim == SHAPE_5D && format != NCDHW) {
    MS_LOG(EXCEPTION) << kernel_name_ << " only supports 5D input with NCDHW format, but got format" << format;
  }
  const auto pad_mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, PAD_MODE);
  const auto kernel_include_nc = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, KERNEL_SIZE);
  const auto strides_include_nc = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
  if (kernel_include_nc.size() != src_dim) {
    MS_LOG(EXCEPTION) << kernel_name_ << " requires kernel_size should be " << src_dim << "D, but got "
                      << kernel_include_nc.size() << "D!";
  }
  if (strides_include_nc.size() != src_dim) {
    MS_LOG(EXCEPTION) << kernel_name_ << " requires strides should be " << src_dim << "D, but got "
                      << strides_include_nc.size() << "D!";
  }
  const dnnl::memory::dims kernel(kernel_include_nc.begin() + NC_LEN, kernel_include_nc.end());
  const dnnl::memory::dims strides(strides_include_nc.begin() + NC_LEN, strides_include_nc.end());
  const dnnl::memory::dims dilation(kernel.size(), kPoolingDilation);
  dnnl::memory::dims padding_l;
  dnnl::memory::dims padding_r;
  kernel_ = kernel;
  PaddingInfo padding_info{pad_mode, kernel, strides, dilation, &padding_l, &padding_r, &padding_invalid_, ceil_mode_};
  GetPadding(kernel_node, src_shape, padding_info);

  // Pooling_avg forward description
  const auto desc = CreateDesc<dnnl::pooling_forward::desc>(dnnl::prop_kind::forward_training, algorithm_, src_desc_,
                                                            dst_desc_, strides, kernel, padding_l, padding_r);
  auto forward_prim_desc = CreateDesc<dnnl::pooling_forward::primitive_desc>(desc, engine_);

  // Pooling_avg backward description
  const auto backward_desc =
    CreateDesc<dnnl::pooling_backward::desc>(algorithm_, src_desc_, dst_desc_, strides, kernel, padding_l, padding_r);
  const auto backward_prim_desc =
    CreateDesc<dnnl::pooling_backward::primitive_desc>(backward_desc, engine_, forward_prim_desc);
  primitive_ = CreatePrimitive<dnnl::pooling_backward>(backward_prim_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, src_desc_);
  AddArgument(DNNL_ARG_DIFF_DST, dst_desc_);

  // For pooling_max, need a workspace that generated in forward and stored the max value indexes to compute grad.
  if (algorithm_ == dnnl::algorithm::pooling_max) {
    primitive_forward_ = CreatePrimitive<dnnl::pooling_forward>(forward_prim_desc);
    workspace_desc_ = GetWorkspaceDesc(forward_prim_desc);
    AddArgument(DNNL_ARG_WORKSPACE, workspace_desc_);
  }
}

#ifdef USE_MS_THREADPOOL_FOR_DNNL
void PoolingGradCpuKernelMod::ExecuteForwardByMSThreadPool(const std::unordered_map<int, dnnl::memory> &arguments) {
  const size_t MAX_POW = 6;
  const size_t AVG_COUNT = 5;
  const size_t DIFF = 2;
  size_t current_pow = forward_parallel_info_.search_count / AVG_COUNT;
  int current_thread_nums = static_cast<int>(std::pow(2.0f, current_pow));
  auto mkl_pool = dynamic_cast<mkl_threadpool *>(mkl_threadpool_.get());
  if (current_pow >= MAX_POW) {
    int best_thread_nums = static_cast<int>(std::pow(2.0f, forward_parallel_info_.best_pow));
    mkl_pool->set_num_threads(best_thread_nums);
    MS_LOG(DEBUG) << "begin to invoke primitive::execute";
    primitive_forward_->execute(stream_, arguments);
    MS_LOG(DEBUG) << "end to invoke primitive::execute";
    return;
  }

  if (forward_parallel_info_.search_count % AVG_COUNT == 0) {
    forward_parallel_info_.tmp_sum_cost_time = 0;
  }
  double start_time = GetTime();
  mkl_pool->set_num_threads(current_thread_nums);
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
  primitive_forward_->execute(stream_, arguments);
  MS_LOG(DEBUG) << "end to invoke primitive::execute";
  double cost_time = GetTime() - start_time;
  forward_parallel_info_.tmp_sum_cost_time += cost_time;
  forward_parallel_info_.search_count++;
  if (forward_parallel_info_.search_count % AVG_COUNT == 0) {
    if (forward_parallel_info_.min_cost_time > forward_parallel_info_.tmp_sum_cost_time) {
      forward_parallel_info_.min_cost_time = forward_parallel_info_.tmp_sum_cost_time;
      forward_parallel_info_.best_pow = current_pow;
    } else if (current_pow - forward_parallel_info_.best_pow >= DIFF) {
      forward_parallel_info_.search_count = AVG_COUNT * MAX_POW;
    }
  }
}
#endif

void PoolingGradCpuKernelMod::ComputeMaxValueIndex(void *src, void *dst, void *work_array) {
  // Compute maxvalue index for pooling_backward_max.
  MS_LOG(INFO) << "Compute maxvalue index for " << kernel_name_;
  std::unordered_map<int, dnnl::memory> arguments;
  dnnl::memory src_mem = dnnl::memory(src_desc_, engine_, nullptr);
  dnnl::memory dst_mem = dnnl::memory(dst_desc_, engine_, nullptr);
  dnnl::memory work_mem = dnnl::memory(workspace_desc_, engine_, nullptr);
  src_mem.set_data_handle(src);
  dst_mem.set_data_handle(dst);
  work_mem.set_data_handle(work_array);
  arguments[DNNL_ARG_SRC] = src_mem;
  arguments[DNNL_ARG_DST] = dst_mem;
  arguments[DNNL_ARG_WORKSPACE] = work_mem;

#ifdef USE_MS_THREADPOOL_FOR_DNNL
  ExecuteForwardByMSThreadPool(arguments);
#else
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
  primitive_forward_->execute(stream_, arguments);
  MS_LOG(DEBUG) << "end to invoke primitive::execute";
#endif
  (void)stream_.wait();
}

bool PoolingGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &workspace,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  size_t input_num = kernel_name_ == kAvgPool3DGradOpName ? kAvgPooling3DGradInputsNum : kPoolingGradInputsNum;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPoolingGradOutputsNum, kernel_name_);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[grad_index_]->addr);

  // For pooling_max, get the workspace that store the max value indexes.
  if (algorithm_ == dnnl::algorithm::pooling_max) {
    CHECK_KERNEL_WORKSPACE_SIZE(workspace.size(), kPoolingGradWorkSpaceNum, kernel_name_);
    ComputeMaxValueIndex(inputs[0]->addr, workspace[1]->addr, workspace[0]->addr);
    SetArgumentHandle(DNNL_ARG_WORKSPACE, workspace[0]->addr);
    ExecutePrimitive();
    return true;
  }

  float *dst = reinterpret_cast<float *>(inputs[grad_index_]->addr);
  if (divisor_override_ != 0) {
    ReComputeDivisor(dst);
  } else {
    bool has_invalid_padding = std::any_of(padding_invalid_.begin(), padding_invalid_.end(),
                                           [](const int64_t &padding) { return padding != 0; });
    if (algorithm_ == dnnl::algorithm::pooling_avg_include_padding && has_invalid_padding) {
      EliminateInvalidPadding(dst);
    }
  }
  ExecutePrimitive();
  return true;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AvgPoolGrad,
                                 []() { return std::make_shared<PoolingGradCpuKernelMod>(kAvgPoolGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AvgPool3DGrad,
                                 []() { return std::make_shared<PoolingGradCpuKernelMod>(kAvgPool3DGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, MaxPoolGrad,
                                 []() { return std::make_shared<PoolingGradCpuKernelMod>(kMaxPoolGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, MaxPool3DGrad,
                                 []() { return std::make_shared<PoolingGradCpuKernelMod>(kMaxPool3DGrad); });
}  // namespace kernel
}  // namespace mindspore
