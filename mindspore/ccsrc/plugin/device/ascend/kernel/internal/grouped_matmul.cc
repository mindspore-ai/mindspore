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

#include "plugin/device/ascend/kernel/internal/grouped_matmul.h"
#include <memory>
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "param/grouped_matmul_param.h"

namespace {
constexpr const size_t kAttrInputNum = 3;
constexpr const size_t kListNum = 7;
constexpr const size_t kMaxGroupNum = 128;
}  // namespace

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalGroupedMatmul::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                          const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::GroupedMatmul;

  size_t input_num = inputs.size();
  const size_t kMinInputNum = 10;
  if (input_num < kMinInputNum) {
    MS_LOG(ERROR) << "GroupedMatmul op input number is invalid, minimum input num is 10, but get " << input_num;
    return nullptr;
  }

  const size_t kTranspose_weight = input_num - 1;
  const size_t kDtype = input_num - 2;
  const size_t kSplitItem = input_num - 3;

  split_item_ = static_cast<int32_t>(inputs[kSplitItem]->GetValueWithCheck<int64_t>());
  int dtype = static_cast<int32_t>(inputs[kDtype]->GetValueWithCheck<int64_t>());
  bool transpose_weight = inputs[kTranspose_weight]->GetValueWithCheck<bool>();

  if (split_item_ != 0 && split_item_ != 1 && split_item_ != 2 && split_item_ != 3) {
    MS_LOG(ERROR) << "GroupedMatmul split_item only support 0, 1, 2 or 3, but get " << split_item_;
    return nullptr;
  }

  real_input_num_ = input_num - kAttrInputNum;

  MS_LOG(INFO) << "GroupedMatmul split_item: " << split_item_ << ", dtype: " << dtype;

  if (split_item_ == 1 || split_item_ == 3) {
    // In this case, x need grouped_list to split.
    const size_t kGroupedList = input_num - 4;
    auto group_list = inputs[kGroupedList]->GetValueWithCheck<std::vector<int64_t>>();
    MS_LOG(INFO) << "GroupedMatmul grouped_list: " << group_list;
    group_num_.resize(group_list.size());
    std::adjacent_difference(group_list.begin(), group_list.end(), group_num_.begin());
    dyn_input_sizes_[0] = group_num_.size();
    dyn_input_sizes_[1] = group_num_.size();
    // TODO if bias is need
    MS_LOG(ERROR) << "split_item_ == 1 || split_item_ == 3, GroupedMatmul dyn_input_sizes_: " << dyn_input_sizes_;
  } else if (split_item_ == 2) {
    group_num_.clear();
    for (size_t i = 0; i < static_cast<size_t>(dyn_input_sizes_[0]); ++i) {
      group_num_.push_back(inputs[i]->GetShapeVector()[0]);
    }
  }

  MS_LOG(INFO) << "group_num_: " << group_num_;

  internal::GroupedMatmulParam op_param;
  op_param.split_item = split_item_;
  op_param.dtype = dtype;
  op_param.transpose_weight = transpose_weight;
  op_param.group_num = static_cast<int32_t>(dyn_input_sizes_[0]);

  param_ptr->specificParam = op_param;
  return param_ptr;
}

void InternalGroupedMatmul::SetInOutIdx() {}

void InternalGroupedMatmul::SetTilingInfo(const uint64_t key) {
  size_t tiling_size = impl_->GetTilingBufSize();
  auto tiling_func = [this](internal::HostRawBuf &host_buf, internal::CacheInfo &cache_info) {
    auto ret = this->impl_->Tiling(host_buf);
    cache_info = this->impl_->GetCacheInfo();
    return ret;
  };
  tiling_info_ = TilingCacheMgr::GetInstance().GetOrCreateTilingInfo(key, tiling_func, tiling_size);
  impl_->SetCacheInfo(tiling_info_.cache_info_);
  impl_->SetDeviceTilingBuf(tiling_info_.device_buf_);
}

int InternalGroupedMatmul::Build(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  MS_LOG(INFO) << "input num: " << inputs.size() << ", output num: " << outputs.size();
  auto param = CreateOpParam(inputs, outputs);
  if (param == nullptr) {
    MS_LOG(ERROR) << "Create param failed.";
    return -1;
  }
  if (impl_ == nullptr) {
    impl_ = internal::CreateInternalKernelImpl(param);
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "Create internal kernel impl failed.";
      return -1;
    }
  }

  return 0;
}

bool InternalGroupedMatmul::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  // Alloc max inputs and outputs space
  // one is the optional input, the last one is the tensor which keep the dyn_list info.
  inputs_.resize(kMaxGroupNum * kListNum + 2);
  std::generate(inputs_.begin(), inputs_.end(), []() { return new internal::Tensor(); });

  outputs_.resize(kMaxGroupNum);
  std::generate(outputs_.begin(), outputs_.end(), []() { return new internal::Tensor(); });

  tiling_info_.device_buf_.size_ = 0;
  tiling_info_.device_buf_.addr_ = nullptr;

  input_host_list_ptr_.resize(kMaxGroupNum * kListNum);
  output_host_list_ptr_.resize(kMaxGroupNum);
  return true;
}

int InternalGroupedMatmul::CheckIntegratedInputShape(const std::vector<int64_t> &shape,
                                                     const std::vector<int64_t> &group_list) {
  if (shape.size() != kDim2) {
    MS_LOG(ERROR) << "Only support split kernel tensor which rank is 2.";
    return -1;
  }
  auto bs = std::accumulate(group_list.begin(), group_list.end(), 0);
  if (shape[0] != bs) {
    MS_LOG(ERROR) << "The first dimension of internal tensor should be equal to the sum of group_list, but got "
                  << "shape[0]: " << shape[0] << "bs: " << bs;
    return -1;
  }
  return 0;
}

int InternalGroupedMatmul::CheckIntegratedWeightShape(const std::vector<int64_t> &shape,
                                                      const std::vector<int64_t> &group_list) {
  if (shape.size() != 3) {
    MS_LOG(ERROR) << "When split_item is 1 or 3, shape.size() should be 3, but got: " << shape.size();
    return -1;
  }
  if (shape[0] != static_cast<int64_t>(group_list.size())) {
    MS_LOG(ERROR) << "When split_item is 1 or 3, shape[0] should be group_list.size(), but got: " << shape[0]
                  << ", group_list.size(): " << group_list.size();
    return -1;
  }
  return 0;
}

int InternalGroupedMatmul::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  MS_LOG(INFO) << "inputs size is " << inputs.size() << ", outputs size is " << outputs.size();
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " invoke resize failed";
    return KRET_RESIZE_FAILED;
  }

  if (primitive_->GetAttr("group_info") == nullptr) {
    MS_LOG(ERROR) << "Get GroupedMatmul group_info attr failed";
    return KRET_RESIZE_FAILED;
  }
  MS_LOG(INFO) << "primitive_ is: " << primitive_->ToString();
  dyn_input_sizes_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr("group_info"));

  ret = Build(inputs, outputs);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " build kernel failed";
    return KRET_RESIZE_FAILED;
  }

  MS_LOG(INFO) << "split_item is: " << split_item_;
  if (split_item_ == 0 || split_item_ == 2) {
    for (size_t i = 0; i < real_input_num_ + 1; i++) {
      InternalKernelUtils::ToInternalTensor(inputs_[i], inputs[i]);
    }
  } else {
    // convert input
    if (CheckIntegratedInputShape(inputs[0]->GetShapeVector(), group_num_) != 0) {
      return KRET_RESIZE_FAILED;
    }
    size_t offset = 0;
    for (size_t i = 0; i < group_num_.size(); i++) {
      InternalKernelUtils::ToInternalTensor(inputs_[i], inputs[0]);
      inputs_[i]->desc.dims[0] = group_num_.at(i);
    }
    offset += group_num_.size();
    // convert weight
    auto shape = inputs[1]->GetShapeVector();
    if (CheckIntegratedWeightShape(shape, group_num_) != 0) {
      return KRET_RESIZE_FAILED;
    }
    std::vector<int64_t> split_weight_shape = {shape[1], shape[2]};
    for (size_t i = 0; i < group_num_.size(); i++) {
      InternalKernelUtils::ToInternalTensor(inputs_[i + offset], inputs[1]);
      inputs_[i + offset]->desc.dims = internal::VecToSVec<int64_t>(split_weight_shape);
    }
    // convert bias
    offset += group_num_.size();
    for (size_t i = 0; i < group_num_.size(); i++) {
      InternalKernelUtils::ToInternalTensor(inputs_[i + offset], inputs[2]);
    }
  }

  // set group_list info
  inputs_[inputs_.size() - 1]->desc.dims = internal::VecToSVec<int64_t>(dyn_input_sizes_);

  impl_->SetInputs(inputs_);

  if (split_item_ == 0 || split_item_ == 1) {
    for (int i = 0; i < dyn_input_sizes_[0]; i++) {
      InternalKernelUtils::ToInternalTensor(outputs_[i], outputs[i]);
    }
  } else {
    for (size_t i = 0; i < group_num_.size(); i++) {
      InternalKernelUtils::ToInternalTensor(outputs_[i], outputs[0]);
      outputs_[i]->desc.dims[0] = group_num_.at(i);
    }
  }

  impl_->SetOutputs(outputs_);
  // auto key = GenTilingCacheKey(inputs, outputs);
  // we use key == 0, which means not cache the tiling info.
  SetTilingInfo(0);

  workspace_size_list_.clear();
  workspace_size_list_.resize(kListNum + 2);
  // [0] is the workspace the kernel need.
  workspace_size_list_[0] = impl_->GetWorkSpaceSize()[0];
  // [1] to [7] is the input tensorlist addr, max 128 expert. [8] is the output tensorlist.
  for (size_t i = 1; i < workspace_size_list_.size(); ++i) {
    workspace_size_list_[i] = static_cast<size_t>(sizeof(uint64_t *) * kMaxGroupNum);
    MS_LOG(INFO) << "workspace [" << i << "]: " << workspace_size_list_[i];
  }

  return KRET_OK;
}

void InternalGroupedMatmul::SetHostList(const KernelTensor *kernel_tensor, const std::vector<int64_t> &group_num,
                                        const size_t &list_offset, std::vector<void *> &host_list_ptr) {
  auto x_shape = kernel_tensor->GetShapeVector();
  auto x_mem_size = kernel_tensor->size();
  auto x0_stride = x_mem_size / x_shape[0];
  int x0_idx = 0;
  for (size_t i = 0; i < group_num.size(); i++) {
    host_list_ptr[i + list_offset] = (uint8_t *)(kernel_tensor->device_ptr()) + x0_stride * x0_idx;
    x0_idx += group_num[i];
  }
  return;
}

int InternalGroupedMatmul::PrepareInOutTensors(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto ret = 0;
  // create input tensorlist
  if (split_item_ == 0 || split_item_ == 2) {
    int offset = 0;
    for (size_t i = 0; i < dyn_input_sizes_.size(); i++) {
      MS_LOG(INFO) << "dyn_input_sizes_[" << i << "] = " << dyn_input_sizes_[i];
      for (int j = 0; j < static_cast<int>(dyn_input_sizes_[i]); j++) {
        input_host_list_ptr_[i * kMaxGroupNum + j] = inputs[offset + j]->device_ptr();
      }
      offset += dyn_input_sizes_[i];
      ret = aclrtMemcpyAsync(workspace[i + 1]->device_ptr(), workspace[i + 1]->size(),
                             input_host_list_ptr_.data() + i * kMaxGroupNum, kMaxGroupNum * sizeof(void *),
                             ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
      if (ret != 0) {
        MS_LOG(ERROR) << "aclrtMemcpyAsync failed!";
        return ret;
      }
    }
  } else {
    // input
    SetHostList(inputs[0], group_num_, 0, input_host_list_ptr_);
    ret = aclrtMemcpyAsync(workspace[1]->device_ptr(), workspace[1]->size(), input_host_list_ptr_.data(),
                           kMaxGroupNum * sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
    if (ret != 0) {
      MS_LOG(ERROR) << "aclrtMemcpyAsync failed!";
      return ret;
    }
    // weight
    std::vector<int64_t> w_group_num(group_num_.size(), 1);
    SetHostList(inputs[1], w_group_num, kMaxGroupNum, input_host_list_ptr_);
    ret = aclrtMemcpyAsync(workspace[2]->device_ptr(), workspace[2]->size(), input_host_list_ptr_.data() + kMaxGroupNum,
                           kMaxGroupNum * sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
    if (ret != 0) {
      MS_LOG(ERROR) << "aclrtMemcpyAsync failed!";
      return ret;
    }
    // bias
    SetHostList(inputs[2], w_group_num, kMaxGroupNum * 2, input_host_list_ptr_);
    ret =
      aclrtMemcpyAsync(workspace[3]->device_ptr(), workspace[3]->size(), input_host_list_ptr_.data() + kMaxGroupNum * 2,
                       kMaxGroupNum * sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
    if (ret != 0) {
      MS_LOG(ERROR) << "aclrtMemcpyAsync failed!";
      return ret;
    }
  }

  // create output tensorlist
  if (split_item_ == 0 || split_item_ == 1) {
    for (int i = 0; i < static_cast<int>(dyn_input_sizes_[0]); i++) {
      output_host_list_ptr_[i] = outputs[i]->device_ptr();
    }
  } else {
    // output host list
    SetHostList(outputs[0], group_num_, 0, output_host_list_ptr_);
  }
  size_t ws_size = workspace.size();
  ret =
    aclrtMemcpyAsync(workspace[ws_size - 1]->device_ptr(), workspace[ws_size - 1]->size(), output_host_list_ptr_.data(),
                     output_host_list_ptr_.size() * sizeof(void *), ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (ret != 0) {
    MS_LOG(ERROR) << "aclrtMemcpyAsync failed!";
    return ret;
  }
  return ret;
}

bool InternalGroupedMatmul::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto ret = PrepareInOutTensors(inputs, workspace, outputs, stream_ptr);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " prepare inout tensors failed";
    return false;
  }

  impl_->SetStream(stream_ptr);
  std::vector<internal::DeviceRawBuf> ws_raw_bufs(workspace.size());
  for (size_t i = 0; i < workspace.size(); ++i) {
    ws_raw_bufs[i] = InternalKernelUtils::ToDeviceRawBuf(workspace[i]);
  }

  impl_->SetWorkSpace(ws_raw_bufs);

  ret = impl_->Launch();

  return (ret == 0);
}

uint64_t InternalGroupedMatmul::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  // TODO fix this
  //  User defined CacheKey, the inputs should include all the factors which will affect tiling result.

  return TilingCacheMgr::GetInstance().GenTilingCacheKey(
    kernel_name_, inputs.size(), outputs.size(), inputs[kIndex0]->GetShapeVector(), inputs[kIndex0]->dtype_id(),
    inputs[kIndex1]->GetShapeVector(), inputs[kIndex1]->dtype_id(), inputs[kIndex2]->GetShapeVector(),
    inputs[kIndex2]->dtype_id(), outputs[kIndex0]->GetShapeVector(), outputs[kIndex0]->dtype_id(), dyn_input_sizes_,
    inputs[kIndex0]->device_ptr(), inputs[kIndex0]->device_id(), inputs[kIndex1]->device_id());
}

MS_INTERNAL_KERNEL_FACTORY_REG(GroupedMatmul, InternalGroupedMatmul);
}  // namespace kernel
}  // namespace mindspore
