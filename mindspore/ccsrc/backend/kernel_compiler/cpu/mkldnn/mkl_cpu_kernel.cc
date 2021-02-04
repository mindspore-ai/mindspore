/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/mkldnn/mkl_cpu_kernel.h"
#include <vector>
#include <string>
#include <algorithm>
#include "utils/ms_utils.h"
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"

namespace mindspore {
namespace kernel {
void MKLCPUKernel::GetPadding(const CNodePtr &kernel_node, const std::string &pad_mode,
                              const std::vector<size_t> &src_shape, const std::vector<size_t> &kernel_size,
                              const std::vector<int> &stride, std::vector<int> *padding_l, std::vector<int> *padding_r,
                              const std::vector<int> &dilation) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (src_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "set pad only support src dim >= 2!";
  }
  std::vector<int> weight_height;
  weight_height.emplace_back(src_shape[src_shape.size() - 2]);
  weight_height.emplace_back(src_shape[src_shape.size() - 1]);

  MS_LOG(INFO) << "pad mode: " << pad_mode;
  if (pad_mode == PAD_MODE_LOWER_SAME || pad_mode == PAD_MODE_UPPER_SAME) {
    for (size_t i = 0; i < weight_height.size(); ++i) {
      auto wh = weight_height[i];
      int out = (wh + stride[i] - 1) / stride[i];
      int effective_k = (SizeToInt(kernel_size[i]) - 1) * dilation[i] + 1;
      int pad_along = std::max(0, (out - 1) * stride[i] + effective_k - wh);
      int pad = pad_along / 2;
      padding_l->emplace_back(pad);
      padding_r->emplace_back(pad_along - pad);
    }
  } else if (pad_mode == PAD_MODE_LOWER_VALID || pad_mode == PAD_MODE_UPPER_VALID) {
    MS_LOG(INFO) << "pad valid";
    padding_l->emplace_back(0);
    padding_l->emplace_back(0);
    padding_r->emplace_back(0);
    padding_r->emplace_back(0);
  } else {
    std::vector<int> pad;
    std::vector<int64_t> pad_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, PAD_LIST);
    (void)std::transform(pad_me.begin(), pad_me.end(), std::back_inserter(pad),
                         [](const int64_t &value) { return static_cast<int>(value); });
    padding_l->emplace_back(pad[0]);
    padding_l->emplace_back(pad[2]);
    padding_r->emplace_back(pad[1]);
    padding_r->emplace_back(pad[3]);
  }
}

bool MKLCPUKernel::BinaryBroadCast(std::vector<size_t> *src0_shape, std::vector<size_t> *src1_shape,
                                   std::vector<size_t> *dst_shape) {
  MS_EXCEPTION_IF_NULL(src0_shape);
  MS_EXCEPTION_IF_NULL(src1_shape);
  MS_EXCEPTION_IF_NULL(dst_shape);
  bool need_swap = false;
  if (dst_shape->size() == 0) {
    dst_shape->emplace_back(1);
    src0_shape->emplace_back(1);
    src1_shape->emplace_back(1);
  }
  MS_LOG(DEBUG) << "Binary broadcast in: src0: " << *src0_shape << " src1: " << *src1_shape << " dst: " << *dst_shape;
  if (src0_shape->size() != dst_shape->size()) {
    need_swap = true;
    for (size_t i = src0_shape->size(); i < dst_shape->size(); ++i) {
      src0_shape->insert(src0_shape->begin(), 1);
    }
  } else if (src1_shape->size() != dst_shape->size()) {
    for (size_t i = src1_shape->size(); i < dst_shape->size(); ++i) {
      src1_shape->insert(src1_shape->begin(), 1);
    }
  }
  if (src0_shape->size() == src1_shape->size()) {
    bool visit_src0 = false;
    bool visit_src1 = false;
    for (size_t i = 0; i < src0_shape->size(); ++i) {
      if (src0_shape->at(i) != src1_shape->at(i)) {
        if (src0_shape->at(i) == 1 && !visit_src1) {
          need_swap = true;
          visit_src0 = true;
        } else if (src1_shape->at(i) == 1 && !visit_src0) {
          need_swap = false;
          visit_src1 = true;
        } else {
          MS_LOG(EXCEPTION) << "Invalid broadcast! " << *src0_shape << " vs " << *src1_shape;
        }
      }
    }
  } else {
    MS_LOG(EXCEPTION) << "Invalid broadcast! src0: " << *src0_shape << " src1: " << *src1_shape
                      << " dst: " << *dst_shape;
  }
  MS_LOG(DEBUG) << "Binary broadcast out: src0: " << *src0_shape << " src1: " << *src1_shape << " dst: " << *dst_shape;
  return need_swap;
}

dnnl::memory::format_tag MKLCPUKernel::GetDefaultFormatTag(const dnnl::memory::dims &dims) const {
  dnnl::memory::format_tag mem_tag;
  auto dim_size = dims.size();
  if (dim_size == 5) {
    mem_tag = dnnl::memory::format_tag::abcde;
  } else if (dim_size == 4) {
    mem_tag = dnnl::memory::format_tag::abcd;
  } else if (dim_size == 3) {
    mem_tag = dnnl::memory::format_tag::abc;
  } else if (dim_size == 2) {
    mem_tag = dnnl::memory::format_tag::ab;
  } else if (dim_size == 1) {
    mem_tag = dnnl::memory::format_tag::a;
  } else {
    MS_LOG(EXCEPTION) << "kernel dims invalid " << dim_size;
  }
  return mem_tag;
}

dnnl::memory::desc MKLCPUKernel::GetDefaultMemDesc(const std::vector<size_t> &shape) {
  dnnl::memory::dims dims;
  if (shape.size() == 0) {
    dims.insert(dims.end(), 1);
  } else {
    dims.insert(dims.end(), shape.begin(), shape.end());
  }
  dnnl::memory::format_tag mem_tag = GetDefaultFormatTag(dims);
  dnnl::memory::desc mem_desc(dims, dnnl::memory::data_type::f32, mem_tag);
  return mem_desc;
}

void MKLCPUKernel::AddArgument(int arg_key, const dnnl::memory::desc &mem_desc, bool alloc) {
  arguments_[arg_key] = MKLKernelEngine::Get().CreateMemory(mem_desc, alloc);
}

void MKLCPUKernel::SetArgumentHandle(int arg_key, void *ptr) {
  auto arg_iter = arguments_.find(arg_key);
  if (arg_iter != arguments_.end()) {
    arg_iter->second.set_data_handle(ptr);
  }
}

void MKLCPUKernel::ExecutePrimitive() { MKLKernelEngine::Get().Execute(primitive_, arguments_); }

void MKLCPUKernel::Reorder(dnnl::memory *src_mem, dnnl::memory *dst_mem) {
  MKLKernelEngine::Get().Reorder(src_mem, dst_mem);
}
}  // namespace kernel
}  // namespace mindspore
