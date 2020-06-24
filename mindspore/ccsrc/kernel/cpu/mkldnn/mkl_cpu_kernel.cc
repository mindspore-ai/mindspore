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
#include "kernel/cpu/mkldnn/mkl_cpu_kernel.h"
#include <vector>
#include <string>
#include <algorithm>
#include "common/utils.h"
#include "kernel/cpu/mkldnn/mkl_kernel_engine.h"

namespace mindspore {
namespace kernel {
void MKLCPUKernel::GetPadding(const CNodePtr &kernel_node, const std::string &pad_mode,
                              const std::vector<size_t> &src_shape, int kernel_size, int stride,
                              std::vector<int> *padding_l, std::vector<int> *padding_r) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (src_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "set pad only support src dim >= 2!";
  }
  std::vector<int> weight_height;
  weight_height.emplace_back(src_shape[src_shape.size() - 2]);
  weight_height.emplace_back(src_shape[src_shape.size() - 1]);
  int rad = kernel_size / 2;
  int need_pad = kernel_size - 1;
  MS_LOG(INFO) << "pad mode " << pad_mode;
  if (pad_mode == PAD_MODE_LOWER_SAME || pad_mode == PAD_MODE_UPPER_SAME) {
    for (auto wh : weight_height) {
      int re = (wh - 1) % stride;
      int pad = std::max(rad - (re / 2), 0);
      padding_r->emplace_back(pad);
      pad = std::max(need_pad - pad - re, 0);
      padding_l->emplace_back(pad);
    }
  } else if (pad_mode == PAD_MODE_LOWER_VALID || pad_mode == PAD_MODE_UPPER_VALID) {
    MS_LOG(INFO) << "pad valid";
    padding_l->emplace_back(0);
    padding_l->emplace_back(0);
    padding_r->emplace_back(0);
    padding_r->emplace_back(0);
  } else {
    std::vector<int> pad = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, PAD);
    if (pad.size() != 4) {
      MS_LOG(EXCEPTION) << "wrong pad size in max pooling " << pad.size();
    }
    padding_l->emplace_back(pad[0]);
    padding_l->emplace_back(pad[1]);
    padding_r->emplace_back(pad[2]);
    padding_r->emplace_back(pad[3]);
  }
}

dnnl::memory::format_tag MKLCPUKernel::GetDefaultFormatTag(const dnnl::memory::dims &dims) const {
  dnnl::memory::format_tag mem_tag;
  auto dim_size = dims.size();
  if (dim_size == 4) {
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
  dims.insert(dims.end(), shape.begin(), shape.end());
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
