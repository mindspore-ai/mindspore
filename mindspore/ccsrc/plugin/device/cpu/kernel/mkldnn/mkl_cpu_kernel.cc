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

#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include "utils/profile.h"

namespace mindspore {
namespace kernel {
namespace {
void GeneratePaddingForPadMode(const PaddingInfo &padding_info, std::vector<int64_t> shape_exclude_nc,
                               std::vector<int64_t> pad) {
  if (padding_info.ceil_mode) {
    MS_EXCEPTION_IF_NULL(padding_info.padding_invalid);
  }
  const size_t multiple = 2;
  const size_t dim = shape_exclude_nc.size();
  if (pad.size() != dim * multiple) {
    MS_LOG(EXCEPTION) << "pad list must be " << (dim * multiple) << "D, but got " << pad.size() << "D!";
  }
  for (size_t i = 0; i < dim; ++i) {
    size_t l_index = multiple * i;
    size_t r_index = multiple * i + 1;
    padding_info.padding_l->push_back(pad[l_index]);

    if (padding_info.ceil_mode) {
      int64_t len = shape_exclude_nc[i] + pad[l_index] + pad[r_index] - padding_info.kernel_size[i];
      int64_t padding_iv =
        FloatToLong(std::ceil(LongToFloat(len) / LongToFloat(padding_info.stride[i]))) * padding_info.stride[i] - len;
      int64_t padding_r = pad[r_index] + padding_iv;
      if (padding_r > pad[r_index] && padding_r < padding_info.kernel_size[i]) {
        padding_info.padding_r->push_back(padding_r);
        padding_info.padding_invalid->push_back(padding_iv);
        continue;
      }
      padding_info.padding_invalid->push_back(0);
    }
    padding_info.padding_r->push_back(pad[r_index]);
  }
}
}  // namespace

void DeprecatedMKLCpuKernelMod::GetPadding(const CNodePtr &kernel_node, const std::vector<int64_t> &src_shape,
                                           const PaddingInfo &padding_info) const {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(padding_info.padding_l);
  MS_EXCEPTION_IF_NULL(padding_info.padding_r);
  size_t src_dim = src_shape.size();
  if (src_dim < NC_LEN) {
    MS_LOG(EXCEPTION) << "Set pad only support src dim >= 2!";
  }
  const size_t dim_exclude_nc = src_dim - NC_LEN;
  std::vector<int64_t> shape_exclude_nc;
  for (size_t i = NC_LEN; i < src_dim; ++i) {
    shape_exclude_nc.push_back(src_shape[i]);
  }

  if (padding_info.pad_mode == PAD_MODE_LOWER_SAME || padding_info.pad_mode == PAD_MODE_UPPER_SAME) {
    for (size_t i = 0; i < dim_exclude_nc; ++i) {
      int64_t wh = shape_exclude_nc[i];
      int64_t out = (wh + padding_info.stride[i] - 1) / padding_info.stride[i];
      int64_t effective_k = (SizeToLong(padding_info.kernel_size[i]) - 1) * padding_info.dilation[i] + 1;
      int64_t pad_along = std::max(int64_t(0), (out - 1) * padding_info.stride[i] + effective_k - wh);
      int64_t pad = pad_along / 2;
      padding_info.padding_l->push_back(pad);
      padding_info.padding_r->push_back(pad_along - pad);
    }
  } else if (padding_info.pad_mode == PAD_MODE_LOWER_VALID || padding_info.pad_mode == PAD_MODE_UPPER_VALID) {
    for (size_t i = 0; i < dim_exclude_nc; ++i) {
      padding_info.padding_l->push_back(0);
      padding_info.padding_r->push_back(0);
    }
  } else {
    std::vector<int64_t> pad = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, PAD_LIST);
    GeneratePaddingForPadMode(padding_info, shape_exclude_nc, pad);
  }
}

bool DeprecatedMKLCpuKernelMod::BinaryBroadCast(std::vector<size_t> *src0_shape, std::vector<size_t> *src1_shape,
                                                std::vector<size_t> *dst_shape) const {
  MS_EXCEPTION_IF_NULL(src0_shape);
  MS_EXCEPTION_IF_NULL(src1_shape);
  MS_EXCEPTION_IF_NULL(dst_shape);
  bool need_swap = false;
  if (dst_shape->size() == 0) {
    (void)dst_shape->emplace_back(1);
    (void)src0_shape->emplace_back(1);
    (void)src1_shape->emplace_back(1);
  }
  MS_LOG(DEBUG) << "Binary broadcast in: src0: " << *src0_shape << " src1: " << *src1_shape << " dst: " << *dst_shape;
  if (src0_shape->size() != dst_shape->size()) {
    need_swap = true;
    for (size_t i = src0_shape->size(); i < dst_shape->size(); ++i) {
      (void)src0_shape->insert(src0_shape->begin(), 1);
    }
  } else if (src1_shape->size() != dst_shape->size()) {
    for (size_t i = src1_shape->size(); i < dst_shape->size(); ++i) {
      (void)src1_shape->insert(src1_shape->begin(), 1);
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

dnnl::memory::format_tag DeprecatedMKLCpuKernelMod::GetDefaultFormatTag(const dnnl::memory::dims &dims) const {
  static const std::vector<dnnl::memory::format_tag> tag_vec = {
    dnnl::memory::format_tag::a,      dnnl::memory::format_tag::ab,    dnnl::memory::format_tag::abc,
    dnnl::memory::format_tag::abcd,   dnnl::memory::format_tag::abcde, dnnl::memory::format_tag::abcdef,
    dnnl::memory::format_tag::abcdefg};
  size_t rank = dims.size();
  if (rank > tag_vec.size()) {
    MS_LOG(EXCEPTION) << "The kernel does not support construct " << rank << "-D tensor dnnl memory format_tag.";
  }
  return tag_vec[rank - 1];
}

dnnl::memory::desc DeprecatedMKLCpuKernelMod::GetDefaultMemDesc(const std::vector<int64_t> &shape) const {
  dnnl::memory::dims dims;
  if (shape.empty()) {
    (void)dims.insert(dims.end(), 1);
  } else {
    (void)dims.insert(dims.end(), shape.begin(), shape.end());
  }
  dnnl::memory::format_tag mem_tag = GetDefaultFormatTag(dims);
  auto mem_desc = CreateDesc<dnnl::memory::desc>(dims, dnnl::memory::data_type::f32, mem_tag);
  return mem_desc;
}

void DeprecatedMKLCpuKernelMod::AddArgument(int arg_key, const dnnl::memory::desc &mem_desc, bool alloc) {
  if (alloc) {
    arguments_[arg_key] = dnnl::memory(mem_desc, engine_);
  } else {
    arguments_[arg_key] = dnnl::memory(mem_desc, engine_, nullptr);
  }
}

void DeprecatedMKLCpuKernelMod::SetArgumentHandle(int arg_key, void *ptr) {
  auto arg_iter = arguments_.find(arg_key);
  if (arg_iter != arguments_.end()) {
    MS_LOG(DEBUG) << "begin to invoke dnnl::memory::set_data_handle";
    arg_iter->second.set_data_handle(ptr);
    MS_LOG(DEBUG) << "end to invoke dnnl::memory::set_data_handle";
  }
}

void DeprecatedMKLCpuKernelMod::ExecutePrimitive() {
  MS_EXCEPTION_IF_NULL(primitive_);
#ifdef USE_MS_THREADPOOL_FOR_DNNL
  // add auto search
  const std::vector<size_t> kSearchThreadList{4, 8, 16, 24, 32};
  const size_t kAvgCount = 5;
  const size_t kDiff = 2;
  size_t current_pow = parallel_search_info_.search_count / kAvgCount;
  auto mkl_pool = dynamic_cast<mkl_threadpool *>(mkl_threadpool_.get());
  if (current_pow < kSearchThreadList.size()) {
    if (parallel_search_info_.search_count % kAvgCount == 0) {
      parallel_search_info_.tmp_sum_cost_time = 0;
    }
    double start_time = GetTime();
    int current_thread_nums = kSearchThreadList[current_pow];
    mkl_pool->set_num_threads(current_thread_nums);
    MS_LOG(DEBUG) << "begin to invoke primitive::execute";
    primitive_->execute(stream_, arguments_);
    MS_LOG(DEBUG) << "end to invoke primitive::execute";
    double cost_time = GetTime() - start_time;
    // skip the first step to warm up.
    if (parallel_search_info_.search_count != 0) {
      parallel_search_info_.tmp_sum_cost_time += cost_time;
    }
    parallel_search_info_.search_count++;
    if (parallel_search_info_.search_count % kAvgCount == 0) {
      double avg_time = 0;
      // first avg will skip the first step
      if (parallel_search_info_.search_count / kAvgCount == 0) {
        avg_time = parallel_search_info_.tmp_sum_cost_time / (kAvgCount - 1);
      } else {
        avg_time = parallel_search_info_.tmp_sum_cost_time / kAvgCount;
      }
      if (parallel_search_info_.min_cost_time > avg_time) {
        parallel_search_info_.min_cost_time = avg_time;
        parallel_search_info_.best_pow = current_pow;
      } else if (current_pow - parallel_search_info_.best_pow >= kDiff) {
        parallel_search_info_.search_count = kAvgCount * kSearchThreadList.size();
      }
    }
  } else {
    int best_thread_nums = kSearchThreadList[parallel_search_info_.best_pow];
    mkl_pool->set_num_threads(best_thread_nums);
    MS_LOG(DEBUG) << "begin to invoke primitive::execute";
    primitive_->execute(stream_, arguments_);
    MS_LOG(DEBUG) << "end to invoke primitive::execute";
  }
#else
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
  primitive_->execute(stream_, arguments_);
  MS_LOG(DEBUG) << "end to invoke primitive::execute";
#endif
  (void)stream_.wait();
}

void DeprecatedMKLCpuKernelMod::SetDataHandle(dnnl::memory mem, void *ptr) {
  MS_LOG(DEBUG) << "begin to invoke dnnl::memory::set_data_handle";
  mem.set_data_handle(ptr);
  MS_LOG(DEBUG) << "end to invoke dnnl::memory::set_data_handle";
}

void *DeprecatedMKLCpuKernelMod::GetDataHandle(const dnnl::memory &mem) const {
  MS_LOG(DEBUG) << "begin to invoke dnnl::memory::get_data_handle";
  auto ptr = mem.get_data_handle();
  MS_LOG(DEBUG) << "end to invoke dnnl::memory::get_data_handle";
  return ptr;
}

size_t DeprecatedMKLCpuKernelMod::GetSize(const dnnl::memory::desc &desc) const {
  MS_LOG(DEBUG) << "begin to invoke dnnl::memory::desc::get_size()";
  auto size = desc.get_size();
  MS_LOG(DEBUG) << "end to invoke dnnl::memory::desc::get_size()";
  return size;
}

void DeprecatedMKLCpuKernelMod::Reorder(dnnl::memory *src_mem, dnnl::memory *dst_mem) {
  MS_LOG(DEBUG) << "begin to invoke constructor of dnnl::reorder";
  auto desc = dnnl::reorder(*src_mem, *dst_mem);
  MS_LOG(DEBUG) << "end to invoke constructor of dnnl::reorder";
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
  desc.execute(stream_, *src_mem, *dst_mem);
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
}

void MKLCpuKernelMod::GetPadding(const BaseOperatorPtr &base_operator, const std::vector<int64_t> &src_shape,
                                 const PaddingInfo &padding_info) const {
  MS_EXCEPTION_IF_NULL(base_operator);
  MS_EXCEPTION_IF_NULL(padding_info.padding_l);
  MS_EXCEPTION_IF_NULL(padding_info.padding_r);
  size_t src_dim = src_shape.size();
  if (src_dim < NC_LEN) {
    MS_LOG(EXCEPTION) << "Set pad only support src dim >= 2!";
  }
  const size_t dim_exclude_nc = src_dim - NC_LEN;
  std::vector<int64_t> shape_exclude_nc;
  for (size_t i = NC_LEN; i < src_dim; ++i) {
    shape_exclude_nc.push_back(SizeToLong(src_shape[i]));
  }

  if (padding_info.pad_mode == PAD_MODE_LOWER_SAME || padding_info.pad_mode == PAD_MODE_UPPER_SAME) {
    for (size_t i = 0; i < dim_exclude_nc; ++i) {
      int64_t wh = shape_exclude_nc[i];
      int64_t out = (wh + padding_info.stride[i] - 1) / padding_info.stride[i];
      int64_t effective_k = (SizeToLong(padding_info.kernel_size[i]) - 1) * padding_info.dilation[i] + 1;
      int64_t pad_along = std::max(int64_t(0), (out - 1) * padding_info.stride[i] + effective_k - wh);
      int64_t pad = pad_along / 2;
      padding_info.padding_l->push_back(pad);
      padding_info.padding_r->push_back(pad_along - pad);
    }
  } else if (padding_info.pad_mode == PAD_MODE_LOWER_VALID || padding_info.pad_mode == PAD_MODE_UPPER_VALID) {
    for (size_t i = 0; i < dim_exclude_nc; ++i) {
      padding_info.padding_l->push_back(0);
      padding_info.padding_r->push_back(0);
    }
  } else {
    std::vector<int64_t> pad = GetValue<std::vector<int64_t>>(base_operator->GetAttr(PAD_LIST));
    GeneratePaddingForPadMode(padding_info, shape_exclude_nc, pad);
  }
}

bool MKLCpuKernelMod::BinaryBroadCast(std::vector<size_t> *src0_shape, std::vector<size_t> *src1_shape,
                                      std::vector<size_t> *dst_shape) const {
  MS_EXCEPTION_IF_NULL(src0_shape);
  MS_EXCEPTION_IF_NULL(src1_shape);
  MS_EXCEPTION_IF_NULL(dst_shape);
  bool need_swap = false;
  if (dst_shape->size() == 0) {
    (void)dst_shape->emplace_back(1);
    (void)src0_shape->emplace_back(1);
    (void)src1_shape->emplace_back(1);
  }
  MS_LOG(DEBUG) << "Binary broadcast in: src0: " << *src0_shape << " src1: " << *src1_shape << " dst: " << *dst_shape;
  if (src0_shape->size() != dst_shape->size()) {
    need_swap = true;
    for (size_t i = src0_shape->size(); i < dst_shape->size(); ++i) {
      (void)src0_shape->insert(src0_shape->begin(), 1);
    }
  } else if (src1_shape->size() != dst_shape->size()) {
    for (size_t i = src1_shape->size(); i < dst_shape->size(); ++i) {
      (void)src1_shape->insert(src1_shape->begin(), 1);
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

dnnl::memory::format_tag MKLCpuKernelMod::GetDefaultFormatTag(const dnnl::memory::dims &dims) const {
  static const std::vector<dnnl::memory::format_tag> tag_vec = {
    dnnl::memory::format_tag::a,      dnnl::memory::format_tag::ab,    dnnl::memory::format_tag::abc,
    dnnl::memory::format_tag::abcd,   dnnl::memory::format_tag::abcde, dnnl::memory::format_tag::abcdef,
    dnnl::memory::format_tag::abcdefg};
  size_t rank = dims.size();
  if (rank > tag_vec.size()) {
    MS_LOG(EXCEPTION) << "The kernel does not support construct " << rank << "-D tensor dnnl memory format_tag.";
  }
  return tag_vec[rank - 1];
}

dnnl::memory::desc MKLCpuKernelMod::GetExactMemDesc(const std::vector<size_t> &shape,
                                                    dnnl::memory::data_type type) const {
  dnnl::memory::dims dims;
  if (shape.empty()) {
    (void)dims.insert(dims.end(), 1);
  } else {
    (void)dims.insert(dims.end(), shape.begin(), shape.end());
  }
  dnnl::memory::format_tag mem_tag = GetDefaultFormatTag(dims);
  auto mem_desc = CreateDesc<dnnl::memory::desc>(dims, type, mem_tag);
  return mem_desc;
}

void MKLCpuKernelMod::AddArgument(int arg_key, const dnnl::memory::desc &mem_desc, bool alloc) {
  if (alloc) {
    arguments_[arg_key] = dnnl::memory(mem_desc, engine_);
  } else {
    arguments_[arg_key] = dnnl::memory(mem_desc, engine_, nullptr);
  }
}

void MKLCpuKernelMod::SetArgumentHandle(int arg_key, void *ptr) {
  auto arg_iter = arguments_.find(arg_key);
  if (arg_iter != arguments_.end()) {
    MS_LOG(DEBUG) << "begin to invoke dnnl::memory::set_data_handle";
    arg_iter->second.set_data_handle(ptr);
    MS_LOG(DEBUG) << "end to invoke dnnl::memory::set_data_handle";
  }
}

void MKLCpuKernelMod::ExecutePrimitive() {
  MS_EXCEPTION_IF_NULL(primitive_);
#ifdef USE_MS_THREADPOOL_FOR_DNNL
  // add auto search
  const std::vector<size_t> kSearchThreadList{4, 8, 16, 24, 32};
  const size_t kAvgCount = 5;
  const size_t kDiff = 2;
  size_t current_pow = parallel_search_info_.search_count / kAvgCount;
  auto mkl_pool = dynamic_cast<mkl_threadpool *>(mkl_threadpool_.get());
  if (current_pow < kSearchThreadList.size()) {
    if (parallel_search_info_.search_count % kAvgCount == 0) {
      parallel_search_info_.tmp_sum_cost_time = 0;
    }
    double start_time = GetTime();
    int current_thread_nums = kSearchThreadList[current_pow];
    mkl_pool->set_num_threads(current_thread_nums);
    MS_LOG(DEBUG) << "begin to invoke primitive::execute";
    primitive_->execute(stream_, arguments_);
    MS_LOG(DEBUG) << "end to invoke primitive::execute";
    double cost_time = GetTime() - start_time;
    // skip the first step to warm up.
    if (parallel_search_info_.search_count != 0) {
      parallel_search_info_.tmp_sum_cost_time += cost_time;
    }
    parallel_search_info_.search_count++;
    if (parallel_search_info_.search_count % kAvgCount == 0) {
      double avg_time = 0;
      // first avg will skip the first step
      if (parallel_search_info_.search_count / kAvgCount == 0) {
        avg_time = parallel_search_info_.tmp_sum_cost_time / (kAvgCount - 1);
      } else {
        avg_time = parallel_search_info_.tmp_sum_cost_time / kAvgCount;
      }
      if (parallel_search_info_.min_cost_time > avg_time) {
        parallel_search_info_.min_cost_time = avg_time;
        parallel_search_info_.best_pow = current_pow;
      } else if (current_pow - parallel_search_info_.best_pow >= kDiff) {
        parallel_search_info_.search_count = kAvgCount * kSearchThreadList.size();
      }
    }
  } else {
    int best_thread_nums = kSearchThreadList[parallel_search_info_.best_pow];
    mkl_pool->set_num_threads(best_thread_nums);
    MS_LOG(DEBUG) << "begin to invoke primitive::execute";
    primitive_->execute(stream_, arguments_);
    MS_LOG(DEBUG) << "end to invoke primitive::execute";
  }
#else
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
  primitive_->execute(stream_, arguments_);
  MS_LOG(DEBUG) << "end to invoke primitive::execute";
#endif
  (void)stream_.wait();
}

void MKLCpuKernelMod::SetDataHandle(dnnl::memory mem, void *ptr) {
  MS_LOG(DEBUG) << "begin to invoke dnnl::memory::set_data_handle";
  mem.set_data_handle(ptr);
  MS_LOG(DEBUG) << "end to invoke dnnl::memory::set_data_handle";
}

void *MKLCpuKernelMod::GetDataHandle(const dnnl::memory &mem) const {
  MS_LOG(DEBUG) << "begin to invoke dnnl::memory::get_data_handle";
  auto ptr = mem.get_data_handle();
  MS_LOG(DEBUG) << "end to invoke dnnl::memory::get_data_handle";
  return ptr;
}

size_t MKLCpuKernelMod::GetSize(const dnnl::memory::desc &desc) const {
  MS_LOG(DEBUG) << "begin to invoke dnnl::memory::desc::get_size()";
  auto size = desc.get_size();
  MS_LOG(DEBUG) << "end to invoke dnnl::memory::desc::get_size()";
  return size;
}

dnnl::memory::data_type MKLCpuKernelMod::GetDnnlDataType(TypeId ms_type_id) const {
  static const std::map<TypeId, dnnl::memory::data_type> dnnl_data_type_map = {
    {kNumberTypeFloat16, dnnl::memory::data_type::f16},
    {kNumberTypeFloat32, dnnl::memory::data_type::f32},
    {kNumberTypeInt32, dnnl::memory::data_type::s32},
    {kNumberTypeInt8, dnnl::memory::data_type::s8},
    {kNumberTypeUInt8, dnnl::memory::data_type::u8}};
  auto iter = dnnl_data_type_map.find(ms_type_id);
  if (iter == dnnl_data_type_map.end()) {
    MS_LOG(WARNING) << "Dnnl do not support data type:" << TypeIdToString(ms_type_id);
    return dnnl::memory::data_type::undef;
  }
  return iter->second;
}

void MKLCpuKernelMod::Reorder(dnnl::memory *src_mem, dnnl::memory *dst_mem) {
  MS_LOG(DEBUG) << "begin to invoke constructor of dnnl::reorder";
  auto desc = dnnl::reorder(*src_mem, *dst_mem);
  MS_LOG(DEBUG) << "end to invoke constructor of dnnl::reorder";
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
  desc.execute(stream_, *src_mem, *dst_mem);
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
}
}  // namespace kernel
}  // namespace mindspore
