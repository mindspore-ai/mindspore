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
#include <thread>
#include <string>
#include "kernel/cpu/embedding_look_up_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "device/cpu/mpi/mpi_adapter.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
void EmbeddingLookUpCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);

  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  input_lens_ = 1;
  for (auto shape : input_shape_) {
    MS_LOG(DEBUG) << "input shape: " << shape;
    input_lens_ = input_lens_ * shape;
  }
  MS_LOG(DEBUG) << "input lens: " << input_lens_;

  indices_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  indices_lens_ = 1;
  for (auto shape : indices_shape_) {
    MS_LOG(DEBUG) << "indice shape: " << shape;
    indices_lens_ = indices_lens_ * shape;
  }
  MS_LOG(DEBUG) << "indice lens: " << indices_lens_;

  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  for (auto shape : output_shape_) {
    MS_LOG(DEBUG) << "output shape: " << shape;
  }
  auto output_type = AnfAlgo::GetOutputInferDataType(kernel_node, 0);
  MS_LOG(DEBUG) << "output type: " << output_type;

  int axis = AnfAlgo::GetNodeAttr<int>(kernel_node, "axis");
  MS_LOG(DEBUG) << "axis: " << axis;
  if (axis_ < 0) {
    axis = axis + SizeToInt(input_shape_.size());
  }
  axis_ = 4 - input_shape_.size() + axis;
  MS_LOG(DEBUG) << "axis_: " << axis_;
  reduce_scatter_flag_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "reduce_scatter_flag");
  MS_LOG(DEBUG) << "reduce_scatter_flag: " << reduce_scatter_flag_;
  if (reduce_scatter_flag_) {
    size_t gatherv2_out_lens = 1;
    for (int i = 0; i < SizeToInt(input_shape_.size()); i++) {
      if (i == axis) {
        for (int j = 0; j < SizeToInt(indices_shape_.size()); j++) {
          MS_LOG(DEBUG) << "gatherv2 out shape: " << indices_shape_[j];
          gatherv2_out_lens = gatherv2_out_lens * indices_shape_[j];
        }
      } else {
        MS_LOG(DEBUG) << "gatherv2 out shape: " << input_shape_[i];
        gatherv2_out_lens = gatherv2_out_lens * input_shape_[i];
      }
    }
    gatherv2_out_lens_ = gatherv2_out_lens * sizeof(float);
    MS_LOG(DEBUG) << "gatherv2 out lens: " << gatherv2_out_lens_;
    gather_v2_out_ = malloc(gatherv2_out_lens_);
    if (gather_v2_out_ == nullptr) {
      MS_LOG(EXCEPTION) << "EmbeddingLookUpCPUKernel malloc failed, malloc lens: " << gatherv2_out_lens_;
    }
    memset_s(gather_v2_out_, gatherv2_out_lens_, 0, gatherv2_out_lens_);

    split_num_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "split_num");
    MS_LOG(DEBUG) << "split_num: " << split_num_;
  }
  offset_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "offset");
  MS_LOG(DEBUG) << "offset: " << offset_;
  CPUKernelUtils::ExpandDimsTo4(&input_shape_);
  CPUKernelUtils::ExpandDimsTo4(&output_shape_);
}

bool EmbeddingLookUpCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> & /*workspace*/,
                                      const std::vector<kernel::AddressPtr> &outputs) {
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
#endif
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  MS_LOG(DEBUG) << "output addr: " << output_addr << "output size: " << outputs[0]->size;
  float *gather_out_addr = reduce_scatter_flag_ ? reinterpret_cast<float *>(gather_v2_out_) : output_addr;
  MS_LOG(DEBUG) << "gatherv2 out addr: " << gather_out_addr;
  size_t dim0 = input_shape_[0];
  size_t dim1 = input_shape_[1];
  size_t dim2 = input_shape_[2];

  if (axis_ == 3) {
    for (size_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < dim1; ++j) {
        for (size_t k = 0; k < dim2; ++k) {
          LookUpTable(inputs, i, j, k, &gather_out_addr);
        }
      }
    }
  } else if (axis_ == 2) {
    for (size_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < dim1; ++j) {
        LookUpTable(inputs, i, j, 0, &gather_out_addr);
      }
    }
  } else if (axis_ == 1) {
    for (size_t i = 0; i < dim0; ++i) {
      LookUpTable(inputs, i, 0, 0, &gather_out_addr);
    }
  } else if (axis_ == 0) {
    LookUpTable(inputs, 0, 0, 0, &gather_out_addr);
  }

  if (reduce_scatter_flag_) {
    size_t one_split_lens = gatherv2_out_lens_ / split_num_ / sizeof(float);
    size_t reduce_scatter_out_lens = one_split_lens / 8;
    const std::vector<int> &group = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int i = 0; i < split_num_; i++) {
      device::cpu::MPIAdapter::Instance().ReduceScatter(reinterpret_cast<float *>(gather_v2_out_) + i * one_split_lens,
                                                        output_addr + i * reduce_scatter_out_lens, group,
                                                        one_split_lens / 8, "sum");
    }
  }
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000000>> cost = end_time - start_time;
  MS_LOG(INFO) << "EmbeddingLookUpCPUKernel, used time: " << cost.count() << " us";
#else
  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = 1000000 * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "EmbeddingLookUpCPUKernel, used time: " << time << " us";
#endif
  return true;
}

void memcpy_task(std::vector<float *> mem_dest_addr_list, std::vector<float *> mem_src_addr_list, size_t start,
                 size_t end, size_t lens) {
  for (size_t i = start; i < end; i++) {
    auto ret = memcpy_s(mem_dest_addr_list[i], lens, mem_src_addr_list[i], lens);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memery copy failed.";
    }
  }
  return;
}

void EmbeddingLookUpCPUKernel::LookUpTable(const std::vector<kernel::AddressPtr> &inputs, size_t dim0, size_t dim1,
                                           size_t dim2, float **output_addr) {
  auto input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto indices_addr = reinterpret_cast<int *>(inputs[1]->addr);
  size_t num = CPUKernelUtils::GetElementNumOnAxis(input_shape_, axis_);
  size_t lens = num * sizeof(float);
  std::vector<float *> mem_dest_addr_list;
  std::vector<float *> mem_src_addr_list;
  for (size_t i = 0; i < indices_lens_; ++i) {
    int indices = indices_addr[i] - offset_;
    if (indices >= 0) {
      size_t index = IntToSize(indices);
      if (index < input_shape_[axis_]) {
        size_t pos = 0;
        if (axis_ == 3) {
          pos = CPUKernelUtils::CalcOffset(input_shape_, dim0, dim1, dim2, index);
        } else if (axis_ == 2) {
          pos = CPUKernelUtils::CalcOffset(input_shape_, dim0, dim1, index, 0);
        } else if (axis_ == 1) {
          pos = CPUKernelUtils::CalcOffset(input_shape_, dim0, index, 0, 0);
        } else if (axis_ == 0) {
          pos = CPUKernelUtils::CalcOffset(input_shape_, index, 0, 0, 0);
        }

        if (pos + num <= input_lens_) {
          mem_dest_addr_list.push_back(*output_addr);
          mem_src_addr_list.push_back(input_addr + pos);
        }
      }
    }
    *output_addr += num;
  }

  const size_t thread_num = 8;
  std::thread threads[8];
  size_t memcpy_lens = mem_dest_addr_list.size();
  size_t start = 0;
  size_t ones_copy_lens = (memcpy_lens + thread_num - 1) / thread_num;
  size_t i;
  for (i = 0; i < thread_num; i++) {
    if (start > memcpy_lens) {
      break;
    }
    auto end = (start + ones_copy_lens) > memcpy_lens ? memcpy_lens : start + ones_copy_lens;
    threads[i] = std::thread(memcpy_task, mem_dest_addr_list, mem_src_addr_list, start, end, lens);
    start = start + ones_copy_lens;
  }
  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
}

void EmbeddingLookUpCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > 4) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size()
                      << ", but EmbeddingLookUpCPUKernel olny support 4d or lower.";
  }

  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but EmbeddingLookUpCPUKernel needs 2.";
  }
}
}  // namespace kernel
}  // namespace mindspore
