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
    input_lens_ = input_lens_ * shape;
  }
  indices_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  indices_lens_ = 1;
  for (auto shape : indices_shape_) {
    indices_lens_ = indices_lens_ * shape;
  }
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  axis_ = 4 - input_shape_.size();
  reduce_scatter_flag_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "reduce_scatter_flag");
#ifdef ENABLE_MPI
  if (reduce_scatter_flag_) {
    size_t gatherv2_out_lens = 1;
    for (int i = 0; i < SizeToInt(input_shape_.size()); i++) {
      if (i == 0) {
        for (int j = 0; j < SizeToInt(indices_shape_.size()); j++) {
          gatherv2_out_lens = gatherv2_out_lens * indices_shape_[j];
        }
      } else {
        gatherv2_out_lens = gatherv2_out_lens * input_shape_[i];
      }
    }
    gatherv2_out_lens_ = gatherv2_out_lens * sizeof(float);
    gather_v2_out_ = malloc(gatherv2_out_lens_);
    if (gather_v2_out_ == nullptr) {
      MS_LOG(EXCEPTION) << "EmbeddingLookUpCPUKernel malloc failed, malloc lens: " << gatherv2_out_lens_;
    }
    auto ret = memset_s(gather_v2_out_, gatherv2_out_lens_, 0, gatherv2_out_lens_);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "EmbeddingLookUpCPUKernel memset gatherv2 out buff failed";
    }
    split_num_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "split_num");
  }
#else
  if (reduce_scatter_flag_) {
    MS_LOG(EXCEPTION) << "Not Enable MPI, please build version with -M on when set reduce_scatter_flag true";
  }
#endif
  offset_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "offset");
  CPUKernelUtils::ExpandDimsTo4(&input_shape_);
  CPUKernelUtils::ExpandDimsTo4(&output_shape_);
}

bool EmbeddingLookUpCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> & /*workspace*/,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  float *gather_out_addr = reduce_scatter_flag_ ? reinterpret_cast<float *>(gather_v2_out_) : output_addr;
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
#ifdef ENABLE_MPI
  if (reduce_scatter_flag_) {
    size_t one_split_lens = gatherv2_out_lens_ / split_num_ / sizeof(float);
    size_t reduce_scatter_out_lens = one_split_lens / 8;
    const std::vector<int> &group = {0, 1, 2, 3, 4, 5, 6, 7};
    auto mpi_instance = device::cpu::MPIAdapter::Instance();
    MS_EXCEPTION_IF_NULL(mpi_instance);
    for (int i = 0; i < split_num_; i++) {
      mpi_instance->ReduceScatter(reinterpret_cast<float *>(gather_v2_out_) + i * one_split_lens,
                                  output_addr + i * reduce_scatter_out_lens, group, one_split_lens / 8, "sum");
    }
  }
#endif
  return true;
}

void LookUpTable_task(const float *input_addr, float *output_addr, const int *indices_addr, size_t indices_lens,
                      size_t num, size_t dim0, size_t dim1, size_t dim2, int offset, size_t axis,
                      std::vector<size_t> input_shape, size_t input_lens) {
  size_t lens = num * sizeof(float);
  for (size_t i = 0; i < indices_lens; ++i) {
    int indices = indices_addr[i] - offset;
    if (indices >= 0) {
      size_t index = IntToSize(indices);
      if (index < input_shape[axis]) {
        size_t pos = 0;
        if (axis == 3) {
          pos = CPUKernelUtils::CalcOffset(input_shape, dim0, dim1, dim2, index);
        } else if (axis == 2) {
          pos = CPUKernelUtils::CalcOffset(input_shape, dim0, dim1, index, 0);
        } else if (axis == 1) {
          pos = CPUKernelUtils::CalcOffset(input_shape, dim0, index, 0, 0);
        } else if (axis == 0) {
          pos = CPUKernelUtils::CalcOffset(input_shape, index, 0, 0, 0);
        }
        if (pos + num <= input_lens) {
          auto ret = memcpy_s(output_addr, lens, input_addr + pos, lens);
          if (ret != EOK) {
            MS_LOG(EXCEPTION) << "LookUpTable task memcpy failed.";
          }
        } else {
          auto ret = memset_s(output_addr, lens, 0, lens);
          if (ret != EOK) {
            MS_LOG(EXCEPTION) << "LookUpTable task memset failed.";
          }
        }
      } else {
        auto ret = memset_s(output_addr, lens, 0, lens);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "LookUpTable task memset failed.";
        }
      }
    } else {
      auto ret = memset_s(output_addr, lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "LookUpTable task memset failed.";
      }
    }
    output_addr += num;
  }
}

void EmbeddingLookUpCPUKernel::LookUpTable(const std::vector<kernel::AddressPtr> &inputs, size_t dim0, size_t dim1,
                                           size_t dim2, float **output_addr) {
  auto input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto indices_addr = reinterpret_cast<int *>(inputs[1]->addr);
  size_t num = CPUKernelUtils::GetElementNumOnAxis(input_shape_, axis_);
  float *task_out_addr = *output_addr;
  const size_t thread_num = 8;
  std::thread threads[8];
  size_t task_proc_lens = (indices_lens_ + thread_num - 1) / thread_num;
  size_t i;
  size_t task_offset = 0;
  MS_LOG(DEBUG) << "indices_lens_: " << indices_lens_ << " one task proc lens:" << task_proc_lens;
  for (i = 0; i < thread_num; i++) {
    if (task_offset >= indices_lens_) {
      break;
    }
    MS_LOG(DEBUG) << "task_offset: " << task_offset << " task_proc_lenss:" << task_proc_lens;
    threads[i] =
      std::thread(LookUpTable_task, input_addr, task_out_addr + task_offset * num, indices_addr + task_offset,
                  task_proc_lens, num, dim0, dim1, dim2, offset_, axis_, input_shape_, input_lens_);
    task_offset += task_proc_lens;
    if (task_offset + task_proc_lens > indices_lens_) {
      task_proc_lens = indices_lens_ - task_offset;
    }
  }
  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  *output_addr += num * indices_lens_;
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
