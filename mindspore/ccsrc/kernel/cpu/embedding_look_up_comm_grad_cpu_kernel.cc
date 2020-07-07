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
#include "kernel/cpu/embedding_look_up_comm_grad_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "device/cpu/mpi/mpi_adapter.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
void EmbeddingLookUpCommGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  split_num_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "split_num");
  MS_LOG(INFO) << "split_num: " << split_num_;
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape[0] % split_num_ != 0) {
    MS_LOG(EXCEPTION) << "Input shape[0] is " << input_shape[0] << ", but it must be multiple of split_num.";
  }
}

bool EmbeddingLookUpCommGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> & /*workspace*/,
                                              const std::vector<kernel::AddressPtr> &outputs) {
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
#endif
  auto input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  size_t input_size = inputs[0]->size;
  size_t output_size = outputs[0]->size;
  MS_LOG(DEBUG) << "input addr: " << input_addr << "input size: " << input_size;
  MS_LOG(DEBUG) << "output addr: " << output_addr << "output size: " << output_size;
  memset_s(output_addr, output_size, 0, output_size);
  const std::vector<int> &rank_group = {0, 1, 2, 3, 4, 5, 6, 7};
  size_t input_split_lens = input_size / split_num_ / sizeof(float_t);
  size_t output_split_lens = output_size / split_num_ / sizeof(float_t);
  auto mpi_instance = device::cpu::MPIAdapter::Instance();
  MS_EXCEPTION_IF_NULL(mpi_instance);
  for (int i = 0; i < split_num_; i++) {
    mpi_instance->AllGather(input_addr + i * input_split_lens, output_addr + i * output_split_lens, rank_group,
                            input_split_lens);
  }
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000000>> cost = end_time - start_time;
  MS_LOG(INFO) << "EmbeddingLookUpCommGradCPUKernel, used time: " << cost.count() << " us";
#else
  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = 1000000 * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "EmbeddingLookUpCommGradCPUKernel, used time: " << time << " us";
#endif
  return true;
}

void EmbeddingLookUpCommGradCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but EmbeddingLookUpCommGradCPUKernel needs 1.";
  }
}
}  // namespace kernel
}  // namespace mindspore
