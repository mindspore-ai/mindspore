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

#include "plugin/device/cpu/kernel/embedding_look_up_comm_grad_cpu_kernel.h"
#include <thread>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/hal/device/mpi/mpi_interface.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kEmbeddingLookupCommGradInputsNum = 1;
constexpr size_t kEmbeddingLookupCommGradOutputsNum = 1;
}  // namespace

void EmbeddingLookUpCommGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto split_num = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "split_num");
  split_num_ = LongToSize(split_num);
  MS_LOG(INFO) << "split_num: " << split_num;
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (split_num <= 0 || split_num_ == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'split_num' should be greater than 0, but got "
                      << split_num;
  }
  split_num_ = LongToSize(split_num);
  if (input_shape.size() < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input should be at least 1-D, but got: " << input_shape.size() << "-D";
  }
  if (input_shape[0] % split_num_ != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the first dimension value of input should be multiple of "
                         "'split_num', but got 'split_num': "
                      << split_num_ << " and the first dimension value of input: " << input_shape[0];
  }
}

bool EmbeddingLookUpCommGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kEmbeddingLookupCommGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kEmbeddingLookupCommGradOutputsNum, kernel_name_);
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
#endif
  auto *input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  size_t input_size = inputs[0]->size;
  size_t output_size = outputs[0]->size;
  MS_LOG(DEBUG) << "input addr: " << input_addr << "input size: " << input_size;
  MS_LOG(DEBUG) << "output addr: " << output_addr << "output size: " << output_size;
  auto ret = memset_s(output_addr, output_size, 0, output_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset failed. Error no: " << ret;
  }
  const std::vector<int> &rank_group = {0, 1, 2, 3, 4, 5, 6, 7};
  size_t input_split_lens = input_size / split_num_ / sizeof(float_t);
  size_t output_split_lens = output_size / split_num_ / sizeof(float_t);
  for (size_t i = 0; i < split_num_; ++i) {
    (void)MPIAllGather(input_addr + i * input_split_lens, output_addr + i * output_split_lens, rank_group,
                       input_split_lens);
  }
  const uint64_t kUSecondInSecond = 1000000;
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, kUSecondInSecond>> cost = end_time - start_time;
  MS_LOG(INFO) << "EmbeddingLookUpCommGradCpuKernelMod, used time: " << cost.count() << " us";
#else
  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "EmbeddingLookUpCommGradCpuKernelMod, used time: " << time << " us";
#endif
  return true;
}
}  // namespace kernel
}  // namespace mindspore
