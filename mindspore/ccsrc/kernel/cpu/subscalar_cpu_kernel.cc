/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/subscalar_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void SubscalarCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  offset_ = AnfAlgo::GetNodeAttr<int>(kernel_node, "input_y");
  MS_LOG(DEBUG) << "offset: " << offset_;
}

void sub_task(int *in_addr, int *out_addr, size_t lens, int offset) {
  for (size_t i = 0; i < lens; i++) {
    out_addr[i] = in_addr[i] - offset;
  }
}

bool SubscalarCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspace*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
#if defined(_WIN32) || defined(_WIN64)
  auto start_time = std::chrono::steady_clock::now();
#else
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
#endif
  auto input_addr = reinterpret_cast<int *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<int *>(outputs[0]->addr);
  auto lens = inputs[0]->size / sizeof(int);
  if (lens < 10000) {
    for (size_t i = 0; i < lens; i++) {
      output_addr[i] = input_addr[i] - offset_;
    }
  } else {
    size_t thread_num = 4;
    std::thread threads[4];
    size_t process_lens = (lens + thread_num - 1) / thread_num;
    size_t process_offset = 0;
    for (size_t i = 0; i < thread_num; i++) {
      threads[i] =
        std::thread(sub_task, input_addr + process_offset, output_addr + process_offset, process_lens, offset_);
      if (process_offset + process_lens > lens) {
        process_lens = lens - process_offset;
        process_offset = lens;
      } else {
        process_offset += process_lens;
      }
    }
    for (size_t i = 0; i < thread_num; i++) {
      threads[i].join();
    }
  }
#if defined(_WIN32) || defined(_WIN64)
  auto end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000000>> cost = end_time - start_time;
  MS_LOG(INFO) << "SubscaleCPUKernel, used time: " << cost.count() << " us";
#else
  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = 1000000 * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "SubscalarCPUKernel, used time: " << time << " us";
#endif
  return true;
}
}  // namespace kernel
}  // namespace mindspore
