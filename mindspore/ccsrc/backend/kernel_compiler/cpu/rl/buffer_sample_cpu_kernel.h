/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_SAMPLE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_SAMPLE_CPU_KERNEL_H_
#include <stdlib.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class BufferCPUSampleKernel : public CPUKernel {
 public:
  BufferCPUSampleKernel() : element_nums_(0), capacity_(0), batch_size_(0), exp_size_(0), seed_(0), unique_(false) {}

  ~BufferCPUSampleKernel() override = default;
  void Init(const CNodePtr &kernel_node) {
    auto shapes = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "buffer_elements");
    auto types = AnfAlgo::GetNodeAttr<std::vector<TypePtr>>(kernel_node, "buffer_dtype");
    capacity_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "capacity");
    seed_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "seed");
    unique_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "unique");
    batch_size_ = LongToSize(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "batch_size"));
    element_nums_ = shapes.size();
    for (size_t i = 0; i < element_nums_; i++) {
      exp_element_list.push_back(shapes[i] * UnitSizeInBytes(types[i]->type_id()));
    }
    // init seed for random_shuffle and uniform distribution
    if (seed_ == 0) {
      std::srand(time(nullptr));
      generator_.seed(time(nullptr));
    } else {
      std::srand(seed_);
      generator_.seed(seed_);
    }
    // buffer size
    for (auto i : exp_element_list) {
      input_size_list_.push_back(i * capacity_);
      output_size_list_.push_back(i * batch_size_);
      exp_size_ += i;
    }
    // count and head
    input_size_list_.push_back(sizeof(int));
    input_size_list_.push_back(sizeof(int));
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) {
    auto count_addr = GetDeviceAddress<int>(inputs, element_nums_);
    auto head_addr = GetDeviceAddress<int>(inputs, element_nums_ + 1);

    if ((head_addr[0] > 0 && SizeToLong(batch_size_) > capacity_) ||
        (head_addr[0] == 0 && SizeToLong(batch_size_) > count_addr[0])) {
      MS_LOG(ERROR) << "The batch size " << batch_size_ << " is larger than total buffer size "
                    << std::min(capacity_, IntToLong(count_addr[0]));
    }
    // Generate random indexes
    // If unique_ == true, use random_shuffle to guarantee the index in generated indexes is unique.
    // If unique_ == false, use a uniform distribution to generate the indexes. Some of the indexes may be repeated.
    // Case unique_ == false has a better perfomace than case unique_ ==  true.
    std::vector<size_t> indexes;
    if (unique_) {
      for (size_t i = 0; i < IntToSize(count_addr[0]); ++i) {
        (void)indexes.emplace_back(i);
      }
      random_shuffle(indexes.begin(), indexes.end(), [&](int i) { return std::rand() % i; });
    } else {
      std::uniform_int_distribution<> distrib(0, count_addr[0]);
      for (size_t i = 0; i < batch_size_; ++i) {
        (void)indexes.emplace_back(distrib(generator_));
      }
    }

    auto task = [&](size_t start, size_t end) {
      for (size_t j = start; j < end; j++) {
        size_t index = indexes[j];
        for (size_t i = 0; i < element_nums_; i++) {
          auto buffer_addr = GetDeviceAddress<unsigned char>(inputs, i);
          auto output_addr = GetDeviceAddress<unsigned char>(outputs, i);
          auto one_exp_len = exp_element_list[i];
          size_t dist_len = one_exp_len;
          if (memcpy_s(output_addr + j * one_exp_len, one_exp_len, buffer_addr + index * one_exp_len, dist_len) !=
              EOK) {
            MS_LOG(EXCEPTION) << "Launch kernel error: memcpy failed";
          }
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, batch_size_);
    return true;
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  void InitSizeLists() { return; }

 private:
  size_t element_nums_;
  int64_t capacity_;
  size_t batch_size_;
  int64_t exp_size_;
  int64_t seed_;
  bool unique_;
  std::mt19937 generator_;
  std::vector<size_t> exp_element_list;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_BUFFER_SAMPLE_CPU_KERNEL_H_
