/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/tensor_copy_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include <functional>
#include <map>

namespace mindspore {
namespace kernel {
namespace tensor_copy_cpu {
namespace {
constexpr size_t ktInput = 0;
constexpr size_t ktOutput = 0;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool TensorCopyCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  auto input_type = inputs[ktInput]->dtype_id();
  auto output_type = inputs[ktOutput]->dtype_id();
  if (input_type != output_type) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the type of 'input' and the type of 'output' should be same, but 'input' type is "
                  << input_type << "while 'output' type is " << output_type;
    return false;
  }
  return true;
}

int TensorCopyCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[ktInput]->GetShapeVector();
  auto output_shape = outputs[ktOutput]->GetShapeVector();
  if (input_shape != output_shape) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'input' and the shape of 'output' should be same, but 'input' shape is "
                  << input_shape << "while 'output' shape is " << output_shape;
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

bool TensorCopyCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                    const std::vector<kernel::KernelTensor *> & /* workspace */,
                                    const std::vector<kernel::KernelTensor *> &outputs) {
  auto input = GetDeviceAddress<void>(inputs, 0);
  auto output = GetDeviceAddress<void>(outputs, 0);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(output);

  auto copy_size = inputs[0]->size();
  MS_EXCEPTION_IF_CHECK_FAIL(copy_size == outputs[0]->size(),
                             "For " + kernel_name_ + ", the size of 'input' and the size of 'output' should be same.");

  if (copy_size == 0) {
    MS_LOG(INFO) << "For '" << kernel_name_ << "', the size of 'input' and the size of 'output' is 0, skip copy.";
    return true;
  }

  constexpr size_t kGrainSize = 32768;
  if (copy_size <= kGrainSize) {
    auto ret = memcpy_s(output, outputs[0]->size(), input, inputs[0]->size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', memory copy failed. Error no: " << ret << ", Copy input:" << input
                    << " size=" << inputs[0]->size() << " ,To output:" << output << " size=" << outputs[0]->size();
      return false;
    }
  } else {
    auto copy_task = [input, output](size_t start, size_t end) {
      size_t remain_size = LongToSize((SizeToLong(end) - SizeToLong(start)));
      auto input_ptr = static_cast<uint8_t *>(input) + start;
      auto output_ptr = static_cast<uint8_t *>(output) + start;
      while (remain_size > SECUREC_MEM_MAX_LEN) {
        auto ret = memcpy_s(output_ptr, SECUREC_MEM_MAX_LEN, input_ptr, SECUREC_MEM_MAX_LEN);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "For TensorMove, memcpy_s error. Error no: " << ret << ", output_ptr: " << output_ptr
                            << ", input_ptr: " << input_ptr << ", copy_size: " << SECUREC_MEM_MAX_LEN;
        }
        remain_size = LongToSize(SizeToLong(remain_size) - SECUREC_MEM_MAX_LEN);
        output_ptr += SECUREC_MEM_MAX_LEN;
        input_ptr += SECUREC_MEM_MAX_LEN;
      }
      if (remain_size != 0U) {
        auto ret = memcpy_s(output_ptr, remain_size, input_ptr, remain_size);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "For TensorMove, memcpy_s error. Error no: " << ret << ", output_ptr: " << output_ptr
                            << ", input_ptr: " << input_ptr << ", copy_size: " << remain_size;
        }
      }
    };
    ParallelLaunchAutoSearch(copy_task, copy_size, this, &parallel_search_info_);
  }

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorMove, TensorCopyCpuKernelMod);
}  // namespace tensor_copy_cpu
}  // namespace kernel
}  // namespace mindspore
