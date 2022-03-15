/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BATCHTOSPACE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BATCHTOSPACE_HELPER_H_
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/batchtospace_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t INPUT_NUM = 1;
constexpr size_t OUTPUT_NUM = 1;
constexpr size_t SHAPE_SIZE = 4;
constexpr size_t CROPS_SHAPE_0 = 2;
constexpr size_t CROPS_SHAPE_1 = 2;

struct BatchToSpaceAttr : public GpuKernelAttrBase {
  std::vector<std::vector<int64_t>> crops;
  std::vector<size_t> input_shape;
  size_t block_size;
};

template <typename T>
class BatchToSpaceHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit BatchToSpaceHelperGpuKernel(std::string &kernel_name) : GpuKernelHelperBase(kernel_name) {}
  virtual ~BatchToSpaceHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<size_t>> &input_shapes,
                 const std::vector<std::vector<size_t>> &output_shapes) override {
    int flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (flag != 0) {
      return flag;
    }
    flag = CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (flag != 0) {
      return flag;
    }
    kernel_size_ = output_size_list_[0] / sizeof(T);
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    size_t in = attr_ptr_->input_shape[0];
    size_t ic = attr_ptr_->input_shape[1];
    size_t ih = attr_ptr_->input_shape[2];
    size_t iw = attr_ptr_->input_shape[3];

    size_t on = in / (attr_ptr_->block_size * attr_ptr_->block_size);
    size_t oc = ic;
    size_t oh = ih * attr_ptr_->block_size - attr_ptr_->crops[0][0] - attr_ptr_->crops[0][1];
    size_t ow = iw * attr_ptr_->block_size - attr_ptr_->crops[1][0] - attr_ptr_->crops[1][1];

    T *input_ptr = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    CalBatchToSpace<T>(kernel_size_, input_ptr, in, ih, iw, ic, on, oh, ow, oc, attr_ptr_->crops[0][0],
                       attr_ptr_->crops[0][1], attr_ptr_->crops[1][0], attr_ptr_->crops[1][1], attr_ptr_->block_size,
                       output_ptr, reinterpret_cast<cudaStream_t>(cuda_stream));

    return 0;
  }

  void ResetResource() override {
    kernel_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
  }
  int CheckKernelParam(GpuKernelAttrBase *kernel_attr) override {
    attr_ptr_ = dynamic_cast<BatchToSpaceAttr *>(kernel_attr);
    if (attr_ptr_->block_size < 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'block_size' cannot be less than 1, but got "
                    << attr_ptr_->block_size;
      return -1;
    }

    // check input_shape
    if (attr_ptr_->input_shape.size() != SHAPE_SIZE) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input should be 4, but got "
                    << attr_ptr_->input_shape.size();
      return -1;
    }
    if ((attr_ptr_->input_shape[0] % (attr_ptr_->block_size * attr_ptr_->block_size)) != 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', input_shape[0] should be divisible by product of block_shape, but got input_shape[0]: "
                    << attr_ptr_->input_shape[0] << ", block_shape: " << attr_ptr_->block_size;
      return -1;
    }
    for (size_t idx = 0; idx < SHAPE_SIZE; ++idx) {
      if (attr_ptr_->input_shape[idx] < 1) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the element of shape of input cannot be less than 1, but got "
                      << ConvertVectorToString(attr_ptr_->input_shape);
        return -1;
      }
    }

    // check crops
    if (attr_ptr_->crops.size() != CROPS_SHAPE_0) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the size of 'crops' should be " << CROPS_SHAPE_0 << ", but got "
                    << attr_ptr_->crops.size();
      return -1;
    }
    if (attr_ptr_->crops[0].size() != CROPS_SHAPE_1 || attr_ptr_->crops[1].size() != CROPS_SHAPE_1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the size of element of 'crops' should be " << CROPS_SHAPE_1
                    << ", but got the size of crops[0]: " << attr_ptr_->crops[0].size()
                    << ", the size of crops[1]: " << attr_ptr_->crops[1].size();
      return -1;
    } else {
      for (size_t idx_i = 0; idx_i < CROPS_SHAPE_0; ++idx_i) {
        for (size_t idx_j = 0; idx_j < CROPS_SHAPE_1; ++idx_j) {
          if (attr_ptr_->crops[idx_i][idx_j] < 0) {
            MS_LOG(ERROR) << "For '" << kernel_name_
                          << "', the element of 'crops' should be greater than or equal to 0, but got crops[" << idx_i
                          << "][" << idx_j << "]: " << attr_ptr_->crops[idx_i][idx_j];
            return -1;
          }
        }
        auto tmp_shape = attr_ptr_->input_shape[idx_i + CROPS_SHAPE_1] * attr_ptr_->block_size -
                         attr_ptr_->crops[idx_i][0] - attr_ptr_->crops[idx_i][1];
        if (tmp_shape <= 0) {
          MS_LOG(ERROR) << "For '" << kernel_name_
                        << "', the element of shape of output should be greater than 0, but got " << tmp_shape;
          return -1;
        }
      }
    }
    return 0;
  }

 private:
  BatchToSpaceAttr *attr_ptr_;
  size_t kernel_size_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_BATCHTOSPACE_HELPER_H_
