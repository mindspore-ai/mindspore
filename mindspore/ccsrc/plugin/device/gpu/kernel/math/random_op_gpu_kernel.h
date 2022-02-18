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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_RANDOM_OP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_RANDOM_OP_GPU_KERNEL_H_

#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_op_impl.cuh"
#include "include/curand.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
enum RandomOptype {
  RANDOM_OP_NORMAL = 0,
  RANDOM_OP_UNIFORM_INT,
  RANDOM_OP_UNIFORM_REAL,
  RANDOM_OP_CUDNN_UNIFORM_REAL,
  RANDOM_OP_INVALID_TYPE = 255
};

const std::map<std::string, RandomOptype> kRandomOpTypeMap = {{"StandardNormal", RANDOM_OP_NORMAL},
                                                              {"UniformInt", RANDOM_OP_UNIFORM_INT},
                                                              {"UniformReal", RANDOM_OP_UNIFORM_REAL},
                                                              {"CudnnUniformReal", RANDOM_OP_CUDNN_UNIFORM_REAL}};

template <typename T>
class RandomOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  RandomOpGpuKernelMod()
      : random_op_type_(RANDOM_OP_INVALID_TYPE),
        input_size_0_(sizeof(int32_t)),
        input_size_1_(sizeof(T)),
        input_size_2_(sizeof(T)),
        output_size_(sizeof(T)),
        workspace_size_(sizeof(curandState)),
        seed_(0),
        seed2_(0),
        mask_generator_(nullptr),
        states_init_(false),
        is_null_input_(false) {}
  ~RandomOpGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    curandState *devStates = nullptr;
    // Operator CudnnUniformReal does not need workspace memory.
    if (random_op_type_ != RANDOM_OP_CUDNN_UNIFORM_REAL) {
      void *workspace_addr = GetDeviceAddress<void *>(workspace, 0);
      devStates = reinterpret_cast<curandState *>(workspace_addr);
    }
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    switch (random_op_type_) {
      case RANDOM_OP_NORMAL: {
        StandardNormal(seed_, seed2_, devStates, output_addr, outputs[0]->size / sizeof(T),
                       reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case RANDOM_OP_UNIFORM_INT: {
        T *input_addr_1 = GetDeviceAddress<T>(inputs, 1);
        T *input_addr_2 = GetDeviceAddress<T>(inputs, 2);
        bool ret = UniformInt(seed_, seed2_, devStates, input_addr_1, inputs[1]->size / sizeof(T), input_addr_2,
                              inputs[2]->size / sizeof(T), output_addr, outputs[0]->size / sizeof(T),
                              reinterpret_cast<cudaStream_t>(stream_ptr));
        if (!ret) {
          MS_LOG(ERROR) << "For '" << kernel_name_ << "', `minval` should be strictly less than `maxval`";
          return false;
        }
        break;
      }
      case RANDOM_OP_UNIFORM_REAL: {
        UniformReal(seed_, seed2_, devStates, output_addr, outputs[0]->size / sizeof(T),
                    reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      }
      case RANDOM_OP_CUDNN_UNIFORM_REAL: {
        float *mask_f = GetDeviceAddress<float>(outputs, 0);
        if (!states_init_) {
          CHECK_CURAND_RET_WITH_EXCEPT(curandCreateGenerator(&mask_generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10),
                                       "Failed to create generator");
          CHECK_CURAND_RET_WITH_EXCEPT(curandSetPseudoRandomGeneratorSeed(mask_generator_, seed_),
                                       "Failed to SetPseudoRandomGeneratorSeed");
          MS_EXCEPTION_IF_NULL(mask_generator_);
          states_init_ = true;
        }
        CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(mask_generator_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                     "Failed to set stream for generator");
        // curandGen only support float or double for mask.
        CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateUniform(mask_generator_, mask_f, outputs[0]->size / sizeof(float)),
                                     "Failed to generate uniform");
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: StandardNormal, CudnnUniformReal, "
                          << "UniformInt, UniformReal currently, but got " << random_op_type_;
      }
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kRandomOpTypeMap.find(kernel_name);
    if (iter == kRandomOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << ", only support these types: StandardNormal, CudnnUniformReal, "
                        << "UniformInt, UniformReal currently, but got " << kernel_name;
    } else {
      random_op_type_ = iter->second;
    }
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if ((random_op_type_ == RANDOM_OP_NORMAL || random_op_type_ == RANDOM_OP_UNIFORM_REAL) && input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }
    if (random_op_type_ == RANDOM_OP_UNIFORM_INT && input_num != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 3, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape_0 = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape_0, kernel_name, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < input_shape_0.size(); i++) {
      input_size_0_ *= input_shape_0[i];
    }
    input_size_0_ *= sizeof(int);
    if (random_op_type_ == RANDOM_OP_UNIFORM_INT) {
      input_size_1_ *= 1;
      input_size_2_ *= 1;
    }

    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
      workspace_size_ *= output_shape[i];
    }
    // Operator CudnnUniformReal does not need workspace memory.
    if (random_op_type_ == RANDOM_OP_CUDNN_UNIFORM_REAL) {
      workspace_size_ = 0;
    }

    auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    seed_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("seed")));
    seed2_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("seed2")));
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_0_);
    if (random_op_type_ == RANDOM_OP_UNIFORM_INT) {
      input_size_list_.push_back(input_size_1_);
      input_size_list_.push_back(input_size_2_);
    }
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  RandomOptype random_op_type_;
  size_t input_size_0_;
  size_t input_size_1_;
  size_t input_size_2_;
  size_t output_size_;
  size_t workspace_size_;
  int seed_;
  int seed2_;

  curandGenerator_t mask_generator_;
  bool states_init_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_RANDOM_OP_GPU_KERNEL_H_
