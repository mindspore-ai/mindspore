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

#include "plugin/device/gpu/kernel/math/random_op_gpu_kernel.h"
#include <algorithm>
#include <memory>
namespace mindspore {
namespace kernel {
const size_t UNIFORM_INT_INPUT_NUM = 3;
bool RandomOpGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  kernel_type_ = base_operator->name();
  auto iter = kRandomOpTypeMap.find(kernel_type_);
  if (iter == kRandomOpTypeMap.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_type_
                      << ", only support these types: StandardNormal, UniformInt or UniformReal currently, but got "
                      << kernel_type_;
  } else {
    random_op_type_ = iter->second;
  }
  if (random_op_type_ == RANDOM_OP_UNIFORM_INT) {
    input_num_ = UNIFORM_INT_INPUT_NUM;
  } else {
    input_num_ = 1;
  }
  seed_ = LongToInt(GetValue<int64_t>(base_operator->GetAttr("seed")));
  seed2_ = LongToInt(GetValue<int64_t>(base_operator->GetAttr("seed2")));
  if (base_operator->HasAttr("use_curand")) {
    use_curand_ = GetValue<bool>(base_operator->GetAttr("use_curand"));
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_type_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_type_)[index].second;
  return true;
}

int RandomOpGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  auto real_input_num = inputs.size();
  if (real_input_num != input_num_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_type_ << "', the number of inputs should be " << input_num_ << ", but got "
                      << real_input_num;
  }
  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_type_ << "', the number of outputs should be 1, but got " << output_num;
  }
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  workspace_size_list_.clear();
  if (random_op_type_ != RANDOM_OP_CUDNN_UNIFORM_REAL) {
    auto output_shape = outputs[0]->GetShapeVector();
    auto workspace_size = sizeof(curandStatePhilox4_32_10_t);
    for (size_t i = 0; i < output_shape.size(); i++) {
      workspace_size *= output_shape[i];
    }
    workspace_size_list_.push_back(workspace_size);
  }
  return KRET_OK;
}

template <typename T>
bool RandomOpGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  curandStatePhilox4_32_10_t *devStates = nullptr;
  // Operator CudnnUniformReal does not need workspace memory.
  if (random_op_type_ != RANDOM_OP_CUDNN_UNIFORM_REAL) {
    void *workspace_addr = GetDeviceAddress<void *>(workspace, 0);
    devStates = reinterpret_cast<curandStatePhilox4_32_10_t *>(workspace_addr);
  }
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  switch (random_op_type_) {
    case RANDOM_OP_NORMAL: {
      if (use_curand_) {
        float *mask_f = GetDeviceAddress<float>(outputs, 0);
        if (!states_init_) {
          int RNG_seed = 0;
          std::random_device rd;
          if (seed2_ != 0) {
            RNG_seed = seed2_;
          } else if (seed_ != 0) {
            RNG_seed = seed_;
          } else {
            RNG_seed = static_cast<int>(rd());
          }
          CHECK_CURAND_RET_WITH_EXCEPT(curandCreateGenerator(&mask_generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10),
                                       "Failed to create generator");
          CHECK_CURAND_RET_WITH_EXCEPT(curandSetPseudoRandomGeneratorSeed(mask_generator_, RNG_seed),
                                       "Failed to SetPseudoRandomGeneratorSeed");
          MS_EXCEPTION_IF_NULL(mask_generator_);
          states_init_ = true;
        }
        CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(mask_generator_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                     "Failed to set stream for generator");
        // curandGen only support float or double for mask.
        CHECK_CURAND_RET_WITH_EXCEPT(
          curandGenerateNormal(mask_generator_, mask_f, outputs[0]->size / sizeof(float), 0.0, 1.0),
          "Failed to generate normal");
      } else {
        StandardNormal(seed_, seed2_, devStates, output_addr, outputs[0]->size / sizeof(T),
                       reinterpret_cast<cudaStream_t>(cuda_stream_));
      }
      break;
    }
    case RANDOM_OP_UNIFORM_INT: {
      T *input_addr_1 = GetDeviceAddress<T>(inputs, 1);
      T *input_addr_2 = GetDeviceAddress<T>(inputs, 2);
      bool ret = UniformInt(seed_, seed2_, devStates, input_addr_1, inputs[1]->size / sizeof(T), input_addr_2,
                            inputs[2]->size / sizeof(T), output_addr, outputs[0]->size / sizeof(T),
                            reinterpret_cast<cudaStream_t>(cuda_stream_));
      if (!ret) {
        MS_LOG(ERROR) << "For '" << kernel_type_ << "', `minval` should be strictly less than `maxval`";
        return false;
      }
      break;
    }
    case RANDOM_OP_UNIFORM_REAL: {
      UniformReal(seed_, seed2_, devStates, output_addr, outputs[0]->size / sizeof(T),
                  reinterpret_cast<cudaStream_t>(cuda_stream_));
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
      CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(mask_generator_, reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                   "Failed to set stream for generator");
      // curandGen only support float or double for mask.
      CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateUniform(mask_generator_, mask_f, outputs[0]->size / sizeof(float)),
                                   "Failed to generate uniform");
      break;
    }
    default: {
      MS_LOG(EXCEPTION) << "For '" << kernel_type_ << ", only support these types: StandardNormal, CudnnUniformReal, "
                        << "UniformInt, UniformReal currently, but got " << random_op_type_;
    }
  }
  return true;
}

std::map<std::string, std::vector<std::pair<KernelAttr, RandomOpGpuKernelMod::OpFunc>>>
  RandomOpGpuKernelMod::kernel_attr_map_ = {
    {"StandardNormal",
     {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
       &RandomOpGpuKernelMod::LaunchKernel<float>}}},
    {"UniformReal",
     {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
       &RandomOpGpuKernelMod::LaunchKernel<float>}}},
    {"CudnnUniformReal",
     {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
       &RandomOpGpuKernelMod::LaunchKernel<float>}}},
    {"UniformInt",
     {{KernelAttr()
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &RandomOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeInt64)
         .AddInputAttr(kNumberTypeInt32)
         .AddInputAttr(kNumberTypeInt32)
         .AddOutputAttr(kNumberTypeInt32),
       &RandomOpGpuKernelMod::LaunchKernel<int>}}}};

std::vector<KernelAttr> RandomOpGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_type_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'Random op', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, RandomOpGpuKernelMod::OpFunc>>>(
                       kernel_attr_map_)
                  << ", but got " << kernel_type_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, OpFunc> &item) { return item.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, StandardNormal,
                                 []() { return std::make_shared<RandomOpGpuKernelMod>("StandardNormal"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, UniformInt,
                                 []() { return std::make_shared<RandomOpGpuKernelMod>("UniformInt"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, UniformReal,
                                 []() { return std::make_shared<RandomOpGpuKernelMod>("UniformReal"); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, CudnnUniformReal,
                                 []() { return std::make_shared<RandomOpGpuKernelMod>("CudnnUniformReal"); });
}  // namespace kernel
}  // namespace mindspore
