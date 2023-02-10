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

#include "plugin/device/cpu/kernel/ger_cpu_kernel.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <utility>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/matmul_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kGerInputsNum = 2;
const size_t kGerOutputsNum = 1;
const size_t kNoBatchNum = 1;
}  // namespace

bool GerCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                           const std::vector<KernelTensorPtr> &outputs) {
  if (!base_operator) {
    MS_LOG(ERROR) << "For " << kernel_type_ << ", cast " << kernel_type_ << " ops failed!";
    return false;
  }
  kernel_name_ = base_operator->name();
  if (inputs.size() != kGerInputsNum || outputs.size() != kGerOutputsNum) {
    MS_LOG(ERROR) << "For" << kernel_name_ << ": input and output size should be " << kGerInputsNum << " and "
                  << kGerOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }
  input_type_1_ = inputs[0]->GetDtype();
  input_type_2_ = inputs[1]->GetDtype();
  if (input_type_1_ != input_type_2_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input1 and input2 must have the same type. But got input1 type "
                  << input_type_1_ << ", input2 type " << input_type_2_;
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int GerCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs,
                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }

  input_shape_1_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  input_shape_2_ = std::vector<size_t>(inputs.at(kIndex1)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex1)->GetDeviceShapeAdaptively().end());
  output_shape_ = std::vector<size_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                      outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  auto in_shape_size_1 = input_shape_1_.size();
  auto in_shape_size_2 = input_shape_2_.size();

  if (input_shape_1_.size() > max_dims_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of input should be less than or equal to max_dims 7, but got "
                  << input_shape_1_.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  if (in_shape_size_1 != in_shape_size_2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input1 shape size should be the same as input2 shape size, but got"
                  << " input1 shape size " << in_shape_size_1 << " input2 shape size " << in_shape_size_2;
    return KRET_RESIZE_FAILED;
  }

  for (size_t shape_index = 0; shape_index < input_shape_1_.size() - 1; shape_index++) {
    if (input_shape_1_[shape_index] != input_shape_2_[shape_index]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the " << shape_index << "th dimension of shape size of input1 and"
                    << " input2 should be the same, but got input1 shape size " << input_shape_1_[shape_index]
                    << " input2 shape size " << input_shape_2_[shape_index];
      return KRET_RESIZE_FAILED;
    }
    batches_ *= input_shape_1_[shape_index];
  }

  in1dim_ = input_shape_1_[input_shape_1_.size() - 1];
  in2dim_ = input_shape_2_[input_shape_2_.size() - 1];
  outdim_ = in1dim_ * in2dim_;

#ifdef __APPLE__
  if (input_type_1_ == kNumberTypeFloat64 && batches_ != kNoBatchNum) {
    launch_func_ = &GerCpuKernelMod::LaunchMacBatches<double>;
  } else if (input_type_1_ == kNumberTypeFloat64) {
    launch_func_ = &GerCpuKernelMod::LaunchMacNoBatches<double>;
  } else if (input_type_1_ == kNumberTypeFloat32 && batches_ != kNoBatchNum) {
    launch_func_ = &GerCpuKernelMod::LaunchMacBatches<float>;
  } else if (input_type_1_ == kNumberTypeFloat32) {
    launch_func_ = &GerCpuKernelMod::LaunchMacNoBatches<float>;
  } else {
    MS_LOG(ERROR) << "Ger kernel does not support " << TypeIdToString(input_type_1_);
    return KRET_RESIZE_FAILED;
  }
#else
  if (input_type_1_ == kNumberTypeFloat64) {
    InitLaunchFunc<double>();
  } else if (input_type_1_ == kNumberTypeFloat32) {
    if (batches_ != kNoBatchNum) {
      launch_func_ = &GerCpuKernelMod::LaunchBatches;
    } else {
      launch_func_ = &GerCpuKernelMod::LaunchNoBatches;
    }
  } else {
    MS_LOG(ERROR) << "Ger kernel does not support " << TypeIdToString(input_type_1_);
    return KRET_RESIZE_FAILED;
  }
#endif
  return KRET_OK;
}

template <typename T>
void GerCpuKernelMod::InitLaunchFunc() {
  // get float space float_input1, float_input2, float_output
  workspace_size_list_.push_back(batches_ * in1dim_ * sizeof(float));
  workspace_size_list_.push_back(batches_ * in2dim_ * sizeof(float));
  workspace_size_list_.push_back(batches_ * outdim_ * sizeof(float));
  if (batches_ != kNoBatchNum) {
    launch_func_ = &GerCpuKernelMod::LaunchBatchesElse<T>;
  } else {
    launch_func_ = &GerCpuKernelMod::LaunchNoBatchesElse<T>;
  }
}

template <typename T>
bool GerCpuKernelMod::LaunchBatchesElse(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &workspace,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input1 = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  const auto *input2 = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  float *float_input1 = reinterpret_cast<float *>(workspace[kIndex0]->addr);
  float *float_input2 = reinterpret_cast<float *>(workspace[kIndex1]->addr);
  float *float_output = reinterpret_cast<float *>(workspace[kIndex2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  auto task = [this, &float_input1, &float_input2, &input1, &input2, &output, &float_output](size_t start, size_t end) {
    for (size_t i = start * this->in1dim_; i < end * this->in1dim_; ++i) {
      float_input1[i] = static_cast<float>(input1[i]);
    }
    for (size_t i = start * this->in2dim_; i < end * this->in2dim_; ++i) {
      float_input2[i] = static_cast<float>(input2[i]);
    }

    for (size_t i = start; i < end; i++) {
      MatMulOpt(float_input1 + i * this->in1dim_, float_input2 + i * this->in2dim_, float_output + i * this->outdim_,
                nullptr, ActType_No, 1, static_cast<int>(this->in1dim_), static_cast<int>(this->in2dim_), this->in2dim_,
                1);
    }

    for (size_t i = start * this->outdim_; i < end * this->outdim_; ++i) {
      output[i] = static_cast<T>(float_output[i]);
    }
  };
  ParallelLaunchAutoSearch(task, batches_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool GerCpuKernelMod::LaunchNoBatchesElse(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &workspace,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  const auto *input1 = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  const auto *input2 = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  float *float_input1 = reinterpret_cast<float *>(workspace[kIndex0]->addr);
  float *float_input2 = reinterpret_cast<float *>(workspace[kIndex1]->addr);
  float *float_output = reinterpret_cast<float *>(workspace[kIndex2]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);

  for (size_t i = 0; i < ((inputs[kIndex0]->size) / sizeof(T)); ++i) {
    float_input1[i] = static_cast<float>(input1[i]);
  }
  for (size_t i = 0; i < ((inputs[kIndex1]->size) / sizeof(T)); ++i) {
    float_input2[i] = static_cast<float>(input2[i]);
  }

  auto task = [this, &float_input1, &float_input2, &float_output, &output](size_t start, size_t end) {
    MatMulOpt(float_input1 + start, float_input2, float_output + start * this->in2dim_, nullptr, ActType_No, 1,
              static_cast<int>(end - start), static_cast<int>(this->in2dim_), this->in2dim_, 1);

    for (size_t i = start * this->in2dim_; i < end * this->in2dim_; ++i) {
      output[i] = static_cast<T>(float_output[i]);
    }
  };
  ParallelLaunchAutoSearch(task, in1dim_, this, &parallel_search_info_);
  return true;
}

bool GerCpuKernelMod::LaunchBatches(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  float *input1 = reinterpret_cast<float *>(inputs[kIndex0]->addr);
  float *input2 = reinterpret_cast<float *>(inputs[kIndex1]->addr);
  float *output = reinterpret_cast<float *>(outputs[kIndex0]->addr);
  auto task = [this, &input1, &input2, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      MatMulOpt(input1 + i * this->in1dim_, input2 + i * this->in2dim_, output + i * this->outdim_, nullptr, ActType_No,
                1, static_cast<int>(this->in1dim_), static_cast<int>(this->in2dim_), this->in2dim_, 1);
    }
  };
  ParallelLaunchAutoSearch(task, batches_, this, &parallel_search_info_);
  return true;
}

bool GerCpuKernelMod::LaunchNoBatches(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  float *input1 = reinterpret_cast<float *>(inputs[kIndex0]->addr);
  float *input2 = reinterpret_cast<float *>(inputs[kIndex1]->addr);
  float *output = reinterpret_cast<float *>(outputs[kIndex0]->addr);
  auto task = [this, &input1, &input2, &output](size_t start, size_t end) {
    MatMulOpt(input1 + start, input2, output + start * this->in2dim_, nullptr, ActType_No, 1,
              static_cast<int>(end - start), static_cast<int>(this->in2dim_), this->in2dim_, 1);
  };
  ParallelLaunchAutoSearch(task, in1dim_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool GerCpuKernelMod::LaunchMacBatches(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  T *input1 = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  T *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto task = [this, &input1, &input2, &output](size_t start, size_t end) {
    for (size_t batch_index = start; batch_index < end; batch_index++) {
      size_t row_i_s = batch_index * this->in1dim_;
      size_t col_i_s = batch_index * this->in2dim_;
      size_t out_i_s = batch_index * this->in1dim_ * this->in2dim_;
      for (size_t row_i = 0; row_i < this->in1dim_; row_i++) {
        T in_one = input1[row_i_s + row_i];
        size_t out_i_i_s = out_i_s + row_i * this->in2dim_;
        for (size_t col_i = 0; col_i < this->in2dim_; col_i++) {
          output[out_i_i_s + col_i] = in_one * input2[col_i_s + col_i];
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, batches_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool GerCpuKernelMod::LaunchMacNoBatches(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  T *input1 = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  T *input2 = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  T *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  auto task = [this, &input1, &input2, &output](size_t start, size_t end) {
    for (size_t row = start; row < end; row++) {
      T in_one = input1[row];
      size_t row_i_s = row * this->in2dim_;
      for (size_t col = 0; col < this->in2dim_; col++) {
        output[row_i_s + col] = in_one * input2[col];
      }
    }
  };
  ParallelLaunchAutoSearch(task, in1dim_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool GerCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<AddressPtr> &workspace,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGerInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGerOutputsNum, kernel_name_);
  return launch_func_(this, inputs, workspace, outputs);
}

const std::vector<std::pair<KernelAttr, GerCpuKernelMod::KernelRunFunc>> &GerCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, GerCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &GerCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &GerCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Ger,
                                 []() { return std::make_shared<GerCpuKernelMod>(prim::kPrimGer->name()); });
}  // namespace kernel
}  // namespace mindspore
