/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/reduce_cpu_kernel.h"
#include <complex>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <map>
#include "nnacl/fp32/reduce_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/check_convert_utils.h"
#include "ops/reduce.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kReduceMean = "ReduceMean";
constexpr auto kReduceMax = "ReduceMax";
constexpr auto kReduceSum = "ReduceSum";
constexpr auto kReduceMin = "ReduceMin";
constexpr auto kReduceProd = "ReduceProd";
constexpr auto kReduceAll = "ReduceAll";
constexpr auto kReduceAny = "ReduceAny";

constexpr size_t kReduceSmallVectorSize = 200000;
constexpr size_t kReduceOutputsNum = 1;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T>
class ReduceCpuKernelFunc : public CpuKernelFunc {
 public:
  ReduceCpuKernelFunc() = default;
  ~ReduceCpuKernelFunc() override = default;
  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

 private:
  void AccelerateLongVector(T *input_addr, T *output_addr, size_t input_size);
  void ChooseFunc(const std::string &kernel_name_);
  void HandleInputAxis();
  void SpecialExcute();
  void CalAxesAndStride(std::vector<size_t> *axes, size_t *stride);

  enum class ReduceFuncType {
    kReduceAllType,
    kReduceAnyType,
    kReduceMaxType,
    kReduceMinType,
    kReduceSumType,
    kReduceMeanType,
    kReduceProdType
  };
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> axis_;
  static constexpr size_t kAxisIndex_{1};
  ReduceFuncType reduce_type_{ReduceFuncType::kReduceAllType};
  std::function<void(const T *, T *, size_t, size_t, TransposeIterator *)> reduce_func_;
  bool simple_execute_{false};
  std::string kernel_name_;
  bool need_skip_execute_{false};
  bool skip_mode_{false};
};

template <typename T>
void ReduceSum(const T *in, T *out, size_t start, size_t end, TransposeIterator *iter) {
  // float is prone to cumulative error, use double instead of float.
  constexpr bool is_float = std::is_same<T, float>::value;
  auto value = is_float ? static_cast<double>(0.f) : static_cast<T>(0);
  value += *out;
  if (iter != nullptr) {
    for (size_t i = start; i < end; i++) {
      value += in[iter->GetPos()];
      iter->GenNextPos();
    }
  } else {
    for (size_t i = start; i < end; i++) {
      value += in[i];
    }
  }
  *out = value;
}

template <typename T>
void ReduceMean(const T *in, T *out, size_t start, size_t end, TransposeIterator *iter) {
  ReduceSum(in, out, start, end, iter);
}

template <typename T>
void ReduceProd(const T *in, T *out, size_t start, size_t end, TransposeIterator *iter) {
  if (iter != nullptr) {
    for (size_t i = start; i < end; i++) {
      *out *= in[iter->GetPos()];
      iter->GenNextPos();
    }
    return;
  }

  for (size_t i = start; i < end; i++) {
    *out *= in[i];
  }
}

template <typename T>
void ReduceMax(const T *in, T *out, size_t start, size_t end, TransposeIterator *iter) {
  if (iter != nullptr) {
    for (size_t i = start; i < end; i++) {
      *out = std::max(*out, in[iter->GetPos()]);
      iter->GenNextPos();
    }
    return;
  }

  for (size_t i = start; i < end; i++) {
    *out = std::max(*out, in[i]);
  }
}

template <typename T>
void ReduceMin(const T *in, T *out, size_t start, size_t end, TransposeIterator *iter) {
  if (iter != nullptr) {
    for (size_t i = start; i < end; i++) {
      *out = std::min(*out, in[iter->GetPos()]);
      iter->GenNextPos();
    }
    return;
  }

  for (size_t i = start; i < end; i++) {
    *out = std::min(*out, in[i]);
  }
}

template <typename T>
void ReduceAll(const T *in, T *out, size_t start, size_t end, TransposeIterator *iter) {
  if (iter != nullptr) {
    for (size_t i = start; i < end; i++) {
      *out = *out && in[iter->GetPos()];
      iter->GenNextPos();
    }
    return;
  }

  for (size_t i = start; i < end; i++) {
    *out = *out && in[i];
  }
}

template <typename T>
void ReduceAny(const T *in, T *out, size_t start, size_t end, TransposeIterator *iter) {
  if (iter != nullptr) {
    for (size_t i = start; i < end; i++) {
      *out = *out || in[iter->GetPos()];
      iter->GenNextPos();
    }
    return;
  }

  for (size_t i = start; i < end; i++) {
    *out = *out || in[i];
  }
}

template <typename T>
void ReduceCpuKernelFunc<T>::SpecialExcute() {
  // reset simple_execute_
  simple_execute_ = false;
  // special accelerate for axis = 1 and input has 2 dims
  if ((reduce_type_ == ReduceFuncType::kReduceMeanType || reduce_type_ == ReduceFuncType::kReduceSumType) &&
      axis_.size() == 1 && axis_[0] == 1 && input_shape_.size() == kDim2) {
    simple_execute_ = true;
  }
  // special accelerate for axis[0] = 0 and other dims for axis is 1.
  if (reduce_type_ == ReduceFuncType::kReduceSumType && axis_.size() >= 1 && axis_[0] == 0 &&
      input_shape_.size() >= kDim2) {
    simple_execute_ = true;
    for (size_t i = 1; i < axis_.size(); ++i) {
      if (static_cast<int64_t>(input_shape_.size()) > axis_[i] && input_shape_[axis_[i]] != 1) {
        simple_execute_ = false;
        break;
      }
    }
  }
}

template <typename T>
void ReduceCpuKernelFunc<T>::HandleInputAxis() {
  int64_t dimension = SizeToLong(input_shape_.size());
  (void)std::for_each(axis_.begin(), axis_.end(), [dimension](auto &a) {
    if (dimension == 0) {
      if (a != -1 && a != 0) {
        MS_LOG(EXCEPTION) << "For reduce, the each axis element should be in [-1, 0], but got " << a;
      }
      a = 0;
    } else {
      if (a < -dimension || a >= dimension) {
        MS_LOG(EXCEPTION) << "For reduce, the each axis element should be in [" << -dimension << ", " << dimension
                          << "), but got " << a;
      }
      a = a < 0 ? dimension + a : a;
    }
  });

  // Delete the duplicate axis.
  sort(axis_.begin(), axis_.end());
  auto last = std::unique(axis_.begin(), axis_.end());
  axis_.erase(last, axis_.end());
  if constexpr (std::is_same<T, float>::value) {
    SpecialExcute();
  }
}

template <typename T>
int ReduceCpuKernelFunc<T>::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  input_shape_ = inputs[0]->GetDeviceShapeAdaptively();
  if (!TryGetIntValue(inputs, kIndex1, kernel_name_, &axis_, false)) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " can't get axis input! ";
  }
  if (inputs.size() > kAxisIndex_ &&
      AnfAlgo::IsDynamicShapeSkipExecute(skip_mode_, inputs[kAxisIndex_]->GetShapeVector())) {
    need_skip_execute_ = true;
  } else {
    need_skip_execute_ = false;
  }
  HandleInputAxis();
  return KRET_OK;
}

template <typename T>
void ReduceCpuKernelFunc<T>::ChooseFunc(const std::string &kernel_name_) {
  if constexpr (std::is_same<T, bool>::value) {
    if (kernel_name_ == kReduceAll) {
      reduce_type_ = ReduceFuncType::kReduceAllType;
      reduce_func_ = ReduceAll<T>;
    } else if (kernel_name_ == kReduceAny) {
      reduce_type_ = ReduceFuncType::kReduceAnyType;
      reduce_func_ = ReduceAny<T>;
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unsupported reduce operation for bool.";
    }
  } else if constexpr (((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>))) {  // NOLINT
    if (kernel_name_ == kReduceProd) {
      reduce_type_ = ReduceFuncType::kReduceProdType;
      reduce_func_ = ReduceProd<T>;
    } else if (kernel_name_ == kReduceMean) {
      reduce_type_ = ReduceFuncType::kReduceMeanType;
      reduce_func_ = ReduceMean<T>;
    } else if (kernel_name_ == kReduceSum) {
      reduce_type_ = ReduceFuncType::kReduceSumType;
      reduce_func_ = ReduceSum<T>;
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unsupported reduce operation for complex.";
    }
  } else {
    if (kernel_name_ == kReduceMax) {
      reduce_type_ = ReduceFuncType::kReduceMaxType;
      reduce_func_ = ReduceMax<T>;
    } else if (kernel_name_ == kReduceMin) {
      reduce_type_ = ReduceFuncType::kReduceMinType;
      reduce_func_ = ReduceMin<T>;
    } else if (kernel_name_ == kReduceSum) {
      reduce_type_ = ReduceFuncType::kReduceSumType;
      reduce_func_ = ReduceSum<T>;
    } else if (kernel_name_ == kReduceMean) {
      reduce_type_ = ReduceFuncType::kReduceMeanType;
      reduce_func_ = ReduceMean<T>;
    } else if (kernel_name_ == kReduceProd) {
      reduce_type_ = ReduceFuncType::kReduceProdType;
      reduce_func_ = ReduceProd<T>;
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unsupported reduce operation.";
    }
  }
}

template <typename T>
void ReduceCpuKernelFunc<T>::InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &,
                                      const std::vector<KernelTensorPtr> &) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  ChooseFunc(kernel_name_);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Reduce>(base_operator);
  skip_mode_ = kernel_ptr->get_skip_mode();
}

template <typename T>
void ReduceCpuKernelFunc<T>::CalAxesAndStride(std::vector<size_t> *axes, size_t *stride) {
  int dimension = SizeToInt(input_shape_.size());
  size_t j = 0;
  size_t k = 0;
  for (int i = 0; i < dimension; ++i) {
    if (j == axis_.size() || i != axis_[j]) {
      (*axes)[k] = IntToSize(i);
      ++k;
    } else {
      *stride *= LongToSize(input_shape_[IntToSize(i)]);
      ++j;
    }
  }
  for (auto &it : axis_) {
    (*axes)[k] = IntToSize(it);
    ++k;
  }
}

template <typename T>
bool ReduceCpuKernelFunc<T>::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReduceOutputsNum, kernel_name_);
  size_t input_size = inputs[0]->size / sizeof(T);
  auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (need_skip_execute_) {
    auto ret = memcpy_s(output_addr, outputs[0]->size, input_addr, inputs[0]->size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', launch kernel error: memcpy failed. Error no: " << ret;
    }
    return true;
  }

  if (axis_.empty() || input_shape_.empty() || input_shape_.size() == 1) {
    if (input_size < kReduceSmallVectorSize) {
      // Get one ret
      *output_addr = input_addr[0];
      reduce_func_(input_addr, output_addr, 1, input_size, nullptr);
      if (reduce_type_ == ReduceFuncType::kReduceMeanType) {
        *output_addr /= SizeToFloat(input_size);
      }
    } else {
      AccelerateLongVector(input_addr, output_addr, input_size);
    }
  } else {
    // Calculate transpose axes and stride
    size_t stride = 1;
    std::vector<size_t> axes(input_shape_.size());
    CalAxesAndStride(&axes, &stride);

    size_t output_size = outputs[0]->size / sizeof(T);
    if constexpr (std::is_same<T, float>::value) {
      if (simple_execute_) {
        if (axis_[0] == 1) {
          auto task = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
              (void)ReduceSumDim2Axis1(stride, input_addr + i * stride, output_addr + i);
              if (reduce_type_ == ReduceFuncType::kReduceMeanType) {
                output_addr[i] /= SizeToFloat(stride);
              }
            }
          };
          ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
          return true;
        } else {
          auto task = [&](size_t start, size_t end) {
            int ret =
              ReduceSumDim2Axis0(end - start, output_size, input_shape_[0], input_addr + start, output_addr + start);
            if (ret != NNACL_OK) {
              MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', ReduceSumDim2Axis0 failed.Error no: " << ret;
            }
          };
          ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
          return true;
        }
      }
    }

    // Calculate transpose shape
    std::vector<int64_t> transpose_shape(input_shape_.size());
    int dimension = SizeToInt(input_shape_.size());
    for (int i = 0; i < dimension; ++i) {
      transpose_shape[i] = input_shape_[axes[i]];
    }
    TransposeIterator base_iter(std::move(transpose_shape), std::move(axes), input_shape_);
    auto task = [this, &base_iter, input_addr, output_addr, stride](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start * stride);
      for (size_t i = start; i < end; ++i) {
        output_addr[i] = input_addr[iter.GetPos()];
        iter.GenNextPos();
        reduce_func_(input_addr, &output_addr[i], 1, stride, &iter);
        if (reduce_type_ == ReduceFuncType::kReduceMeanType) {
          output_addr[i] /= SizeToFloat(stride);
        }
      }
    };
    ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  }
  return true;
}

template <typename T>
void ReduceCpuKernelFunc<T>::AccelerateLongVector(T *input_addr, T *output_addr, size_t input_size) {
  // init output_addr
  *output_addr = input_addr[0];
  std::mutex task_mutex;
  auto task = [this, input_addr, output_addr, &task_mutex](size_t start, size_t end) {
    if (start == 0) {
      ++start;
    }
    if (start == end) {
      return;
    }
    auto block_output = input_addr[start];
    reduce_func_(input_addr, &block_output, start + 1, end, nullptr);
    {
      std::lock_guard<std::mutex> task_lock(task_mutex);
      reduce_func_(&block_output, output_addr, 0, 1, nullptr);
    }
  };
  ParallelLaunchAutoSearch(task, input_size, this, &parallel_search_info_);
  if (reduce_type_ == ReduceFuncType::kReduceMeanType) {
    *output_addr /= SizeToFloat(input_size);
  }
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeReduceFunc() {
  return std::make_shared<ReduceCpuKernelFunc<T>>();
}
using SpecializeReduceFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
#define REDUCE_CPU_REG(MS_T, MS_S, T) \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddOutputAttr(MS_T), SpecializeReduceFunc<T>
static std::vector<std::pair<KernelAttr, SpecializeReduceFuncCreator>> kernel_all_any_list = {
  {REDUCE_CPU_REG(kNumberTypeBool, kNumberTypeInt32, bool)}, {REDUCE_CPU_REG(kNumberTypeBool, kNumberTypeInt64, bool)}};
static std::vector<std::pair<KernelAttr, SpecializeReduceFuncCreator>> kernel_max_min_list = {
  {REDUCE_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {REDUCE_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {REDUCE_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {REDUCE_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {REDUCE_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t)},
  {REDUCE_CPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t)},
  {REDUCE_CPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t)},
  {REDUCE_CPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t)},
  {REDUCE_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {REDUCE_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t)},
  {REDUCE_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t)},
  {REDUCE_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt32, kNumberTypeInt32, uint32_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt64, kNumberTypeInt32, uint64_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t)}};
static std::vector<std::pair<KernelAttr, SpecializeReduceFuncCreator>> kernel_sum_prod_mean_list = {
  {REDUCE_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
  {REDUCE_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
  {REDUCE_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
  {REDUCE_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
  {REDUCE_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t)},
  {REDUCE_CPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t)},
  {REDUCE_CPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t)},
  {REDUCE_CPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t)},
  {REDUCE_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t)},
  {REDUCE_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t)},
  {REDUCE_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t)},
  {REDUCE_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt32, kNumberTypeInt32, uint32_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt64, kNumberTypeInt32, uint64_t)},
  {REDUCE_CPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t)},
  {REDUCE_CPU_REG(kNumberTypeComplex64, kNumberTypeInt32, complex64)},
  {REDUCE_CPU_REG(kNumberTypeComplex64, kNumberTypeInt64, complex64)},
  {REDUCE_CPU_REG(kNumberTypeComplex128, kNumberTypeInt32, complex128)},
  {REDUCE_CPU_REG(kNumberTypeComplex128, kNumberTypeInt64, complex128)}};
static std::map<std::string, std::vector<std::pair<KernelAttr, SpecializeReduceFuncCreator>>> kernel_attr_list = {
  {prim::kPrimReduceSum->name(), kernel_sum_prod_mean_list},
  {prim::kPrimReduceMean->name(), kernel_sum_prod_mean_list},
  {prim::kPrimReduceProd->name(), kernel_sum_prod_mean_list},
  {prim::kPrimReduceMax->name(), kernel_max_min_list},
  {prim::kPrimReduceMin->name(), kernel_max_min_list},
  {prim::kPrimReduceAll->name(), kernel_all_any_list},
  {prim::kPrimReduceAny->name(), kernel_all_any_list}};
}  // namespace

std::vector<KernelAttr> ReduceCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_list.find(kernel_type_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(ERROR) << "For 'Reduce ops', it does not support " << kernel_type_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SpecializeReduceFuncCreator> &pair) { return pair.first; });
  return support_list;
}

bool ReduceCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Suppose to be " << kernel_type_ << " but got " << kernel_name_;
  }

  auto iter = kernel_attr_list.find(kernel_type_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(EXCEPTION) << "Reduce cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SpecializeReduceFuncCreator> &pair) { return pair.first; });

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Reduce does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_list[kernel_type_][index].second();
  func_obj_->InitFunc(base_operator, inputs, outputs);
  return true;
}

int ReduceCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }
  ret = func_obj_->Resize(base_operator, inputs, outputs, inputsOnHost);

  return ret;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceMean,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(kReduceMean); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceMax,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(kReduceMax); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceSum,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(kReduceSum); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceMin,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(kReduceMin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceProd,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(kReduceProd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceAll,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(kReduceAll); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceAny,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(kReduceAny); });
}  // namespace kernel
}  // namespace mindspore
