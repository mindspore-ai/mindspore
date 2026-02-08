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
#include "kernel/cpu/native/rolling_cpu_kernel.h"
#include <cmath>
#include <map>
#include <limits>
#include <algorithm>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include "include/utils/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace rolling_cpu {
namespace {
using rolling::Method;

template <typename T, typename S>
class RollingCpuKernelFunc : public CpuKernelFunc {
 public:
  RollingCpuKernelFunc() = default;
  explicit RollingCpuKernelFunc(const PrimitivePtr prim, std::vector<size_t> *size_list)
      : primitive(prim), workspace_size_list(size_list) {}
  ~RollingCpuKernelFunc() override = default;

  void InitFunc(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool RunFunc(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
               const std::vector<KernelTensor *> &outputs) override;

 protected:
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    size_t element_size = axisIterator_.OuterSize() * axisIterator_.InnerSize() * axisIterator_.AxisSize();
    // input values
    workspace_size_list->clear();
    workspace_size_list->emplace_back((sizeof(size_t) * element_size));
    return KRET_OK;
  }

 private:
  void RollingBoundsCalculate();
  void MethodSwitch();
  S Var(const T *input_addr, const size_t *ids, size_t start, size_t end) const {
    // float for division
    float n = SizeToFloat(end - start);
    MS_EXCEPTION_IF_CHECK_FAIL(n > 1, "Divisor n must be lager than 1.");
    T sum1 = 0;
    for (size_t x = start; x < end; ++x) {
      sum1 += input_addr[ids[x]];
    }
    double mean = sum1 / n;
    double sum2 = 0;
    for (size_t x = start; x < end; ++x) {
      double diff = input_addr[ids[x]] - mean;
      sum2 += diff * diff;
    }
    // ddof = 1
    return sum2 / (n - 1);
  }

  int32_t window_{0};
  int64_t min_periods_{0};
  bool center_{false};
  std::string closed_{};
  rolling::Method method_{};
  std::function<S(const T *values, const size_t *ids, size_t start, size_t end)> reduceMethod_{};
  // shape info
  AxisIterator axisIterator_{};
  // rolling info
  std::vector<size_t> starts_{};
  std::vector<size_t> ends_{};
  std::string kernel_name_;
  PrimitivePtr primitive;
  std::vector<size_t> *workspace_size_list;
};

template <typename T, typename S>
void RollingCpuKernelFunc<T, S>::InitFunc(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  kernel_name_ = primitive->name();
  const auto &input_shape = inputs[kIndex0]->GetShapeVector();

  static const std::map<std::string, Method> kValidMethods = {
    {"max", Method::Max}, {"min", Method::Min}, {"mean", Method::Mean},
    {"sum", Method::Sum}, {"std", Method::Std}, {"var", Method::Var},
  };
  auto method = GetValue<std::string>(primitive->GetAttr(METHOD));
  if (kValidMethods.find(method) == kValidMethods.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the 'method' must be in (max, min, sum, mean, std, var), but got " << method;
  }
  method_ = kValidMethods.at(method);
  auto window = GetValue<int64_t>(primitive->GetAttr(WINDOW));
  if (window <= 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'window' must be greater than 0, but got " << window;
  }
  window_ = LongToInt(window);
  min_periods_ = GetValue<int64_t>(primitive->GetAttr(MIN_PERIODS));
  if (min_periods_ <= 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'min_periods' must be greater than 0, but got "
                      << min_periods_;
  }
  center_ = GetValue<bool>(primitive->GetAttr(CENTER));
  auto axis = GetValue<int64_t>(primitive->GetAttr(AXIS));
  size_t axis_t = axis < 0 ? LongToSize(axis + SizeToLong(input_shape.size())) : LongToSize(axis);
  closed_ = GetValue<std::string>(primitive->GetAttr(CLOSED));
  if (axis_t >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be less than the dimension of input tensor "
                      << input_shape.size() << "D, but got " << axis_t;
  }
  axisIterator_.Init(input_shape, axis_t);
  RollingBoundsCalculate();
  MethodSwitch();
}

template <typename T, typename S>
void RollingCpuKernelFunc<T, S>::RollingBoundsCalculate() {
  int offset = 0;
  if (center_) {
    offset = (window_ - 1) / 2;
  }

  starts_.resize(axisIterator_.AxisSize());
  ends_.resize(axisIterator_.AxisSize());
  int start_offset = 1;
  int end_offset = 1;
  if (closed_ == "left") {
    start_offset -= 1;
    end_offset -= 1;
  } else if (closed_ == "both") {
    start_offset -= 1;
  } else if (closed_ == "neither") {
    end_offset -= 1;
  }
  int axis_size = SizeToInt(axisIterator_.AxisSize());
  for (int i = 0; i < axis_size; ++i) {
    int end = offset + i + end_offset;
    int start = offset + i - window_ + start_offset;
    ends_[i] = IntToSize(std::max(0, std::min(end, axis_size)));
    starts_[i] = IntToSize(std::max(0, std::min(start, axis_size)));
  }
}

template <typename T, typename S>
void RollingCpuKernelFunc<T, S>::MethodSwitch() {
  switch (method_) {
    case Method::Max:
      reduceMethod_ = [](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        T max_value = std::numeric_limits<T>::min();
        for (size_t x = start; x < end; ++x) {
          if (max_value < input_addr[ids[x]]) {
            max_value = input_addr[ids[x]];
          }
        }
        return max_value;
      };
      break;
    case Method::Min:
      reduceMethod_ = [](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        T min_value = std::numeric_limits<T>::max();
        for (size_t x = start; x < end; ++x) {
          if (min_value > input_addr[ids[x]]) {
            min_value = input_addr[ids[x]];
          }
        }
        return min_value;
      };
      break;
    case Method::Sum:
      reduceMethod_ = [](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        T sum = 0;
        for (size_t x = start; x < end; ++x) {
          sum += input_addr[ids[x]];
        }
        return sum;
      };
      break;
    case Method::Mean:
      reduceMethod_ = [](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        T sum = 0;
        for (size_t x = start; x < end; ++x) {
          sum += input_addr[ids[x]];
        }
        return sum / SizeToFloat(end - start);
      };
      break;
    case Method::Var:
      reduceMethod_ = [this](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        return Var(input_addr, ids, start, end);
      };
      break;
    case Method::Std:
      reduceMethod_ = [this](const T *input_addr, const size_t *ids, size_t start, size_t end) {
        return std::sqrt(Var(input_addr, ids, start, end));
      };
      break;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the 'method' must be in (max, min, sum, mean, std, var), but got " << method_;
  }
}

template <typename T, typename S>
bool RollingCpuKernelFunc<T, S>::RunFunc(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  auto input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  auto workspace_addr = GetDeviceAddress<size_t>(workspace, kIndex0);
  auto output_addr = GetDeviceAddress<S>(outputs, kIndex0);

  T nan_value;
  if constexpr (std::is_same_v<T, float>) {
    nan_value = std::nanf("");
  } else if constexpr (std::is_same_v<T, double>) {
    nan_value = std::nan("");
  } else {
    // integer values not support nan
    nan_value = 0;
  }

  auto task = [this, nan_value, input_addr, workspace_addr, output_addr](size_t start, size_t end) {
    size_t axis_size = axisIterator_.AxisSize();
    AxisIterator iter(axisIterator_);
    for (size_t index = start; index < end; index++) {
      iter.SetOffset(index);

      size_t offset = index * axisIterator_.AxisSize();
      size_t *ids = workspace_addr + offset;
      // set indexes to avoid duplicate calculation
      for (size_t k = 0; k < axis_size; ++k) {
        ids[k] = iter.GetPos(k);
      }

      for (size_t w = 0; w < axis_size; ++w) {
        if (ends_[w] - starts_[w] < static_cast<size_t>(min_periods_)) {
          output_addr[iter.GetPos(w)] = nan_value;
        } else {
          output_addr[iter.GetPos(w)] = reduceMethod_(input_addr, ids, starts_[w], ends_[w]);
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, axisIterator_.OuterSize() * axisIterator_.InnerSize(), this, &parallel_search_info_);
  return true;
}

template <typename T, typename S>
std::shared_ptr<CpuKernelFunc> SpecializeRollingFunc(const PrimitivePtr prim, std::vector<size_t> *size_list) {
  return std::make_shared<RollingCpuKernelFunc<T, S>>(prim, size_list);
}
using SpecializeRollingFuncCreator =
  std::function<std::shared_ptr<CpuKernelFunc>(const PrimitivePtr prim, std::vector<size_t> *size_list)>;
static std::vector<std::pair<KernelAttr, SpecializeRollingFuncCreator>> kernel_attr_list = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   SpecializeRollingFunc<float, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   SpecializeRollingFunc<double, double>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   SpecializeRollingFunc<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   SpecializeRollingFunc<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   SpecializeRollingFunc<int32_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   SpecializeRollingFunc<int64_t, double>}};
}  // namespace

bool RollingCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Arithmetic does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_list[index].second(primitive_, &workspace_size_list_);
  func_obj_->InitFunc(inputs, outputs);
  return true;
}

int RollingCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  return func_obj_->Resize(inputs, outputs);
}

std::vector<KernelAttr> RollingCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr_list.begin(), kernel_attr_list.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SpecializeRollingFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Rolling, RollingCpuKernelMod);
}  // namespace rolling_cpu
}  // namespace kernel
}  // namespace mindspore
