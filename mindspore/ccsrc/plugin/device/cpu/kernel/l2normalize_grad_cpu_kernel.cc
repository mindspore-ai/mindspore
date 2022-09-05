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

#include "plugin/device/cpu/kernel/l2normalize_grad_cpu_kernel.h"
#include <algorithm>
#include <string>
#include <memory>
#include <utility>
#include "mindspore/core/ops/grad/l2_normalize_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kL2NormalizeGradInputsNum = 3;
constexpr size_t kL2NormalizeGradOutputsNum = 1;

template <typename T>
class L2NormalizeGradCpuFunc : public CpuKernelFunc {
 public:
  L2NormalizeGradCpuFunc() = default;
  ~L2NormalizeGradCpuFunc() override = default;
  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  int CheckInputShape(const ShapeVector &output_shape);
  std::vector<size_t> OneDimIndexToHighDimIndex(size_t one_dim_index);
  void HighDimIndexToOneDimIndex(size_t *one_dim_index, const std::vector<size_t> &high_dim_index);
  std::vector<T> GetVector(const std::vector<size_t> &high_dim_index, const T *x);
  void GetSumOfProduct(const std::vector<T> &x_vector, const std::vector<T> &y_vector, T *ss) const;
  void GetOutput(const std::vector<T> &input_x_vector, const std::vector<T> &y_vector,
                 const std::vector<T> &dout_vector, const std::vector<size_t> &high_dim_index, T *output);
  std::vector<ShapeVector> input_shape_list_;
  std::vector<size_t> dim_elem_num_list_;
  int axis_{0};
  T epsilon_{0};
  std::string kernel_name_;
};

template <typename T>
void L2NormalizeGradCpuFunc<T>::InitFunc(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto l2_normalize_grad_ptr = std::dynamic_pointer_cast<ops::L2NormalizeGrad>(base_operator);
  if (l2_normalize_grad_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', cast 'L2NormalizeGrad' ops failed!";
  }
  epsilon_ = static_cast<T>(l2_normalize_grad_ptr->get_epsilon());
  axis_ = LongToInt(l2_normalize_grad_ptr->get_axis());
}

template <typename T>
int L2NormalizeGradCpuFunc<T>::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  input_shape_list_.clear();
  for (size_t i = 0; i < kL2NormalizeGradInputsNum; i++) {
    (void)input_shape_list_.emplace_back(inputs[i]->GetShapeVector());
  }
  auto output_shape = outputs[0]->GetShapeVector();
  auto ret = CheckInputShape(output_shape);
  if (ret != KRET_OK) {
    return ret;
  }

  int input_dim_length = SizeToInt(input_shape_list_[0].size());
  axis_ = axis_ < 0 ? (axis_ + input_dim_length) : axis_;

  int output_dim_length = output_shape.size();
  dim_elem_num_list_.resize(output_dim_length, 1);
  for (int i = output_dim_length - 2; i >= 0; i--) {  // from -2 to 0 dim
    auto idx = IntToSize(i);
    dim_elem_num_list_[idx] = LongToSize(output_shape[idx + 1]) * dim_elem_num_list_[idx + 1];
  }
  return KRET_OK;
}

template <typename T>
bool L2NormalizeGradCpuFunc<T>::RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kL2NormalizeGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kL2NormalizeGradOutputsNum, kernel_name_);
  auto input_x = reinterpret_cast<T *>(inputs[0]->addr);
  auto y = reinterpret_cast<T *>(inputs[1]->addr);
  auto dout = reinterpret_cast<T *>(inputs[2]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto output_size = outputs[0]->size / sizeof(T);
  auto task = [this, input_x, y, dout, output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> high_dim_index = OneDimIndexToHighDimIndex(i);
      std::vector<T> input_x_vector = GetVector(high_dim_index, input_x);
      std::vector<T> dout_vector = GetVector(high_dim_index, dout);
      std::vector<T> y_vector = GetVector(high_dim_index, y);
      GetOutput(input_x_vector, y_vector, dout_vector, high_dim_index, &output[i]);
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

template <typename T>
int L2NormalizeGradCpuFunc<T>::CheckInputShape(const ShapeVector &output_shape) {
  if (std::any_of(input_shape_list_.begin(), input_shape_list_.end(),
                  [output_shape](ShapeVector item) { return output_shape != item; })) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input must be the same as output's";
    return KRET_RESIZE_FAILED;
  }

  auto input_x_shape = input_shape_list_[0];
  if (input_x_shape.size() != 0) {
    if (std::any_of(input_x_shape.begin(), input_x_shape.end(), [](int64_t i) -> bool { return i == 0; })) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input 'x' can not be null.";
      return KRET_RESIZE_FAILED;
    }
  }
  return KRET_OK;
}

template <typename T>
std::vector<size_t> L2NormalizeGradCpuFunc<T>::OneDimIndexToHighDimIndex(size_t one_dim_index) {
  std::vector<size_t> high_dim_index;
  high_dim_index.reserve(dim_elem_num_list_.size());
  for (const auto &item : dim_elem_num_list_) {
    high_dim_index.push_back(one_dim_index / item);
    one_dim_index %= item;
  }
  // referred to Copy elision https://en.cppreference.com/w/cpp/language/copy_elision
  // returning a vector won't cause extra vector constructed or moved
  return high_dim_index;
}

template <typename T>
void L2NormalizeGradCpuFunc<T>::HighDimIndexToOneDimIndex(size_t *one_dim_index,
                                                          const std::vector<size_t> &high_dim_index) {
  *one_dim_index = 0;
  int len = high_dim_index.size();
  for (int i = 0; i < len; i++) {
    *one_dim_index += high_dim_index[i] * dim_elem_num_list_[i];
  }
}

template <typename T>
std::vector<T> L2NormalizeGradCpuFunc<T>::GetVector(const std::vector<size_t> &high_dim_index, const T *x) {
  auto x_shape = input_shape_list_[0];
  std::vector<T> x_vector;
  auto idx = IntToSize(axis_);
  x_vector.reserve(LongToSize(x_shape[idx]));
  for (size_t i = 0; i < LongToSize(x_shape[idx]); i++) {
    size_t oneDimIndex = 0;
    std::vector<size_t> tmp_high_dim_index = high_dim_index;
    tmp_high_dim_index[idx] = i;
    HighDimIndexToOneDimIndex(&oneDimIndex, tmp_high_dim_index);
    (void)x_vector.emplace_back(x[oneDimIndex]);
  }
  // referred to Copy elision https://en.cppreference.com/w/cpp/language/copy_elision
  // returning a vector won't cause extra vector constructed or moved
  return x_vector;
}

template <typename T>
void L2NormalizeGradCpuFunc<T>::GetSumOfProduct(const std::vector<T> &x_vector, const std::vector<T> &y_vector,
                                                T *ss) const {
  size_t len = x_vector.size();
  std::vector<T> tmp_vector(len);
  for (size_t i = 0; i < len; i++) {
    tmp_vector[i] = x_vector[i] * y_vector[i];
  }
  const size_t half = 2;
  if (len % half == 1) {
    tmp_vector[0] += tmp_vector[len - 1];
  }
  for (size_t stride = len / half; stride > 0; stride >>= 1) {
    for (size_t i = 0; i < stride; i++) {
      tmp_vector[i] += tmp_vector[i + stride];
    }
    if (stride > half && stride % half == 1) {
      tmp_vector[0] += tmp_vector[stride - 1];
    }
  }
  *ss = tmp_vector[0];
}

template <typename T>
void L2NormalizeGradCpuFunc<T>::GetOutput(const std::vector<T> &input_x_vector, const std::vector<T> &y_vector,
                                          const std::vector<T> &dout_vector, const std::vector<size_t> &high_dim_index,
                                          T *output) {
  size_t axis_index = high_dim_index[axis_];
  T dout = dout_vector[axis_index];
  T y = y_vector[axis_index];
  T tmp_sum1;
  GetSumOfProduct(y_vector, dout_vector, &tmp_sum1);
  T tmp_sum2;
  GetSumOfProduct(input_x_vector, input_x_vector, &tmp_sum2);
  tmp_sum2 = sqrt(tmp_sum2);
  if (tmp_sum2 >= epsilon_) {
    *output = (dout - y * tmp_sum1) / tmp_sum2;
  } else {
    *output = (dout - y * tmp_sum1) / epsilon_;
  }
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeL2NormGradFunc() {
  return std::make_shared<L2NormalizeGradCpuFunc<T>>();
}
using SpecializeL2NormGradFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
std::vector<std::pair<KernelAttr, SpecializeL2NormGradFuncCreator>> func_class_list = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   SpecializeL2NormGradFunc<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   SpecializeL2NormGradFunc<float16>}};
}  // namespace

bool L2NormalizeGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Arithmetic', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  func_obj_ = func_class_list[index].second();
  func_obj_->InitFunc(base_operator, inputs, outputs);
  return true;
}

int L2NormalizeGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return func_obj_->Resize(base_operator, inputs, outputs);
}

std::vector<KernelAttr> L2NormalizeGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_class_list.begin(), func_class_list.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SpecializeL2NormGradFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, L2NormalizeGrad, L2NormalizeGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
