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

#include "plugin/device/cpu/kernel/mkldnn/eltwise_cpu_kernel.h"
#include <functional>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kElu = "Elu";
constexpr auto kReLU = "ReLU";
constexpr auto kReLU6 = "ReLU6";
constexpr auto kExp = "Exp";
constexpr auto kSigmoid = "Sigmoid";
constexpr auto kSoftplus = "Softplus";
constexpr auto kLog = "Log";
constexpr auto kTanh = "Tanh";
constexpr auto kMish = "Mish";
constexpr auto kSqrt = "Sqrt";
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T>
class EltwiseCpuKernelFunc : public CpuKernelFunc {
 public:
  EltwiseCpuKernelFunc() = default;
  ~EltwiseCpuKernelFunc() override = default;
  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    dtype_ = inputs.at(kIndex0)->GetDtype();
    static std::map<string, std::vector<std::pair<KernelAttr, TypeComputeFunc>>> eltwise_func_map = {
      {kSigmoid,
       {{KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
         &EltwiseCpuKernelFunc<T>::SigmoidComplex},
        {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
         &EltwiseCpuKernelFunc<T>::SigmoidComplex},
        {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
         &EltwiseCpuKernelFunc<T>::SigmoidComplex}}}};

    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto iter = eltwise_func_map.find(kernel_name_);
    if (iter == eltwise_func_map.end()) {
      MS_LOG(EXCEPTION) << "For 'EltWise Op', the kernel name must be in "
                        << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, TypeComputeFunc>>>(
                             eltwise_func_map)
                        << ", but got " << kernel_name_;
    }
    std::vector<KernelAttr> support_list;
    (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                         [](const std::pair<KernelAttr, TypeComputeFunc> &pair) { return pair.first; });
    auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
    if (!is_match) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    }
    compute_func_ = eltwise_func_map[kernel_name_][index].second;

    // calculate input element num
    auto input_shape = inputs.at(kIndex0)->GetShapeVector();
    std::vector<size_t> src_shape;
    (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(src_shape), LongToSize);
    input_element_num_ = std::accumulate(src_shape.begin(), src_shape.end(), size_t(1), std::multiplies<size_t>());
  }

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override {
    auto *input = reinterpret_cast<T *>(inputs[0]->addr);
    auto *output = reinterpret_cast<T *>(outputs[0]->addr);
    compute_func_(this, input, output);
    return true;
  }

 private:
  std::string kernel_name_{kUnKnown};
  TypeId dtype_{kTypeUnknown};
  using TypeComputeFunc = std::function<void(EltwiseCpuKernelFunc *, const T *input, T *output)>;
  TypeComputeFunc compute_func_{nullptr};
  size_t input_element_num_{0};
  void SigmoidComplex(const T *input, T *output);
};  // namespace

template <typename T>
void EltwiseCpuKernelFunc<T>::SigmoidComplex(const T *input, T *output) {
  T one_complex{1, 0};
  auto task = [&input, &output, &one_complex](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] = one_complex / (one_complex + exp(-input[i]));
    }
  };
  ParallelLaunchAutoSearch(task, input_element_num_, this, &parallel_search_info_);
}

template <>
void EltwiseCpuKernelFunc<double>::SigmoidComplex(const double *input, double *output) {
  auto task = [&input, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
  };
  ParallelLaunchAutoSearch(task, input_element_num_, this, &parallel_search_info_);
}

struct DescParam {
  dnnl::algorithm algorithm{dnnl::algorithm::undef};
  float alpha{0.0f};
  float beta{0.0f};
};
}  // namespace

dnnl::eltwise_forward::desc EltWiseCpuKernelMod::GetForwardEltwiseDesc(const dnnl::memory::desc src_desc) {
  static const std::unordered_map<std::string, DescParam> eltwise_op_desc_map{
    {prim::kPrimReLU6->name(), DescParam{dnnl::algorithm::eltwise_clip, 0.0f, 6.0f}},
    {prim::kPrimExp->name(), DescParam{dnnl::algorithm::eltwise_exp}},
    {prim::kPrimLog->name(), DescParam{dnnl::algorithm::eltwise_log}},
    {prim::kPrimSigmoid->name(), DescParam{dnnl::algorithm::eltwise_logistic}},
    {prim::kPrimSqrt->name(), DescParam{dnnl::algorithm::eltwise_sqrt}},
    {prim::kPrimTanh->name(), DescParam{dnnl::algorithm::eltwise_tanh}},
    {prim::kPrimElu->name(), DescParam{dnnl::algorithm::eltwise_elu, 1.0f, 0.0f}},
    {prim::kPrimSoftplus->name(), DescParam{dnnl::algorithm::eltwise_soft_relu}},
    {prim::kPrimMish->name(), DescParam{dnnl::algorithm::eltwise_mish}},
  };
  const auto desc_pair = eltwise_op_desc_map.find(kernel_name_);
  if (desc_pair == eltwise_op_desc_map.end()) {
    MS_LOG(EXCEPTION) << "For 'EltWise Op', it does not support " << kernel_name_;
  }
  auto desc = CreateDesc<dnnl::eltwise_forward::desc>(dnnl_forward_, desc_pair->second.algorithm, src_desc,
                                                      desc_pair->second.alpha, desc_pair->second.beta);
  return desc;
}

bool EltWiseCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }

  dtype_ = inputs[0]->GetDtype();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto dnnl_type_id = GetDnnlDataType(dtype_);
  if (dnnl_type_id == dnnl::memory::data_type::undef) {
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetAdditionalDtypeOpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
      return false;
    }
    auto iter = additional_kernel_attr_map_.find(kernel_name_);
    if (iter == additional_kernel_attr_map_.end()) {
      MS_LOG(ERROR)
        << "For 'EltWise Op', the kernel name must be in "
        << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, EltWiseCpuKernelMod::EltwiseCpuFuncCreator>>>(
             additional_kernel_attr_map_)
        << ", but got " << kernel_name_;
      return false;
    }
    additional_func_ = iter->second[index].second();
    use_mkl_ = false;
  } else {
    auto is_match_pair = MatchKernelAttr(kernel_attr, GetMklOpSupport());
    if (!is_match_pair.first) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
      return false;
    }
    auto iter = mkl_kernel_attr_map_.find(kernel_name_);
    if (iter == mkl_kernel_attr_map_.end()) {
      MS_LOG(ERROR) << "For 'EltWise Op', the kernel name must be in "
                    << kernel::Map2Str<std::map, std::vector<KernelAttr>>(mkl_kernel_attr_map_) << ", but got "
                    << kernel_name_;
      return false;
    }
  }
  return true;
}

int EltWiseCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  src_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(src_shape_), LongToSize);
  input_element_num_ = std::accumulate(src_shape_.begin(), src_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num_ == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  dst_shape_.clear();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(dst_shape_), LongToSize);

  TypeId input_type_id = inputs.at(kIndex0)->GetDtype();
  auto dnnl_type_id = GetDnnlDataType(input_type_id);
  if (dnnl_type_id != dnnl::memory::data_type::undef) {
    if (src_shape_.empty()) {
      (void)src_shape_.insert(src_shape_.begin(), 1);
    }
    dnnl::memory::desc src_desc = GetExactMemDesc(src_shape_, dnnl_type_id);

    auto desc = GetForwardEltwiseDesc(src_desc);
    auto prim_desc = CreateDesc<dnnl::eltwise_forward::primitive_desc>(desc, engine_);
    primitive_ = CreatePrimitive<dnnl::eltwise_forward>(prim_desc);
    AddArgument(DNNL_ARG_SRC, src_desc);
    AddArgument(DNNL_ARG_DST, src_desc);
  }
  if (additional_func_) {
    additional_func_->InitFunc(base_operator, inputs, outputs);
  }
  return KRET_OK;
}

std::vector<KernelAttr> EltWiseCpuKernelMod::GetMklOpSupport() {
  auto iter = mkl_kernel_attr_map_.find(kernel_name_);
  if (iter == mkl_kernel_attr_map_.end()) {
    return std::vector<KernelAttr>{};
  }
  return mkl_kernel_attr_map_.at(kernel_name_);
}

std::vector<KernelAttr> EltWiseCpuKernelMod::GetAdditionalDtypeOpSupport() {
  auto iter = additional_kernel_attr_map_.find(kernel_name_);
  if (iter == additional_kernel_attr_map_.end()) {
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, EltWiseCpuKernelMod::EltwiseCpuFuncCreator> &pair) { return pair.first; });
  return support_list;
}

std::vector<KernelAttr> EltWiseCpuKernelMod::GetOpSupport() {
  // only mkl_kernel_attr_map_ need to be checked since it contains all kind of ops
  auto iter = mkl_kernel_attr_map_.find(kernel_name_);
  if (iter == mkl_kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'EltWise Op', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<KernelAttr>>(mkl_kernel_attr_map_) << ", but got "
                  << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  auto mkl_ops = GetMklOpSupport();
  if (!mkl_ops.empty()) {
    support_list.insert(support_list.end(), mkl_ops.begin(), mkl_ops.end());
  }
  auto additional_ops = GetAdditionalDtypeOpSupport();
  if (!additional_ops.empty()) {
    support_list.insert(support_list.end(), additional_ops.begin(), additional_ops.end());
  }
  return support_list;
}

bool EltWiseCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  if (use_mkl_) {
    CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
    CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
    SetArgumentHandle(DNNL_ARG_SRC, inputs.at(kIndex0)->addr);
    SetArgumentHandle(DNNL_ARG_DST, outputs.at(kIndex0)->addr);
    ExecutePrimitive();
  } else {
    std::vector<kernel::AddressPtr> workspace;
    additional_func_->RunFunc(inputs, workspace, outputs);
  }

  return true;
}

std::map<std::string, std::vector<KernelAttr>> EltWiseCpuKernelMod::mkl_kernel_attr_map_ = {
  {kElu, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
  {kReLU6, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
  {kExp, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
  {kLog, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
  {kSigmoid, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
  {kTanh, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
  {kSoftplus, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
  {kMish, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
  {kSqrt, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}}};

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeEltwiseFunc() {
  return std::make_shared<EltwiseCpuKernelFunc<T>>();
}

std::map<std::string, std::vector<std::pair<KernelAttr, EltWiseCpuKernelMod::EltwiseCpuFuncCreator>>>
  EltWiseCpuKernelMod::additional_kernel_attr_map_ = {
    {kSigmoid,
     {{KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       SpecializeEltwiseFunc<complex64>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       SpecializeEltwiseFunc<complex128>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       SpecializeEltwiseFunc<double>}}},
};

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Elu, []() { return std::make_shared<EltWiseCpuKernelMod>(kElu); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReLU6,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kReLU6); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sigmoid,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kSigmoid); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Softplus,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kSoftplus); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Mish,
                                 []() { return std::make_shared<EltWiseCpuKernelMod>(kMish); });
}  // namespace kernel
}  // namespace mindspore
