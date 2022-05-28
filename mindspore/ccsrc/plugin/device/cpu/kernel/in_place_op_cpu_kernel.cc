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

#include "plugin/device/cpu/kernel/in_place_op_cpu_kernel.h"
#include <memory>
#include <string>
#include <algorithm>
#include "mindspore/core/ops/inplace_add.h"
#include "mindspore/core/ops/inplace_sub.h"

namespace mindspore {
namespace kernel {
namespace {
struct Add {
  template <typename T>
  inline T operator()(const T &lhs, const T &rhs) const {
    return lhs + rhs;
  }
};

struct Sub {
  template <typename T>
  inline T operator()(const T &lhs, const T &rhs) const {
    return lhs - rhs;
  }
};
template <typename T>
class InplaceOpCpuTypeFunc : public DeprecatedCpuKernelFunc {
 public:
  ~InplaceOpCpuTypeFunc() override = default;
  explicit InplaceOpCpuTypeFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &) {
    MS_EXCEPTION_IF_NULL(base_operator);
    kernel_name_ = base_operator->GetPrim()->name();
    auto x_shape = inputs.at(0)->GetShapeVector();
    auto v_shape = inputs.at(1)->GetShapeVector();

    if (kernel_name_ == ops::kNameInplaceAdd) {
      auto kernel_ptr = std::make_shared<ops::InplaceAdd>(base_operator->GetPrim());
      indices_ = kernel_ptr->get_indices();
    } else if (kernel_name_ == ops::kNameInplaceSub) {
      auto kernel_ptr = std::make_shared<ops::InplaceSub>(base_operator->GetPrim());
      indices_ = kernel_ptr->get_indices();
    } else {
      MS_LOG(EXCEPTION) << "InplaceOp cpu does not support " << kernel_name_;
    }

    // x_shape_.size() == v_shape.size() is checked at front end
    // x_shape_[1:] == v_shape[1:] is checked at front end
    band_size_ = 1;
    for (size_t i = 1; i < x_shape.size(); ++i) {
      band_size_ *= x_shape[i];
    }

    // indices_.size() == v_shape[0] is checked at front end
    output_size_ = band_size_ * v_shape[0];

    InitComputeFunc();
  }

  template <typename Op>
  void InplaceOp(const T *input1, const T *input2, T *out) {
    const int64_t band_size = band_size_;
    const int64_t *indices = indices_.data();
    auto task = [band_size, indices, input1, input2, out](size_t start, size_t end) {
      while (start < end) {
        const int64_t v_row = SizeToLong(start) / band_size;
        const int64_t x_row = indices[v_row];

        size_t offset = SizeToLong(start) % band_size;
        size_t up_bound = (LongToSize((v_row + 1) * band_size) > end) ? end % band_size : band_size;

        size_t x_offset = x_row * band_size;
        size_t v_offset = v_row * band_size;
        for (size_t j = offset; j < up_bound; ++j) {
          out[x_offset + j] = Op()(input1[x_offset + j], input2[v_offset + j]);
        }
        start = v_row * band_size + up_bound;
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
               const std::vector<AddressPtr> &outputs) override {
    auto *input1 = reinterpret_cast<T *>(inputs[0]->addr);
    const auto *input2 = reinterpret_cast<T *>(inputs[1]->addr);
    auto *output = reinterpret_cast<T *>(outputs[0]->addr);
    if (memcpy_s(output, outputs[0]->size, input1, inputs[0]->size) != EOK) {
      MS_LOG(ERROR) << "Function memcpy_s failed in 'InplaceOp'.";
      return false;
    }
    compute_func_(this, input1, input2, output);
    return true;
  }

 private:
  void InitComputeFunc() {
    static std::unordered_map<std::string, TypeComputeFunc> inplaceOpFuncMap = {
      {prim::kPrimInplaceAdd->name(), &InplaceOpCpuTypeFunc<T>::InplaceOp<Add>},
      {prim::kPrimInplaceSub->name(), &InplaceOpCpuTypeFunc<T>::InplaceOp<Sub>},
    };
    if (inplaceOpFuncMap.find(kernel_name_) == inplaceOpFuncMap.end()) {
      MS_LOG(EXCEPTION) << "For 'InplaceOp', only supports operators in " << Unorderedmap2Str(inplaceOpFuncMap)
                        << ", but got " << kernel_name_ << ".";
    }
    compute_func_ = inplaceOpFuncMap.at(kernel_name_);
  }

  std::string kernel_name_;
  int64_t band_size_{1};
  int64_t output_size_{1};
  std::vector<int64_t> indices_;

  using TypeComputeFunc = std::function<void(InplaceOpCpuTypeFunc *, const T *in_x, const T *in_y, T *out)>;
  TypeComputeFunc compute_func_{nullptr};
};

template <typename T>
std::shared_ptr<DeprecatedCpuKernelFunc> InplaceOpCpuFunc(const BaseOperatorPtr &base_operator,
                                                          const std::vector<KernelTensorPtr> &inputs,
                                                          const std::vector<KernelTensorPtr> &outputs) {
  return std::make_shared<InplaceOpCpuTypeFunc<T>>(base_operator, inputs, outputs);
}
using InplaceOpCpuFuncCreator = std::function<std::shared_ptr<DeprecatedCpuKernelFunc>(
  const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
  const std::vector<KernelTensorPtr> &outputs)>;
using OpFuncList = std::vector<std::pair<KernelAttr, InplaceOpCpuFuncCreator>>;
static const mindspore::HashMap<std::string, OpFuncList> kernel_attr_list = {
  {ops::kNameInplaceAdd,
   {
     {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      InplaceOpCpuFunc<int32_t>},
     {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      InplaceOpCpuFunc<float>},
     {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
      InplaceOpCpuFunc<float16>},
   }},
  {ops::kNameInplaceSub,
   {
     {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      InplaceOpCpuFunc<int32_t>},
     {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      InplaceOpCpuFunc<float>},
     {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
      InplaceOpCpuFunc<float16>},
   }},
};
}  // namespace

bool InPlaceOpCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "InplaceOp does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_list.at(kernel_name_)[index].second(base_operator, inputs, outputs);
  return true;
}

std::vector<KernelAttr> InPlaceOpCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_list.find(kernel_type_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(EXCEPTION) << "InplaceOp cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, InplaceOpCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG_WITH_NAME_PARAM(NativeCpuKernelMod, InplaceAdd, InPlaceOpCpuKernelMod);
MS_KERNEL_FACTORY_REG_WITH_NAME_PARAM(NativeCpuKernelMod, InplaceSub, InPlaceOpCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
