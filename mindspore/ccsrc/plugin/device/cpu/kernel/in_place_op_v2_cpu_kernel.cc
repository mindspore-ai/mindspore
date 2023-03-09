/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/in_place_op_v2_cpu_kernel.h"
#include <map>
#include <memory>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "mindspore/core/ops/inplace_update_v2.h"

namespace mindspore {
namespace kernel {
namespace {
struct Update {
  template <typename T>
  inline T operator()(const T &rhs) const {
    return rhs;
  }
};
template <typename Op>
struct NoCheckUpdate {
  template <typename T>
  static inline void compute(T *x, const int64_t x_idx, const T *v, const int64_t v_idx) {
    x[x_idx] = Op()(v[v_idx]);
  }
};
template <typename T, typename S>
class InplaceOpV2CpuTypeFunc : public CpuKernelFunc {
 public:
  InplaceOpV2CpuTypeFunc() = default;
  ~InplaceOpV2CpuTypeFunc() override = default;
  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &,
                const std::vector<KernelTensorPtr> &) override {
    MS_EXCEPTION_IF_NULL(base_operator);
    MS_EXCEPTION_IF_NULL(base_operator->GetPrim());
    kernel_name_ = base_operator->GetPrim()->name();

    static std::unordered_map<std::string, TypeComputeFuncV2> inplaceOpV2FuncMap = {
      {prim::kPrimInplaceUpdateV2->name(), &InplaceOpV2CpuTypeFunc<T, S>::InplaceOpV2<NoCheckUpdate<Update>>},
    };
    if (inplaceOpV2FuncMap.find(kernel_name_) == inplaceOpV2FuncMap.end()) {
      MS_LOG(EXCEPTION) << "For 'InplaceOpV2', only supports operators in "
                        << Map2Str<std::unordered_map, TypeComputeFuncV2>(inplaceOpV2FuncMap) << ", but got "
                        << kernel_name_ << ".";
    }
    compute_func_ = inplaceOpV2FuncMap.at(kernel_name_);
  }

  int Resize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &,
             const std::map<uint32_t, tensor::TensorPtr> &) override {
    if (inputs.size() != kInplaceOpV2InputNum) {
      MS_LOG(ERROR) << "For 'InplaceOpV2', the size of inputs must be 3, but got " << inputs.size() << ".";
      return KRET_RESIZE_FAILED;
    }
    MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
    MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
    auto indice_shape = inputs[kIndex1]->GetShapeVector();
    auto v_shape = inputs[kIndex2]->GetShapeVector();
    if (v_shape.empty()) {
      MS_LOG(ERROR) << "For 'InplaceOpV2', the shape size of value:" << v_shape.size() << " must not be 0.";
      return KRET_RESIZE_FAILED;
    }
    const auto &indices_num = (indice_shape.empty() ? 1 : indice_shape[0]);
    if (indices_num != v_shape[0]) {
      MS_LOG(ERROR) << "For 'InplaceOpV2', the size of indices must equal to input_v's shape[0].";
      return KRET_RESIZE_FAILED;
    }

    band_size_ = std::accumulate(v_shape.begin() + 1, v_shape.end(), int64_t(1), std::multiplies{});
    v_size_ = band_size_ * v_shape[0];

    return KRET_OK;
  }

  template <typename Op>
  void InplaceOpV2(T *x, const std::vector<int64_t> &indices, const T *v) {
    const int64_t band_size = band_size_;
    auto task = [band_size, indices, x, v](size_t start, size_t end) {
      int64_t start_long = SizeToLong(start);
      const int64_t end_long = SizeToLong(end);
      while (start_long < end_long) {
        const int64_t v_row = start_long / band_size;
        const int64_t x_row = (indices.data())[v_row];

        int64_t offset = start_long % band_size;
        int64_t up_bound = (((v_row + 1) * band_size) > end_long) ? end_long % band_size : band_size;

        int64_t x_offset = x_row * band_size;
        int64_t v_offset = v_row * band_size;
        for (int64_t j = offset; j < up_bound; ++j) {
          Op::compute(x, x_offset + j, v, v_offset + j);
        }
        start_long = v_row * band_size + up_bound;
      }
    };
    ParallelLaunchAutoSearch(task, LongToSize(v_size_), this, &parallel_search_info_);
  }

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
               const std::vector<AddressPtr> &outputs) override {
    auto *x = reinterpret_cast<T *>(inputs[0]->addr);
    const auto *v = reinterpret_cast<T *>(inputs[kIndex2]->addr);
    auto *output = reinterpret_cast<T *>(outputs[0]->addr);
    if (memcpy_s(output, outputs[0]->size, x, inputs[0]->size) != EOK) {
      MS_LOG(ERROR) << "Function memcpy_s failed in 'InplaceOpV2'.";
      return false;
    }

    std::vector<int64_t> indices;
    const auto *indice_ptr = reinterpret_cast<S *>(inputs[kIndex1]->addr);
    MS_EXCEPTION_IF_NULL(indice_ptr);
    for (size_t i = 0; i < inputs[kIndex1]->size / sizeof(S); ++i) {
      indices.emplace_back(IntToLong(indice_ptr[i]));
    }

    compute_func_(this, output, indices, v);
    return true;
  }

 private:
  std::string kernel_name_;
  int64_t band_size_{1};
  int64_t v_size_{1};

  using TypeComputeFuncV2 =
    std::function<void(InplaceOpV2CpuTypeFunc *, T *x, const std::vector<int64_t> &indices, const T *v)>;
  TypeComputeFuncV2 compute_func_{nullptr};
};

template <typename T, typename S>
std::shared_ptr<CpuKernelFunc> InplaceOpV2CpuFunc() {
  return std::make_shared<InplaceOpV2CpuTypeFunc<T, S>>();
}
using InplaceOpCpuFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
using OpFuncList = std::vector<std::pair<KernelAttr, InplaceOpCpuFuncCreator>>;

#define DTYPE_REGISTER(INPUT_X, INPUT_INDICES, INPUT_V, OUTPUT, T, S)                                           \
  {                                                                                                             \
    KernelAttr().AddInputAttr(INPUT_X).AddInputAttr(INPUT_INDICES).AddInputAttr(INPUT_V).AddOutputAttr(OUTPUT), \
      InplaceOpV2CpuFunc<T, S>                                                                                  \
  }

static const mindspore::HashMap<std::string, OpFuncList> kernel_attr_list = {
  {ops::kNameInplaceUpdateV2,
   {
     DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t),
     DTYPE_REGISTER(kNumberTypeFloat32, kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32, float, int32_t),
     DTYPE_REGISTER(kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat16, float16, int32_t),
     DTYPE_REGISTER(kNumberTypeInt32, kNumberTypeInt64, kNumberTypeInt32, kNumberTypeInt32, int32_t, int64_t),
     DTYPE_REGISTER(kNumberTypeFloat32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat32, float, int64_t),
     DTYPE_REGISTER(kNumberTypeFloat16, kNumberTypeInt64, kNumberTypeFloat16, kNumberTypeFloat16, float16, int64_t),
   }},
};
}  // namespace

bool InPlaceOpV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "InplaceOpV2 does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_list.at(kernel_name_)[index].second();

  func_obj_->InitFunc(base_operator, inputs, outputs);

  return true;
}

int InPlaceOpV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return func_obj_->Resize(base_operator, inputs, outputs);
}

std::vector<KernelAttr> InPlaceOpV2CpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_list.find(kernel_type_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(EXCEPTION) << "InplaceOpV2 cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, InplaceOpCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG_WITH_NAME_PARAM(NativeCpuKernelMod, InplaceUpdateV2, InPlaceOpV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
