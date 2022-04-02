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

#include "plugin/device/cpu/kernel/scatter_nd_arithmetic_cpu_kernel.h"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "include/common/thread_pool.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kScatterNdArithmeticInputsNum = 3;
constexpr size_t kScatterNdArithmeticOutputsNum = 1;
constexpr size_t kMinIndicesRank = 2;
constexpr size_t kMaxIndicesRank = 8;
constexpr size_t kInputIndex = 0;
constexpr size_t kIndicesIndex = 1;
constexpr size_t kUpdatesIndex = 2;
constexpr size_t kOutputIndex = 0;

template <typename T, typename S>
class ScatterNdArithmeticCpuKernelFunc : public CpuKernelFunc {
 public:
  ScatterNdArithmeticCpuKernelFunc() = default;
  ~ScatterNdArithmeticCpuKernelFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  void InitComputeFunc();
  void ScatterNdMul(const T *in, T *out) const;

  using TypeComputeFunc = std::function<void(ScatterNdArithmeticCpuKernelFunc *, const T *in, T *out)>;

  TypeComputeFunc compute_func_;
  bool use_locking_{true};
  size_t slice_size_;
  size_t batch_size_{1};
  size_t inner_size_{1};
  std::vector<size_t> batch_strides_;
  std::vector<size_t> input_shape_;
  std::string kernel_name_;
};

template <typename T, typename S>
void ScatterNdArithmeticCpuKernelFunc<T, S>::InitComputeFunc() {
  static const std::map<std::string, TypeComputeFunc> scatterNdArithmeticFuncMap{
    {prim::kPrimScatterNdMul->name(), &ScatterNdArithmeticCpuKernelFunc<T, S>::ScatterNdMul}};
  if (scatterNdArithmeticFuncMap.find(kernel_name_) == scatterNdArithmeticFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the current operator does not support this operation.";
  }
  compute_func_ = scatterNdArithmeticFuncMap.at(kernel_name_);
}

template <typename T, typename S>
void ScatterNdArithmeticCpuKernelFunc<T, S>::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto updates_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);

  auto P = input_shape_.size();
  auto Q = indices_shape.size();
  auto K = indices_shape.back();
  auto U = updates_shape.size();
  if (K > P) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the value of last dimension of 'indices' should be less than "
                         "or equal to the dimension of 'input_x', but got the value of last dimension of 'indices': "
                      << K << " and the dimension of 'input_x': " << P << ".";
  }
  if (Q < kMinIndicesRank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'indices' should be at least 2, "
                      << "but got " << Q << ".";
  }
  if (U != Q - 1 + P - K) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'updates' must satisfy the equivalence relationship: "
                      << "len(updates.shape) == len(indices.shape) - 1 + len(input_x.shape) - indices.shape[-1], "
                      << "but got " << updates_shape.size() << " vs " << (Q - 1 + P - K) << ".";
  }
  for (size_t i = 0; i < U; ++i) {
    bool is_valid = (i <= Q - kMinIndicesRank && updates_shape[i] == indices_shape[i]) |
                    (i > Q - kMinIndicesRank && updates_shape[i] == input_shape_[i - (Q - 1) + K]);
    if (!is_valid) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of 'updates' must satisfy the equivalence relationship: "
                        << "updates.shape == indices.shape[:-1] + input_x.shape[indices.shape[-1]:].";
    }
  }
  // Initialize use_locking_, slice_size_, batch_size_, inner_size_, batch_strides
  if (common::AnfAlgo::HasNodeAttr(USE_LOCKING, kernel_node)) {
    use_locking_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, USE_LOCKING);
  }
  slice_size_ = K;
  for (size_t i = 0; i < U; ++i) {
    if (i <= Q - kMinIndicesRank) {
      batch_size_ *= indices_shape[i];
    } else {
      inner_size_ *= updates_shape[i];
    }
  }
  batch_strides_.resize(K);
  // Since the quit condition(i >= 0) is about negative integer,
  // we convert iterated index from unsigned integer to signed integer.
  for (auto i = SizeToLong(K) - 1; i >= 0; i--) {
    auto idx = LongToSize(i);
    if (idx == K - 1) {
      batch_strides_[idx] = 1;
    } else {
      batch_strides_[idx] = batch_strides_[idx + 1] * input_shape_[idx + 1];
    }
  }
  InitComputeFunc();
}

template <typename T, typename S>
void ScatterNdArithmeticCpuKernelFunc<T, S>::ScatterNdMul(const T *in, T *out) const {
  for (size_t i = 0; i < inner_size_; i++) {
    out[i] *= in[i];
  }
}

template <typename T, typename S>
bool ScatterNdArithmeticCpuKernelFunc<T, S>::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScatterNdArithmeticInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScatterNdArithmeticOutputsNum, kernel_name_);
  auto *input = reinterpret_cast<T *>(inputs[kInputIndex]->addr);
  auto *indices = reinterpret_cast<S *>(inputs[kIndicesIndex]->addr);
  auto *updates = reinterpret_cast<T *>(inputs[kUpdatesIndex]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kOutputIndex]->addr);

  // ScatterNd* operations need to write input data and copy into output data,
  // while TensorScatter* operations need to copy input data and write into output data.
  auto *target = input;
  auto task = [this, &target, &indices, &updates](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      size_t upd_index = i * inner_size_;
      size_t out_index = 0;
      bool is_valid = true;  // Check if index is valid.
      for (size_t j = 0; j < slice_size_; j++) {
        auto idx_index = indices[i * slice_size_ + j];
        out_index += batch_strides_[j] * idx_index * inner_size_;
        is_valid &= (idx_index >= 0 && idx_index < static_cast<S>(input_shape_[j]));
      }
      if (!is_valid) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input 'indices' is out of bounds.";
      }
      compute_func_(this, updates + upd_index, target + out_index);
    }
    return common::SUCCESS;
  };

  // If 'use_locking' is false, we can use multi-process to parallelize the scatter operation.
  if (use_locking_) {
    task(0, batch_size_);
  } else {
    auto block_size = batch_size_ / GetActorMgrInnerThreadPool()->GetKernelThreadNum();
    ParallelLaunch(task, batch_size_, block_size, this);
  }

  auto ret = memcpy_s(output, outputs[kOutputIndex]->size, input, inputs[kInputIndex]->size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy_s error. Error no: " << ret;
  }
  return true;
}

template <typename T, typename S>
std::shared_ptr<CpuKernelFunc> SpecializeScatterNdArithFunc() {
  return std::make_shared<ScatterNdArithmeticCpuKernelFunc<T, S>>();
}
using ScatterNdArithFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::vector<std::pair<KernelAttr, ScatterNdArithFuncCreator>> func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   SpecializeScatterNdArithFunc<double, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   SpecializeScatterNdArithFunc<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   SpecializeScatterNdArithFunc<int64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   SpecializeScatterNdArithFunc<int32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   SpecializeScatterNdArithFunc<int16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   SpecializeScatterNdArithFunc<int8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   SpecializeScatterNdArithFunc<uint64_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   SpecializeScatterNdArithFunc<uint32_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   SpecializeScatterNdArithFunc<uint16_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   SpecializeScatterNdArithFunc<uint8_t, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   SpecializeScatterNdArithFunc<double, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   SpecializeScatterNdArithFunc<float, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   SpecializeScatterNdArithFunc<int64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   SpecializeScatterNdArithFunc<int32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt16)
     .AddOutputAttr(kNumberTypeInt16),
   SpecializeScatterNdArithFunc<int16_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt8)
     .AddOutputAttr(kNumberTypeInt8),
   SpecializeScatterNdArithFunc<int8_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   SpecializeScatterNdArithFunc<uint64_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddOutputAttr(kNumberTypeUInt32),
   SpecializeScatterNdArithFunc<uint32_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt16)
     .AddOutputAttr(kNumberTypeUInt16),
   SpecializeScatterNdArithFunc<uint16_t, int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeUInt8)
     .AddOutputAttr(kNumberTypeUInt8),
   SpecializeScatterNdArithFunc<uint8_t, int32_t>}};
}  // namespace

std::vector<KernelAttr> ScatterNdArithmeticCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ScatterNdArithFuncCreator> &pair) { return pair.first; });
  return support_list;
}

void ScatterNdArithmeticCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << ", but got kernel name as " << kernel_name_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "ScatterArithmetic does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = func_list_[index].second();
  func_obj_->InitFunc(kernel_node);
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterNdMul,
                                 []() { return std::make_shared<ScatterNdArithmeticCpuKernelMod>(kScatterNdMul); });
}  // namespace kernel
}  // namespace mindspore
