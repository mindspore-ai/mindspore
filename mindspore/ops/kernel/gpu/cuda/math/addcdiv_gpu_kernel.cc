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
#include "kernel/gpu/cuda/math/addcdiv_gpu_kernel.h"
#include <utility>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/cuda_fp16.h"

namespace mindspore {
namespace kernel {
namespace {
#define F32 kNumberTypeFloat32
#define F16 kNumberTypeFloat16
#define I32 kNumberTypeInt32
#define I64 kNumberTypeInt64
#define F64 kNumberTypeFloat64
template <typename T, typename VT>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateAddcdivKernelPtr(const std::string &kernel_name,
                                                                      const uint32_t &device_id) {
  return std::make_unique<cukernel::AddcdivHelperGpuKernel<T, VT>>(kernel_name, device_id);
}
using AddcdivPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, AddcdivPtrCreatorFunc>> kernel_attr = {
  {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F16).AddOutputAttr(F32),
   CreateAddcdivKernelPtr<float, half>},
  {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddOutputAttr(F32),
   CreateAddcdivKernelPtr<float, float>},
  {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I32).AddOutputAttr(F32),
   CreateAddcdivKernelPtr<float, int>},
  {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F64).AddOutputAttr(F32),
   CreateAddcdivKernelPtr<float, double>},
  {KernelAttr().AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(F32).AddInputAttr(I64).AddOutputAttr(F32),
   CreateAddcdivKernelPtr<float, int64_t>},
  {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F16).AddOutputAttr(F64),
   CreateAddcdivKernelPtr<double, half>},
  {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F32).AddOutputAttr(F64),
   CreateAddcdivKernelPtr<double, float>},
  {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I32).AddOutputAttr(F64),
   CreateAddcdivKernelPtr<double, int>},
  {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddOutputAttr(F64),
   CreateAddcdivKernelPtr<double, double>},
  {KernelAttr().AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(F64).AddInputAttr(I64).AddOutputAttr(F64),
   CreateAddcdivKernelPtr<double, int64_t>},
  {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddOutputAttr(F16),
   CreateAddcdivKernelPtr<half, half>},
  {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F32).AddOutputAttr(F16),
   CreateAddcdivKernelPtr<half, float>},
  {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I32).AddOutputAttr(F16),
   CreateAddcdivKernelPtr<half, int>},
  {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F64).AddOutputAttr(F16),
   CreateAddcdivKernelPtr<half, double>},
  {KernelAttr().AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(F16).AddInputAttr(I64).AddOutputAttr(F16),
   CreateAddcdivKernelPtr<half, int64_t>},
  {KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(F16).AddOutputAttr(I64),
   CreateAddcdivKernelPtr<int64_t, half>},
  {KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(F32).AddOutputAttr(I64),
   CreateAddcdivKernelPtr<int64_t, float>},
  {KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I32).AddOutputAttr(I64),
   CreateAddcdivKernelPtr<int64_t, int>},
  {KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(F64).AddOutputAttr(I64),
   CreateAddcdivKernelPtr<int64_t, double>},
  {KernelAttr().AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddInputAttr(I64).AddOutputAttr(I64),
   CreateAddcdivKernelPtr<int64_t, int64_t>}};
}  // namespace

bool AddcdivGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool AddcdivGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [ "
                  << " float16, float32, float64], but got: " << kernel_attr << ".";

    return false;
  }

  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  return true;
}

int AddcdivGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  constexpr size_t kIdx2 = 2;
  constexpr size_t kIdx3 = 3;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;

  std::vector<int64_t> inp_shape = inputs[0]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);

  inp_shape = inputs[1]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);

  inp_shape = inputs[kIdx2]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);

  inp_shape = inputs[kIdx3]->GetShapeVector();
  input_shapes.emplace_back(inp_shape);

  std::vector<int64_t> out_shape = outputs[0]->GetShapeVector();
  output_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> AddcdivGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AddcdivPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Addcdiv, AddcdivGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
