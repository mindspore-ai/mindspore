/**
 *
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tools/silent_detect/checksum/checksum_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ir/dtype/tensor_type.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace checksum {
constexpr auto kCheckSum = "CheckSum";
constexpr auto kOutputAddress = "OutputAddress";
constexpr auto kWorkspaceAddress = "WorkspaceAddress";
constexpr float kRatioThreshold = 5.0f;
constexpr float kFactor = 1.0f / 256;  // 2 ** (-8)

KernelTensorPtr CreateOutPutKernelTensor(const DeviceContext *device_context, const TypeId &dtype_id,
                                         const ShapeVector &shape) {
  MS_EXCEPTION_IF_NULL(device_context);

  auto shape_ptr = std::make_shared<abstract::Shape>(shape);
  auto type = std::make_shared<TensorType>(TypeIdToType(dtype_id));
  size_t byte_size = std::accumulate(shape.begin(), shape.end(), UnitSizeInBytes(dtype_id), std::multiplies<size_t>());
  auto tensor = AnfAlgo::CreateKernelTensor(
    shape_ptr, type, nullptr, nullptr, byte_size, kernel::GetFormatFromEnumToStr(Format::DEFAULT_FORMAT), dtype_id,
    shape, device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  tensor->set_stream_id(kDefaultStreamIndex);
  auto device_addr = tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, kCheckSum, kOutputAddress, "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, kCheckSum, device::tracker::MemType::kOther,
                                                 device_addr->GetSize(), device_addr.get());
  if (!device_context->device_res_manager_->AllocateMemory(device_addr.get(), kDefaultStreamIndex)) {
    MS_LOG(EXCEPTION) << "CheckSum allocate outputs memory failed";
  } else {
    static std::string name = "Alloc memory";
    tensor->IncreaseNewRefCount(name);
  }
  return tensor;
}

KernelTensorPtr CreateWorkspaceKernelTensor(const DeviceContext *device_context, const size_t &workspace_size) {
  MS_EXCEPTION_IF_NULL(device_context);

  auto kernel_tensor =
    AnfAlgo::CreateKernelTensor(nullptr, workspace_size, Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
                                device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                device_context->device_context_key().device_id_);
  kernel_tensor->set_stream_id(kDefaultStreamIndex);

  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, kCheckSum, kWorkspaceAddress, "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, kCheckSum, device::tracker::MemType::kWorkSpace,
                                                 device_address->GetSize(), device_address.get());
  if (!device_context->device_res_manager_->AllocateMemory(device_address.get(), kDefaultStreamIndex)) {
    MS_LOG(EXCEPTION) << "CheckSum allocate dynamic workspace memory failed";
  } else {
    static std::string name = "Alloc memory";
    kernel_tensor->IncreaseNewRefCount(name);
  }
  MS_LOG(DEBUG) << "Create workspace device address:" << device_address;
  return kernel_tensor;
}

class BaseKernel {
 public:
  BaseKernel(const DeviceContext *device_context, const string &kernel_name)
      : device_context_(device_context), kernel_name_(kernel_name) {
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context_->device_res_manager_);
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Construct kernel, kernel name: " << kernel_name;
    kernel_mod_ = device_context_->GetKernelExecutor()->CreateKernelMod(kernel_name);
    MS_EXCEPTION_IF_NULL(kernel_mod_);
  }

  void LaunchKernelAsync(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                         const uint32_t stream_id) {
    auto workspace = GetWorkSpaceDeviceAddress(inputs, outputs);
    std::vector<KernelTensor *> workspaces;
    if (workspace != nullptr) {
      workspaces.emplace_back(workspace.get());
    }

    stream_id_ = stream_id;
    void *stream_ptr = device_context_->device_res_manager_->GetStream(stream_id_);
    MS_EXCEPTION_IF_NULL(stream_ptr);

    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch kernel, kernel name: " << kernel_name_ << ", stream: " << stream_id_;
    bool ret = kernel_mod_->Launch(inputs, workspaces, outputs, stream_ptr);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch Kernel failed, kernel name: " << kernel_name_ << ", stream: " << stream_id_;
    }
  }

 protected:
  std::vector<KernelTensorPtr> GetExtraInputsDeviceAddress(int64_t dim_value, bool keepdim_value) {
    std::vector<int64_t> dim_vec = {dim_value};
    auto dim = std::make_shared<KernelTensor>(nullptr, kInt64, MakeValue(dim_vec));
    MS_EXCEPTION_IF_NULL(dim);
    auto keepdim = std::make_shared<KernelTensor>(nullptr, kBool, MakeValue(keepdim_value));
    MS_EXCEPTION_IF_NULL(keepdim);
    return {dim, keepdim};
  }

  KernelTensorPtr GetOutputDeviceAddress(const KernelTensor *input, int64_t dim, bool keepdim,
                                         TypeId dtype = kTypeUnknown) {
    ShapeVector shape = input->GetShapeVector();
    if (dim >= static_cast<int64_t>(shape.size())) {
      MS_LOG(EXCEPTION) << "dim " << dim << " should be less than input dimension " << shape.size();
    }
    if (keepdim) {
      shape[dim] = 1;
    } else {
      shape.erase(shape.begin() + dim);
    }
    if (dtype == kTypeUnknown) {
      dtype = input->dtype_id();
    }
    return CreateOutPutKernelTensor(device_context_, dtype, shape);
  }

  virtual KernelTensorPtr GetWorkSpaceDeviceAddress(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
    auto ret = kernel_mod_->Resize(inputs, outputs);
    if (ret) {
      MS_LOG(EXCEPTION) << "Kernel resize failed, kernel name: " << kernel_name_ << ", errno: " << ret;
    }
    auto workspace = kernel_mod_->GetWorkspaceSizeList();
    if (!workspace.empty() && workspace[0] != 0) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Kernel name: " << kernel_name_
                                      << ", input shape: " << inputs[0]->GetShapeVector()
                                      << ", input dtype: " << TypeIdToString(inputs[0]->dtype_id())
                                      << ", output shape: " << outputs[0]->GetShapeVector()
                                      << ", output dtype: " << TypeIdToString(outputs[0]->dtype_id())
                                      << ", workspace size: " << workspace[0];
      return CreateWorkspaceKernelTensor(device_context_, workspace[0]);
    }
    return nullptr;
  }

  const DeviceContext *device_context_{nullptr};
  string kernel_name_;
  kernel::KernelModPtr kernel_mod_;
  uint32_t stream_id_ = kDefaultStreamIndex;
};

class SumKernel : public BaseKernel {
 public:
  explicit SumKernel(const DeviceContext *device_context, const string &kernel_name = ops::kNameSumExt)
      : BaseKernel(device_context, kernel_name) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *input, int64_t dim, bool keepdim, TypeId dtype,
                                    const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    std::vector<KernelTensor *> inputs{input};
    auto extra_inputs = GetExtraInputsDeviceAddress(dim, keepdim);
    std::transform(extra_inputs.begin(), extra_inputs.end(), std::back_inserter(inputs),
                   [](const auto &extra_input) { return extra_input.get(); });

    auto output = GetOutputDeviceAddress(input, dim, keepdim, dtype);
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }
};

class MeanKernel : public SumKernel {
 public:
  explicit MeanKernel(const DeviceContext *device_context) : SumKernel(device_context, ops::kNameMeanExt) {}
};

class AbsKernel : public BaseKernel {
 public:
  explicit AbsKernel(const DeviceContext *device_context) : BaseKernel(device_context, ops::kNameAbs) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *input, const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    std::vector<KernelTensor *> inputs{input};

    auto output = CreateOutPutKernelTensor(device_context_, input->dtype_id(), input->GetShapeVector());
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }
};

class MatMulKernel : public BaseKernel {
 public:
  explicit MatMulKernel(const DeviceContext *device_context) : BaseKernel(device_context, ops::kNameMatMul) {}

  // inputs: {input_a, input_b, transpose_a, transpose_b}, dtype_a == dtype_b, dim=2
  KernelTensorPtr LaunchKernelAsync(const std::vector<KernelTensor *> &inputs, const uint32_t stream_id) {
    auto output = GetOutputDeviceAddress(inputs);
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }

 protected:
  KernelTensorPtr GetOutputDeviceAddress(const std::vector<KernelTensor *> &inputs) {
    KernelTensor *input_a = inputs[0];
    MS_EXCEPTION_IF_NULL(input_a);
    ShapeVector shape_a = input_a->GetShapeVector();
    KernelTensor *input_b = inputs[1];
    MS_EXCEPTION_IF_NULL(input_b);
    ShapeVector shape_b = input_b->GetShapeVector();
    bool trans_a = inputs[2]->GetValueWithCheck<bool>();
    bool trans_b = inputs[3]->GetValueWithCheck<bool>();
    ShapeVector shape(kDim2);
    shape[0] = trans_a ? shape_a[1] : shape_a[0];
    shape[1] = trans_b ? shape_b[0] : shape_b[1];
    return CreateOutPutKernelTensor(device_context_, input_a->dtype_id(), shape);
  }

  KernelTensorPtr GetWorkSpaceDeviceAddress(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) override {
    auto ret = kernel_mod_->Resize(inputs, outputs);
    if (ret) {
      MS_LOG(EXCEPTION) << "Kernel resize failed, kernel name: " << kernel_name_ << ", errno: " << ret;
    }
    auto workspace = kernel_mod_->GetWorkspaceSizeList();
    if (!workspace.empty() && workspace[0] != 0) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Kernel name: " << kernel_name_
                                      << ", input_a shape: " << inputs[0]->GetShapeVector()
                                      << ", input_b shape: " << inputs[1]->GetShapeVector()
                                      << ", output shape: " << outputs[0]->GetShapeVector()
                                      << ", dtype: " << TypeIdToString(inputs[0]->dtype_id())
                                      << ", workspace size: " << workspace[0];
      return CreateWorkspaceKernelTensor(device_context_, workspace[0]);
    }
    return nullptr;
  }
};

// tensor * scalar
class MulsKernel : public BaseKernel {
 public:
  explicit MulsKernel(const DeviceContext *device_context) : BaseKernel(device_context, ops::kNameMuls) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *input, float scalar, const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    // MulsAscend not support bf16 scalar
    auto scalar_tensor = std::make_shared<KernelTensor>(nullptr, kFloat32, MakeValue(scalar));
    std::vector<KernelTensor *> inputs{input, scalar_tensor.get()};

    auto output = CreateOutPutKernelTensor(device_context_, input->dtype_id(), input->GetShapeVector());
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }
};

class MaxDimKernel : public BaseKernel {
 public:
  explicit MaxDimKernel(const DeviceContext *device_context) : BaseKernel(device_context, ops::kNameMaxDim) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *input, int64_t dim, bool keepdim, const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    std::vector<KernelTensor *> inputs{input};
    auto extra_inputs = GetExtraInputsDeviceAddress(dim, keepdim);
    std::transform(extra_inputs.begin(), extra_inputs.end(), std::back_inserter(inputs),
                   [](const auto &extra_input) { return extra_input.get(); });

    std::vector<KernelTensorPtr> output = GetOutputDeviceAddress(input, dim, keepdim);
    MS_EXCEPTION_IF_CHECK_FAIL(output.size() == kDim2, "output size is not equal 2");
    std::vector<KernelTensor *> outputs{output[0].get(), output[1].get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output[0];
  }

 protected:
  std::vector<KernelTensorPtr> GetOutputDeviceAddress(const KernelTensor *input, int64_t dim, bool keepdim) {
    auto indices = BaseKernel::GetOutputDeviceAddress(input, dim, keepdim, kNumberTypeInt64);
    auto output = BaseKernel::GetOutputDeviceAddress(input, dim, keepdim);
    std::vector<KernelTensorPtr> outputs{output, indices};
    return outputs;
  }
};

class CastKernel : public BaseKernel {
 public:
  explicit CastKernel(const DeviceContext *device_context) : BaseKernel(device_context, ops::kNameCast) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *input, TypeId dtype, const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    std::vector<KernelTensor *> inputs{input};

    auto output = CreateOutPutKernelTensor(device_context_, dtype, input->GetShapeVector());
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }
};

class SubKernel : public BaseKernel {
 public:
  explicit SubKernel(const DeviceContext *device_context, const string &kernel_name = ops::kNameSub)
      : BaseKernel(device_context, kernel_name) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *input, KernelTensor *other, const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    std::vector<KernelTensor *> inputs{input, other};

    auto output = CreateOutPutKernelTensor(device_context_, input->dtype_id(), input->GetShapeVector());
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }
};

class DivKernel : public SubKernel {
 public:
  explicit DivKernel(const DeviceContext *device_context) : SubKernel(device_context, ops::kNameDiv) {}
};

class MaxKernel : public BaseKernel {
 public:
  explicit MaxKernel(const DeviceContext *device_context, const string &kernel_name = ops::kNameMax)
      : BaseKernel(device_context, kernel_name) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *input, const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    std::vector<KernelTensor *> inputs{input};

    auto output = CreateOutPutKernelTensor(device_context_, input->dtype_id(), ShapeVector());
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }
};

class MinKernel : public MaxKernel {
 public:
  explicit MinKernel(const DeviceContext *device_context) : MaxKernel(device_context, ops::kNameMin) {}
};

class GeKernel : public BaseKernel {
 public:
  explicit GeKernel(const DeviceContext *device_context) : BaseKernel(device_context, ops::kNameGreaterEqualScalar) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *input, float scalar, const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    auto scalar_tensor = std::make_shared<KernelTensor>(nullptr, kBFloat16, MakeValue(scalar));
    std::vector<KernelTensor *> inputs{input, scalar_tensor.get()};

    auto output = CreateOutPutKernelTensor(device_context_, kNumberTypeBool, input->GetShapeVector());
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }
};

// condition ? input : other
class SelectKernel : public BaseKernel {
 public:
  explicit SelectKernel(const DeviceContext *device_context) : BaseKernel(device_context, ops::kNameSelect) {}

  KernelTensorPtr LaunchKernelAsync(KernelTensor *condition, KernelTensor *input, KernelTensor *other,
                                    const uint32_t stream_id) {
    MS_EXCEPTION_IF_NULL(input);
    std::vector<KernelTensor *> inputs{condition, input, other};

    auto output = CreateOutPutKernelTensor(device_context_, input->dtype_id(), input->GetShapeVector());
    MS_EXCEPTION_IF_NULL(output);
    std::vector<KernelTensor *> outputs{output.get()};

    BaseKernel::LaunchKernelAsync(inputs, outputs, stream_id);
    return output;
  }
};

const std::set<TypeId> CheckSumKernel::supported_dtype_ = {kNumberTypeBFloat16};

bool CheckSumKernel::IsCheckSumSupported(const std::vector<KernelTensor *> &matmul_inputs,
                                         const std::vector<KernelTensor *> &matmul_outputs) {
  // inputs: a, b, transpose_a, transpose_b
  if (matmul_inputs.size() < kInputNum4 || matmul_outputs.size() < 1) {
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "CheckSum is not supported for MatMul with input size " << matmul_inputs.size()
                                    << "and output size " << matmul_outputs.size();
    return false;
  }
  const std::vector<KernelTensor *> matrices = {matmul_inputs[0], matmul_inputs[1], matmul_outputs[0]};
  for (const auto &matrix : matrices) {
    MS_EXCEPTION_IF_NULL(matrix);
    size_t dim = matrix->GetShapeVector().size();
    if (dim != kDim2) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "CheckSum is not supported for " << dim << "D input tensors";
      return false;
    }
    TypeId type_id = matrix->dtype_id();
    if (supported_dtype_.find(type_id) == supported_dtype_.end()) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "CheckSum is not supported for dtype " << TypeIdToType(type_id);
      return false;
    }
  }
  return true;
}

KernelTensorPtr CheckSumKernel::LaunchKernelAsync(std::vector<KernelTensor *> matmul_inputs,
                                                  std::vector<KernelTensor *> matmul_outputs,
                                                  const std::uint32_t stream_id) {
  tensor_a_ = matmul_inputs[kIndex0];
  MS_EXCEPTION_IF_NULL(tensor_a_);
  tensor_b_ = matmul_inputs[kIndex1];
  MS_EXCEPTION_IF_NULL(tensor_b_);
  tensor_trans_a_ = matmul_inputs[kIndex2];
  MS_EXCEPTION_IF_NULL(tensor_trans_a_);
  tensor_trans_b_ = matmul_inputs[kIndex3];
  MS_EXCEPTION_IF_NULL(tensor_trans_b_);
  tensor_c_ = matmul_outputs[kIndex0];
  MS_EXCEPTION_IF_NULL(tensor_c_);
  shape_a_ = tensor_a_->GetShapeVector();  // (M, N)
  shape_b_ = tensor_b_->GetShapeVector();  // (N, K)
  shape_c_ = tensor_c_->GetShapeVector();  // (M, K)
  stream_id_ = stream_id;

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch CheckSum kernels, stream id: " << stream_id_;
  KernelTensorPtr error = CalculateError();             // (M, )
  KernelTensorPtr error_total = CalculateErrorTotal();  // (M, )
  // diff_max = max(error - error_total)
  KernelTensorPtr diff = SubKernel(device_context_).LaunchKernelAsync(error.get(), error_total.get(), stream_id_);
  return MaxKernel(device_context_).LaunchKernelAsync(diff.get(), stream_id_);  // (1, )
}

/*
c_sum = sum(c, dim=-1, dtype=float32)
b1 = sum(b, dim=-1, keepdim=True, dtype=float32)
c1 = matmul(a.to(float32), b1)
c1_trans = c1.squeeze(-1)
error = abs(c_sum - c1_trans)
*/
KernelTensorPtr CheckSumKernel::CalculateError() {
  KernelTensorPtr c1_trans = CalculateC1Trans();
  // c_sum = sum(c, dim=-1, dtype=float32)
  int64_t dim = static_cast<int64_t>(shape_c_.size()) - 1;
  KernelTensorPtr c_sum =
    SumKernel(device_context_).LaunchKernelAsync(tensor_c_, dim, false, kNumberTypeFloat32, stream_id_);
  KernelTensorPtr error = SubKernel(device_context_).LaunchKernelAsync(c_sum.get(), c1_trans.get(), stream_id_);
  return AbsKernel(device_context_).LaunchKernelAsync(error.get(), stream_id_);
}

KernelTensorPtr CheckSumKernel::CalculateC1Trans() {
  // b1 = sum(b, dim=-1, keepdim=True, dtype=float32)
  bool trans_b = tensor_trans_b_->GetValueWithCheck<bool>();
  int64_t dim = trans_b ? 0 : static_cast<int64_t>(shape_b_.size()) - 1;
  KernelTensorPtr b1 =
    SumKernel(device_context_).LaunchKernelAsync(tensor_b_, dim, true, kNumberTypeFloat32, stream_id_);
  // c1 = matmul(a.to(float32), b1)
  KernelTensorPtr a_cast = CastKernel(device_context_).LaunchKernelAsync(tensor_a_, kNumberTypeFloat32, stream_id_);
  std::vector<KernelTensor *> inputs{a_cast.get(), b1.get(), tensor_trans_a_, tensor_trans_b_};
  KernelTensorPtr c1 = MatMulKernel(device_context_).LaunchKernelAsync(inputs, stream_id_);  // (M, 1)
  // c1_trans = c1.squeeze(-1)
  ShapeVector shape_c1 = c1->GetShapeVector();
  if (shape_c1.size() < kDim2 || shape_c1[shape_c1.size() - 1] != 1) {
    MS_LOG(EXCEPTION) << "c1 shape " << shape_c1 << " cannot be squeezed.";
  }
  shape_c1.erase(shape_c1.end() - 1);
  c1->SetShapeVector(shape_c1);
  return c1;  // (M, )
}

/*
n_b = b.shape[-1]
c_max, _ = max(abs(c), dim=-1)
c_mean = mean(abs(c), dim=-1)
if min(c_max / c_mean) > 5:
    c_ele_round_error_accum = c_max * 2 ** (-8) * sqrt(n_b)
else:
    c_ele_round_error_accum = c_mean * 2 ** (-8) * n_b
error_total = (c_ele_round_error_accum).to(float)
*/
KernelTensorPtr CheckSumKernel::CalculateErrorTotal() {
  bool trans_b = tensor_trans_b_->GetValueWithCheck<bool>();
  float n_b = static_cast<float>(trans_b ? shape_b_[0] : shape_b_[shape_b_.size() - 1]);
  KernelTensorPtr c_abs = AbsKernel(device_context_).LaunchKernelAsync(tensor_c_, stream_id_);
  int64_t dim = static_cast<int64_t>(shape_c_.size()) - 1;
  KernelTensorPtr c_max = MaxDimKernel(device_context_).LaunchKernelAsync(c_abs.get(), dim, false, stream_id_);
  KernelTensorPtr c_mean =
    MeanKernel(device_context_).LaunchKernelAsync(c_abs.get(), dim, false, c_abs->dtype_id(), stream_id_);
  KernelTensorPtr c_ratio = DivKernel(device_context_).LaunchKernelAsync(c_max.get(), c_mean.get(), stream_id_);
  KernelTensorPtr c_ratio_min = MinKernel(device_context_).LaunchKernelAsync(c_ratio.get(), stream_id_);  // (1, )
  // calculate both branches and select based on the condition, to avoid sync
  KernelTensorPtr condition =
    GeKernel(device_context_).LaunchKernelAsync(c_ratio_min.get(), kRatioThreshold, stream_id_);
  float scalar_1 = std::sqrt(n_b) * kFactor;
  KernelTensorPtr error_1 = MulsKernel(device_context_).LaunchKernelAsync(c_max.get(), scalar_1, stream_id_);
  float scalar_2 = n_b * kFactor;
  KernelTensorPtr error_2 = MulsKernel(device_context_).LaunchKernelAsync(c_mean.get(), scalar_2, stream_id_);
  KernelTensorPtr c_ele_round_error_accum =
    SelectKernel(device_context_).LaunchKernelAsync(condition.get(), error_1.get(), error_2.get(), stream_id_);
  return CastKernel(device_context_).LaunchKernelAsync(c_ele_round_error_accum.get(), kNumberTypeFloat32, stream_id_);
}
}  // namespace checksum
}  // namespace mindspore
