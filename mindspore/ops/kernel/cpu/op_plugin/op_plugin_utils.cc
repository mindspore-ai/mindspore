/**
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

#include "kernel/cpu/op_plugin/op_plugin_utils.h"
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <algorithm>
#include "ops_utils/dl_utils.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "ir/tensor_new.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "view/view_strides_calculator.h"

namespace mindspore::kernel {
namespace op_plugin {
namespace {
const char *GetCachedTypeName(TypeId type_id) {
  // The plugin ABI consumes dtype names only for the duration of the call. A
  // worker-local cache avoids rebuilding the same std::string for every input
  // and output on every invocation, while keeping returned c_str pointers
  // stable as the node-based unordered_map grows.
  thread_local std::unordered_map<TypeId, std::string> type_names;
  const auto iter = type_names.find(type_id);
  if (iter != type_names.end()) {
    return iter->second.c_str();
  }
  const auto result = type_names.emplace(type_id, TypeIdToString(type_id, true));
  return result.first->second.c_str();
}
}  // namespace

OpPluginKernelParam CreateOpPluginParam(std::vector<KernelTensor *> inputs, const std::vector<KernelTensor *> &outputs,
                                        const std::vector<KernelTensor *> &workspace, size_t layout_output_count) {
  OpPluginKernelParam param;

  const auto tensor_param_size = inputs.size() + outputs.size();
  param.params.reserve(tensor_param_size + workspace.size());
  param.shapes.reserve(tensor_param_size);
  param.ndims.reserve(tensor_param_size);
  param.dtypes.reserve(tensor_param_size);

  // Process inputs
  for (const auto &input : inputs) {
    param.params.push_back(input->device_ptr());
    const auto &in_shape = input->GetShapeVector();
    param.shapes.push_back(const_cast<int64_t *>(in_shape.data()));
    param.ndims.push_back(SizeToInt(in_shape.size()));
    param.dtypes.push_back(GetCachedTypeName(input->dtype_id()));
  }

  // Process outputs
  for (const auto &output : outputs) {
    param.params.push_back(output->device_ptr());
    const auto &out_shape = output->GetShapeVector();
    param.shapes.push_back(const_cast<int64_t *>(out_shape.data()));
    param.ndims.push_back(SizeToInt(out_shape.size()));
    param.dtypes.push_back(GetCachedTypeName(output->dtype_id()));
  }

  // Process workspace
  std::transform(workspace.begin(), workspace.end(), std::back_inserter(param.params),
                 [](const KernelTensor *ws) { return ws->device_ptr(); });

  param.kernel_info.SetKernelInput(std::move(inputs));
  if (layout_output_count != 0) {
    auto info = std::make_unique<OpPluginOutputShapeInfo>();
    info->output_shapes.resize(layout_output_count);
    info->shape_calculated.resize(layout_output_count, false);
    info->output_layouts.resize(layout_output_count);
    param.kernel_info.SetOutputShapeInfo(std::move(info));
  }
  return param;
}

int LaunchOpPluginKernel(const std::string &op_name, size_t nparam, void **params, int *ndims, int64_t **shapes,
                         const char **dtypes, void *kernel_info, void *stream) {
  using OpPluginFunc = int (*)(int, void **, int *, int64_t **, const char **, void *, void *);
  OpPluginFunc op_plugin_func = nullptr;

  const auto resolve_symbol = [&op_name, &op_plugin_func]() {
    // PyBoost CPU tasks normally execute on a stable worker thread. Cache one
    // resolved entry per op and worker to avoid repeated dlsym calls without a
    // global lock on the hot path.
    static thread_local std::unordered_map<std::string, OpPluginFunc> op_func_cache;
    const auto cached = op_func_cache.find(op_name);
    if (cached != op_func_cache.end()) {
      op_plugin_func = cached->second;
      return true;
    }
    void *handle = ops::GetOpPluginHandle();
    if (handle == nullptr) {
      MS_LOG(ERROR) << "Op plugin handle is not initialized. Please ensure MS_OP_PLUGIN_PATH is set correctly.";
      return false;
    }

    // Clear previous errors before dlsym
    (void)DL_ERROR();
#ifdef _WIN32
    SetLastError(0);
#endif
    op_plugin_func = reinterpret_cast<OpPluginFunc>(DL_SYM(handle, op_name.c_str()));
    if (auto error_info = DL_ERROR(); error_info != nullptr) {
      MS_LOG(ERROR) << "Failed to load op plugin kernel function for '" << op_name << "'. Error info: " << error_info;
      return false;
    }
    (void)op_func_cache.emplace(op_name, op_plugin_func);
    return true;
  };

  bool resolved = false;
  resolved = resolve_symbol();
  if (!resolved) {
    return -1;
  }

  auto *input_info = static_cast<KernelInputInfo *>(kernel_info);
  MS_EXCEPTION_IF_NULL(input_info);
  // Keep allocation/destruction on the framework side when an older op_plugin
  // fills only the original shape-info prefix.
  if (input_info->GetOutputShapeInfo() == nullptr && ops::IsOpPluginComputeDependOp(op_name)) {
    input_info->SetOutputShapeInfo(std::make_unique<OpPluginOutputShapeInfo>());
  }
  int ret = 0;
  ret = op_plugin_func(nparam, params, ndims, shapes, dtypes, stream, kernel_info);

  return ret;
}

int LaunchOpPluginKernel(const std::string &op_name, OpPluginKernelParam *param) {
  return LaunchOpPluginKernel(op_name, param->params.size(), param->params.data(), param->ndims.data(),
                              param->shapes.data(), param->dtypes.data(), reinterpret_cast<void *>(&param->kernel_info),
                              param->stream);
}

namespace {
using OutputLayoutCapabilityFn = int (*)(const char *);
OutputLayoutCapabilityFn GetOutputLayoutCapability() {
  const auto handle = ops::GetOpPluginHandle();
  if (handle == nullptr) return nullptr;
  static thread_local const auto entry =
    reinterpret_cast<OutputLayoutCapabilityFn>(DL_SYM(handle, "OpPluginSupportsOutputLayoutV1"));
  return entry;
}

TensorStorageInfoPtr CreateOutputStorageInfo(const OpPluginOutputLayout &layout, const ShapeVector &shape, TypeId dtype,
                                             size_t bytes) {
  const auto unit = abstract::TypeIdSize(dtype);
  if (unit == 0 || bytes % unit != 0 || bytes / unit > static_cast<size_t>(INT64_MAX) ||
      layout.strides.size() != shape.size()) {
    MS_LOG(EXCEPTION) << "Invalid op_plugin output dtype, capacity or strides";
  }
  bool empty = false;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0 || layout.strides[i] < 0) MS_LOG(EXCEPTION) << "Negative output shape or stride";
    empty = empty || shape[i] == 0;
  }
  size_t last = layout.storage_offset;
  if (!empty) {
    for (size_t i = 0; i < shape.size(); ++i) {
      const auto extent = static_cast<size_t>(shape[i] - 1);
      const auto stride = static_cast<size_t>(layout.strides[i]);
      if (stride != 0 && extent > (std::numeric_limits<size_t>::max() - last) / stride) {
        MS_LOG(EXCEPTION) << "Op_plugin output layout overflow";
      }
      last += extent * stride;
    }
  }
  // An empty view has no elements to access, even if its storage offset exceeds the buffer capacity.
  if (!empty && last >= bytes / unit) {
    MS_LOG(EXCEPTION) << "Op_plugin output exceeds MindSpore storage capacity";
  }
  return std::make_shared<TensorStorageInfo>(shape, layout.strides, layout.storage_offset,
                                             ShapeVector{static_cast<int64_t>(bytes / unit)}, ShapeVector{1},
                                             ops::IsContiguous(shape, layout.strides));
}
}  // namespace

bool GetPluginOutputLayoutKind(const std::string &op_name) {
  const auto capability = GetOutputLayoutCapability();
  if (capability == nullptr) return false;
  static thread_local std::unordered_map<std::string, bool> supported;
  const auto cached = supported.find(op_name);
  if (cached != supported.end()) return cached->second;
  return supported.emplace(op_name, capability(op_name.c_str()) == 1).first->second;
}

std::vector<tensor::TensorPtr> PreparePluginOutputBuffers(const std::string &op_name,
                                                          const std::vector<tensor::TensorPtr> &outputs) {
  if (op_name != "NonZeroExt" || outputs.empty()) return outputs;
  const auto shape = outputs.front()->shape();
  if (shape.size() != 1 || shape[0] < 0 ||
      static_cast<uint64_t>(shape[0]) > static_cast<uint64_t>(INT64_MAX) / outputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid NonZeroExt output upper bound";
  }
  // All tuple columns share one MindSpore allocation of [N, R].
  outputs.front()->set_shape({shape[0], SizeToLong(outputs.size())});
  return {outputs.front()};
}

void UpdatePluginOutputLayouts(OpPluginKernelParam &param, const std::vector<tensor::TensorPtr> &outputs,
                               const std::vector<tensor::TensorPtr> &buffers) {
  auto *info = param.kernel_info.GetOutputShapeInfo();
  MS_EXCEPTION_IF_NULL(info);
  if (info->output_layouts.size() != outputs.size() || info->output_shapes.size() != outputs.size() ||
      info->shape_calculated.size() != outputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid op_plugin output information count";
  }
  std::vector<device::DeviceAddressPtr> addresses;
  std::vector<TensorStorageInfoPtr> layouts;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &layout = info->output_layouts[i];
    if (!info->shape_calculated[i] || layout.buffer_index >= buffers.size()) {
      MS_LOG(EXCEPTION) << "Invalid op_plugin output shape or buffer index";
    }
    auto address = buffers[layout.buffer_index]->device_address();
    MS_EXCEPTION_IF_NULL(address);
    layouts.push_back(
      CreateOutputStorageInfo(layout, info->output_shapes[i], outputs[i]->data_type(), address->GetSize()));
    addresses.push_back(address);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i]->set_device_address(addresses[i]);
    outputs[i]->set_storage_info(layouts[i]);
    outputs[i]->set_need_pipeline_sync(true);
    outputs[i]->set_contiguous_callback(
      [](const tensor::TensorPtr &self) { return pyboost::PyBoostUtils::MakeContiguousDeviceAddress(self); });
  }
}
}  // namespace op_plugin
}  // namespace mindspore::kernel
