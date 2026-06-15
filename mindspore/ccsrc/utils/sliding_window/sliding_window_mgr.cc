/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/utils/sliding_window/sliding_window_mgr.h"

#include <atomic>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <typeinfo>
#include <unordered_set>

#include "abstract/utils.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindspore/ccsrc/utils/sliding_window/sliding_window_internal.h"
#include "ops/infer_info/infer_info_utils.h"
#include "ops/op_def.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace utils {
namespace sliding_window {

namespace {

constexpr char kSlidingWindowBufferSizeEnv[] = "MS_DEV_SLIDING_WINDOW_BUFFER_SIZE_MB";
constexpr size_t kBytesPerMB = static_cast<size_t>(1024) * 1024;
constexpr size_t kDefaultSlidingWindowBufferSize = static_cast<size_t>(1024) * 1024 * 1024;

// Recursively flatten a ValuePtr (single tensor, ValueTuple, ValueList) into TensorPtrs.
void CollectTensors(const ValuePtr &value, std::vector<tensor::TensorPtr> *tensors) {
  if (value == nullptr) {
    return;
  }
  auto tensor = value->cast<tensor::TensorPtr>();
  if (tensor != nullptr) {
    tensors->push_back(tensor);
    return;
  }
  auto value_seq = value->cast<ValueSequencePtr>();
  if (value_seq != nullptr) {
    for (const auto &elem : value_seq->value()) {
      CollectTensors(elem, tensors);
    }
  }
}

std::vector<tensor::TensorPtr> ExtractTensors(const ValuePtr &value) {
  std::vector<tensor::TensorPtr> tensors;
  CollectTensors(value, &tensors);
  return tensors;
}

std::vector<tensor::TensorPtr> CollectInputTensors(const std::vector<ValuePtr> &inputs) {
  std::vector<tensor::TensorPtr> input_tensors;
  for (const auto &val : inputs) {
    CollectTensors(val, &input_tensors);
  }
  return input_tensors;
}

// Compute total output byte size from AbstractBasePtr.
size_t ComputeSizeFromAbstract(const abstract::AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return 0;
  }

  if (abs->isa<abstract::AbstractTensor>()) {
    auto tensor_abs = abs->cast<abstract::AbstractTensorPtr>();
    auto shape_ptr = tensor_abs->GetShape();
    auto type_ptr = tensor_abs->GetType();
    if (shape_ptr == nullptr || type_ptr == nullptr) {
      return 0;
    }
    const auto &shape_vec = shape_ptr->GetShapeVector();
    size_t elem_count = SizeOf(shape_vec);
    if (elem_count == 0) {
      return 0;
    }
    auto tensor_type = type_ptr->cast<TensorTypePtr>();
    if (tensor_type == nullptr || tensor_type->element() == nullptr) {
      return 0;
    }
    return AlignToPoolSize(elem_count * GetTypeByte(tensor_type->element()));
  }

  if (abs->isa<abstract::AbstractSequence>()) {
    auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    const auto &elements = seq_abs->elements();
    return std::accumulate(elements.begin(), elements.end(), size_t{0},
                           [](size_t total, const auto &elem) { return total + ComputeSizeFromAbstract(elem); });
  }

  return 0;
}

// Compute total output byte size from ValueSimpleInfo (SimpleInfer result).
size_t ComputeSizeFromValueSimpleInfo(const ValueSimpleInfoPtr &info) {
  if (info == nullptr) {
    return 0;
  }
  size_t total = 0;
  for (size_t i = 0; i < info->dtype_vector_.size(); ++i) {
    size_t elem_count = SizeOf(info->shape_vector_[i]);
    if (elem_count == 0) {
      continue;
    }
    auto dtype = info->dtype_vector_[i];
    if (dtype == nullptr) {
      continue;
    }
    total += AlignToPoolSize(elem_count * GetTypeByte(dtype));
  }
  return total;
}

bool AnyTensorOnAscend(const std::vector<tensor::TensorPtr> &tensors) {
  for (const auto &t : tensors) {
    if (t == nullptr) continue;
    auto addr = t->device_address();
    if (addr == nullptr) continue;
    if (addr->GetDeviceType() == device::DeviceType::kAscend) return true;
  }
  return false;
}

size_t GetSlidingWindowBufferSize() {
  static const size_t buffer_size = []() -> size_t {
    auto env_value = common::GetEnv(kSlidingWindowBufferSizeEnv);
    if (env_value.empty()) {
      return kDefaultSlidingWindowBufferSize;
    }
    if (!common::IsStrNumeric(env_value)) {
      MS_LOG(WARNING) << "Invalid " << kSlidingWindowBufferSizeEnv << "=" << env_value << ", use default "
                      << kDefaultSlidingWindowBufferSize;
      return kDefaultSlidingWindowBufferSize;
    }
    try {
      const auto parsed = std::stoull(env_value, nullptr, 10);
      constexpr auto kMaxBufferSizeMB = std::numeric_limits<size_t>::max() / kBytesPerMB;
      if (parsed > kMaxBufferSizeMB) {
        MS_LOG(WARNING) << "Out-of-range " << kSlidingWindowBufferSizeEnv << "=" << env_value << ", use default "
                        << kDefaultSlidingWindowBufferSize;
        return kDefaultSlidingWindowBufferSize;
      }
      return static_cast<size_t>(parsed) * kBytesPerMB;
    } catch (const std::exception &) {
      MS_LOG(WARNING) << "Failed to parse " << kSlidingWindowBufferSizeEnv << "=" << env_value << ", use default "
                      << kDefaultSlidingWindowBufferSize;
      return kDefaultSlidingWindowBufferSize;
    }
  }();
  return buffer_size;
}

// Infer output byte size, mirroring the paths PyBoost's OpRunner::InferOutput (op_runner.h:117)
// actually exercises: value-based DoGeneralInfer, value-based SimpleInfer, then abstract-based
// PyBoostUtils::InferByOpDef. Returns 0 for view/inplace ops (offloading them is unsafe anyway)
// and for ops without any of these paths — the caller's 1 GB headroom buffer absorbs misses.
size_t InferOutputSize(const std::string &class_name, const std::vector<ValuePtr> &inputs) {
  auto *op_def = ops::GetOpDef(class_name);
  if (op_def != nullptr) {
    if (op_def->is_view_) {
      return 0;
    }
    for (const auto &ret : op_def->returns_) {
      if (ret.inplace_input_index_ >= 0) {
        return 0;
      }
    }
    if (typeid(op_def->func_impl_) == typeid(ops::OpFuncImpl)) {
      return 0;
    }
  }

  auto prim = std::make_shared<Primitive>(class_name);

  // Value-based: DoGeneralInfer
  if (auto result = ops::DoGeneralInfer(prim, inputs); result != nullptr) {
    return ComputeSizeFromValueSimpleInfo(result);
  }

  // Value-based: SimpleInfer
  if (auto simple_func = ops::SimpleInfer::Instance().GetFunc(prim->name()); simple_func != nullptr) {
    auto result = std::make_shared<ValueSimpleInfo>();
    ops::SimpleInfer::Instance().DoSimpleInfer(prim, result, simple_func, inputs);
    return ComputeSizeFromValueSimpleInfo(result);
  }

  // Abstract-based: PyBoostUtils::InferByOpDef (same call pyboost OpRunner uses).
  if (op_def != nullptr) {
    std::vector<abstract::AbstractBasePtr> input_abs;
    input_abs.reserve(inputs.size());
    for (const auto &v : inputs) {
      input_abs.push_back(v != nullptr ? v->ToAbstract() : kNone->ToAbstract());
    }
    return ComputeSizeFromAbstract(kernel::pyboost::PyBoostUtils::InferByOpDef(prim, input_abs));
  }

  return 0;
}

}  // namespace

SlidingWindowMgr::SlidingWindowMgr() {
  source_group_mgr_ = std::make_unique<SourceGroupManager>();
  space_manager_ = std::make_unique<DeviceSpaceManager>(source_group_mgr_.get());
}

uint64_t SlidingWindowMgr::GetCurrentTimestamp() {
  static std::atomic<uint64_t> global_timestamp{0};
  return ++global_timestamp;
}

void SlidingWindowMgr::RegisterOutputs(const std::vector<tensor::TensorPtr> &outputs, uint64_t timestamp,
                                       const std::unordered_set<TensorId> &view_output_ids) {
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "RegisterOutputs: " << outputs.size() << " tensors, timestamp: " << timestamp;

  for (const auto &tensor : outputs) {
    if (tensor != nullptr && view_output_ids.count(tensor->id()) > 0) {
      MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << "  Skipped view output: " << tensor->ToString();
      continue;
    }
    if (!CanOffloadTensor(tensor)) {
      MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << "  Skipped non-offloadable output: "
                                      << (tensor ? tensor->ToString() : "nullptr");
      continue;
    }
    source_group_mgr_->RegisterSource(tensor, timestamp);
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "RegisterOutputs done, source group size: " << source_group_mgr_->Size();
}

void SlidingWindowMgr::PreProcess(const std::string &op_name, const std::vector<ValuePtr> &inputs) {
  std::vector<tensor::TensorPtr> input_tensors;
  for (const auto &val : inputs) {
    CollectTensors(val, &input_tensors);
  }
  if (input_tensors.empty()) {
    return;
  }
  if (!AnyTensorOnAscend(input_tensors)) {
    return;
  }
  size_t expected_output_size = InferOutputSize(op_name, inputs);

  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "PreProcess op=" << op_name << ", inputs=" << input_tensors.size()
                                 << ", expected_output_size=" << (expected_output_size / 1024 / 1024) << " MB";
  runtime::Pipeline::Get().WaitAll();

  PrintAscendMemStatistics("PreProcess::Entry");

  source_group_mgr_->PurgeExpired();
  space_manager_->PurgeExpiredOffloaded();
  MS_VLOG(VL_HYPER_OFFLOAD_DEBUG) << "Source group size (before purge): " << source_group_mgr_->Size();

  size_t memory_needed = 0;
  size_t new_inputs_size = 0;
  size_t cached_inputs_count = 0;

  for (const auto &input : input_tensors) {
    if (input != nullptr) {
      TensorId source_id = source_group_mgr_->ResolveSourceId(input);
      auto *group = source_group_mgr_->GetGroup(source_id);
      bool in_cache =
        (group != nullptr && group->source_registered && group->status == ResidencyStatus::kOnDevice);
      auto addr = input->device_address();
      bool has_device_mem = (addr != nullptr) && (addr->GetDevicePtr() != nullptr);
      if (in_cache && has_device_mem) {
        cached_inputs_count++;
      } else {
        memory_needed += AlignToPoolSize(input->Size());
        new_inputs_size += AlignToPoolSize(input->Size());
      }
    }
  }
  memory_needed += expected_output_size;

  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Memory analysis: needed=" << (memory_needed / 1024 / 1024)
                                 << " MB, new_inputs=" << (new_inputs_size / 1024 / 1024)
                                 << " MB, cached=" << cached_inputs_count << "/" << input_tensors.size()
                                 << ", source_groups=" << source_group_mgr_->Size();

  if (memory_needed > 0) {
    size_t buffer_size = GetSlidingWindowBufferSize();
    if (buffer_size > std::numeric_limits<size_t>::max() - memory_needed) {
      MS_LOG(WARNING) << "Sliding window memory request overflows with buffer size " << buffer_size
                      << ", use maximum size_t value.";
      buffer_size = std::numeric_limits<size_t>::max() - memory_needed;
    }
    auto protected_source_ids = source_group_mgr_->BuildProtectedSourceIdSet(input_tensors);
    if (!space_manager_->EnsureDeviceSpaceEnough(memory_needed + buffer_size, protected_source_ids)) {
      MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "PreProcess could not ensure enough device space for op=" << op_name;
    }
  }

  if (!space_manager_->EnsureInputsOnDevice(input_tensors)) {
    MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "PreProcess could not reload all inputs for op=" << op_name;
  }

  // Update access times for input tensors (and source tensors for views).
  uint64_t timestamp = GetCurrentTimestamp();
  for (const auto &tensor : input_tensors) {
    if (tensor) {
      TensorId source_id = source_group_mgr_->ResolveSourceId(tensor);
      source_group_mgr_->UpdateAccess(source_id, timestamp);
    }
  }

  // Wait for async tasks (ClearDeviceMemory / SetDevicePtr) to complete before the op
  // allocates memory.
  runtime::Pipeline::Get().WaitAll();

  PrintAscendMemStatistics("PreProcess::Exit");
}

void SlidingWindowMgr::PostProcess(const std::string &op_name, const std::vector<ValuePtr> &inputs,
                                   const ValuePtr &output) {
  auto output_tensors = ExtractTensors(output);
  if (output_tensors.empty()) {
    return;
  }
  if (!AnyTensorOnAscend(output_tensors)) {
    return;
  }

  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "PostProcess op=" << op_name << ", outputs=" << output_tensors.size();
  runtime::Pipeline::Get().WaitAll();

  auto input_tensors = CollectInputTensors(inputs);

  std::unordered_set<TensorId> view_output_ids;
  uint64_t timestamp = GetCurrentTimestamp();
  for (const auto &out : output_tensors) {
    if (out == nullptr || out->device_address() == nullptr) {
      continue;
    }
    for (const auto &input : input_tensors) {
      if (input == nullptr || input->device_address() == nullptr) {
        continue;
      }
      if (input->device_address() == out->device_address() && input->id() != out->id()) {
        source_group_mgr_->RegisterView(input, out, timestamp);
        (void)view_output_ids.insert(out->id());
        MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Registered view output: " << out->ToString()
                                       << ", source input id=" << input->id();
        break;
      }
    }
  }

  RegisterOutputs(output_tensors, GetCurrentTimestamp(), view_output_ids);
}

void SlidingWindowMgr::Finalize() {
  runtime::Pipeline::Get().WaitAll();

  PrintAscendMemStatistics("Finalize::Entry");

  source_group_mgr_->PurgeExpired();
  space_manager_->PurgeExpiredOffloaded();
  if (!space_manager_->EnsureAllOffloadedOnDevice()) {
    MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Finalize could not reload all offloaded source groups.";
  }

  PrintAscendMemStatistics("Finalize::Exit");
}

}  // namespace sliding_window
}  // namespace utils
}  // namespace mindspore
