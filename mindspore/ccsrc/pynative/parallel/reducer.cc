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

#include "include/pynative/parallel/reducer.h"
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <algorithm>
#include "include/cluster/init.h"
#include "tools/profiler/profiler.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "pynative/backward/hook/hook_py.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/pynative/utils/pynative_execute.h"
#include "pynative/backward/op_grad/func_grad.h"

namespace mindspore {
namespace pynative {
namespace distributed {

Reducer::Reducer(tensor::TensorPtrList parameters, std::string process_group, size_t bucket_cap_mb,
                 bool grad_reduce_in_fp32, bool average_in_collective, bool static_graph, bool find_unused_parameters)
    : parameters_(std::move(parameters)),
      process_group_(std::move(process_group)),
      bucket_cap_mb_(bucket_cap_mb),
      grad_reduce_in_fp32_(grad_reduce_in_fp32),
      average_in_collective_(average_in_collective),
      bucket_rebuilt(false),
      static_graph_(static_graph),
      find_unused_parameters_(find_unused_parameters),
      has_marked_unused_params_(false),
      expect_comm_reduce(false) {
  world_size_ = mindspore::distributed::collective::CollectiveManager::instance()->GetGroupSize(process_group_);
  gradient_scaling_factor = tensor::from_scalar(1.0f / world_size_);
  PrepareOpStatus();
  initialize_buckets(parameters_);
  initialize_bucket_views();
  register_backward_hooks();
}

void Reducer::PrepareOpStatus() {
  runtime::Pipeline::Get().WaitFrontend();

  const auto &pynative_executor = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(pynative_executor);
  device_target_ = pynative_executor->forward_executor()->device_target();
  kernel::pyboost::OpStatus status{false, device_target_};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
}

void Reducer::register_backward_hooks() {
  for (size_t variable_idx = 0; variable_idx < parameters_.size(); variable_idx++) {
    auto param = parameters_[variable_idx];

    auto hook = [this, variable_idx](const tensor::TensorPtr &grad) {
      MS_LOG(DEBUG) << "grad received in the hook";
      auto [bucket_idx, inner_idx] = variable_locator[variable_idx];
      auto &bucket = buckets_[bucket_idx];
      auto &param = parameters_[variable_idx];

      MS_LOG(DEBUG) << "In the autograd hook: received grad " << grad->ToString() << std::endl
                    << " for leaf node " << param->ToString() << std::endl
                    << " with its bucket_view shape" << bucket.bucket_views[inner_idx]->shape()
                    << " is contiguous: " << grad->is_contiguous();

      kernel::pyboost::OpStatus status{false, device_target_};
      kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
      auto alpha = std::make_shared<Int64Imm>(1)->cast<ScalarPtr>();
      auto res = kernel::pyboost::inplace_add_ext(bucket.bucket_views[inner_idx], grad, alpha);
      // if no sync is enabled, no need to update pending counter
      if (!expect_comm_reduce) {
        MS_LOG(DEBUG) << "Reduce comm is disabled in the hook.";
        return res;
      }

      // mark unused params when sync is enabled
      if (!has_marked_unused_params_) {
        has_marked_unused_params_ = true;
        for (auto &cur_param : unused_params) {
          auto variable_idx = global_locator[cur_param];
          auto bucket_idx = variable_locator[variable_idx].first;
          auto &cur_bucket = buckets_[bucket_idx];
          cur_bucket.ready_params.insert(cur_param);
          MS_LOG(DEBUG) << "marking unused_param in the hook:" << cur_param->ToString() << " in bucket " << bucket_idx;
          mark_bucket_ready(bucket_idx);
        }
      }

      auto succuss = bucket.ready_params.insert(param).second;
      if (succuss) {
        mark_bucket_ready(bucket_idx);
        if (should_rebuild_buckets()) {
          rebuilt_params_.push_back(param);
        }
      }
      return res;
    };
    autograd::RegisterHook::RegisterCppTensorBackwardHook(param, hook);
  }
}

bool Reducer::should_rebuild_buckets() { return (static_graph_ || !find_unused_parameters_) && !bucket_rebuilt; }

void Reducer::mark_bucket_ready(size_t bucket_index) {
  // locate bucket idx
  auto &bucket = buckets_[bucket_index];

  // if bucket ready, trigger comm func
  MS_LOG(DEBUG) << "mark bucket idx {" << bucket_index << "} with ready count {" << bucket.ready_params.size()
                << "} and bucket size {" << bucket.parameters.size() << "}";
  if (bucket.ready_params.size() == bucket.parameters.size()) {
    MS_LOG(DEBUG) << "Bucket is ready; issue bucket." << std::endl;
    all_reduce_bucket(&bucket);
    MS_LOG(DEBUG) << "Bucket issued!" << std::endl;

    // if last bucket
    if (--buckets_pending == 0) {
      MS_LOG(DEBUG) << "all bucket are issued";
      // finalize bucket
      PyNativeExecutor::grad_executor()->QueueFinalCallback([this]() {
        runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                           "FinalizeBuckets", false);
        MS_LOG(DEBUG) << "Finalizing buckets";
        for (auto &bucket : buckets_) {
          auto &[target, handle] = bucket.comm_handle;
          handle->Wait();
          handle = nullptr;
          if (average_in_collective_) {
            kernel::pyboost::inplace_div(target, tensor::from_scalar(this->world_size_));
          }
        }
        if (should_rebuild_buckets()) {
          this->rebuilt_params_.insert(this->rebuilt_params_.end(), this->unused_params.begin(),
                                       this->unused_params.end());
        }
        this->buckets_pending = buckets_.size();
        this->expect_comm_reduce = false;
        MS_LOG(DEBUG) << "Finalizing done";
      });
    }
  }
}

void Reducer::all_reduce_bucket(Bucket *bucketPtr) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     "BucketAllReducer", false);
  bucketPtr->ready_params.clear();
  std::string op_type = "sum";
  if (!average_in_collective_) {
    kernel::pyboost::inplace_mul(bucketPtr->gradients, gradient_scaling_factor);
  }
  // inplace allreduce
  bucketPtr->comm_handle = mindspore::kernel::pyboost::dist_comm_all_reduce(
    bucketPtr->gradients, std::make_shared<mindspore::StringImm>(op_type),
    std::make_shared<mindspore::StringImm>(process_group_));
}

// update bucket views mapping
void Reducer::initialize_bucket_views() {
  for (auto &bucket : buckets_) {
    const auto gradients_ptr = bucket.gradients;
    for (size_t i = 0; i < bucket.parameters.size(); i++) {
      auto &v = bucket.parameters[i];  // ignore storage_info
      const auto offset = bucket.offsets[i];
      auto sliced_grad = kernel::pyboost::slice(gradients_ptr, {(int64_t)(offset)}, {(int64_t)(v->DataSize())});
      auto strided_grad = kernel::pyboost::view(sliced_grad, v->shape());

      bucket.bucket_views.push_back(strided_grad);
      MS_LOG(DEBUG) << "initialize param grad view:" << v->ToString() << " with start " << offset << " and size "
                    << v->DataSize();
    }
  }
}

tensor::TensorPtrList Reducer::get_bucket_for_debug() {
  tensor::TensorPtrList ret;
  std::transform(buckets_.begin(), buckets_.end(), std::back_inserter(ret), [](auto x) { return x.gradients; });
  return ret;
}

// init buckets
// 1. first bucket is 1M by default
// 2. build buckets
// 3. update metadata
size_t kDefaultBucketCapMb = 25;
size_t kDefaultFirstBucketCapByte = 1024 * 1024;
bool disable_bucketing_if_rebuild = true;
void Reducer::initialize_buckets(tensor::TensorPtrList &parameters) {
  size_t current_bucket_size = 0;
  Bucket current_bucket;
  TypeId bucket_dtype = kTypeUnknown;
  bool inited = !global_locator.empty();

  // if rebuild enabled, do not bucketing
  size_t current_bucket_size_limit;
  if (disable_bucketing_if_rebuild && !inited && should_rebuild_buckets()) {
    current_bucket_size_limit = std::numeric_limits<size_t>::max();
  } else {
    current_bucket_size_limit = kDefaultFirstBucketCapByte;
  }

  std::vector<size_t> bucket_indice;

  // traverse from beginning to construct a small bucket at first by default
  for (size_t i = 0; i < parameters.size(); ++i) {
    size_t global_idx = 0;
    auto &param = parameters[i];

    if (!inited) {
      global_locator[param] = i;
      global_idx = i;
    } else {
      global_idx = global_locator[param];
    }
    size_t itemsize = (size_t)param->DataItemSize();
    if (bucket_dtype == kTypeUnknown) {
      bucket_dtype = param->data_type();
    } else {
      MS_EXCEPTION_IF_CHECK_FAIL(bucket_dtype == param->data_type(),
                                 "All parameters in a bucket must have the same dtype.");
    }
    current_bucket.offsets.push_back(current_bucket_size);
    current_bucket.parameters.push_back(param);
    bucket_indice.push_back(global_idx);
    current_bucket_size += param->DataSize();                           // number of elements
    if (current_bucket_size * itemsize >= current_bucket_size_limit) {  // create new according to bucket_cap_mb_
      current_bucket.bucket_size = current_bucket_size;
      current_bucket.gradients =
        tensor::from_spec(bucket_dtype, ShapeVector({ssize_t(current_bucket_size)}), device_target_);
      MS_EXCEPTION_IF_NULL(current_bucket.gradients);
      buckets_.push_back(std::move(current_bucket));
      bucket_indices.push_back(std::move(bucket_indice));  // note
      MS_LOG(DEBUG) << "bucketing " << buckets_[buckets_.size() - 1].parameters.size()
                    << " params with total number of elements " << current_bucket_size << " and bucket size limit "
                    << current_bucket_size_limit << " and total buckets size" << buckets_.size();

      bucket_indice.clear();
      current_bucket = Bucket();
      current_bucket_size = 0;
      current_bucket_size_limit = bucket_cap_mb_ * kDefaultFirstBucketCapByte;
    }
  }

  if (!current_bucket.parameters.empty()) {
    MS_LOG(DEBUG) << "bucketing left " << current_bucket.parameters.size() << " params with total bucket size "
                  << current_bucket_size;
    current_bucket.bucket_size = current_bucket_size;
    current_bucket.gradients =
      tensor::from_spec(bucket_dtype, ShapeVector({ssize_t(current_bucket_size)}), device_target_);
    MS_EXCEPTION_IF_NULL(current_bucket.gradients);
    buckets_.push_back(std::move(current_bucket));
    bucket_indices.push_back(std::move(bucket_indice));
  }
  buckets_pending = buckets_.size();

  // reverse map
  variable_locator.reserve(parameters.size());
  for (size_t bucket_idx = 0; bucket_idx < bucket_indices.size(); bucket_idx++) {
    auto &cur_bucket_indice = bucket_indices[bucket_idx];
    for (size_t inner_idx = 0; inner_idx < cur_bucket_indice.size(); inner_idx++) {
      auto variable_idx = cur_bucket_indice[inner_idx];
      variable_locator[variable_idx] = std::make_pair(bucket_idx, inner_idx);
      MS_LOG(DEBUG) << "initialized buckets: param " << parameters_[variable_idx]->ToString() << " ,variable_idx "
                    << variable_idx << " ,bucket_idx " << bucket_idx << " ,inner_idx " << inner_idx << " ,param offset "
                    << buckets_[bucket_idx].offsets[inner_idx];
    }
  }

  zero_grad();
}

void Reducer::zero_grad() {
  runtime::Pipeline::Get().WaitFrontend();
  for (auto &bucket : buckets_) {
    auto outputs = kernel::pyboost::inplace_zero(bucket.gradients);
  }
}

void Reducer::prepare_for_backward(const tensor::TensorPtrList &outputs) {
  static bool triggered_once = false;
  expect_comm_reduce = true;
  has_marked_unused_params_ = false;

  runtime::Pipeline::Get().WaitFrontend();
  runtime::Pipeline::Get().WaitBpropStage();

  // if static graph, find unused params only once
  if (static_graph_ && !triggered_once) {
    (void)find_unused_parameters(outputs);
    triggered_once = true;
    find_unused_parameters_ = false;
  } else if (find_unused_parameters_) {
    (void)find_unused_parameters(outputs);
  }
}

void Reducer::prepare_for_forward() {}

// Note: The `unused_params` contain pruning params which are in the forward path but not in the backward path
//   and those params' hook will also be triggered.
tensor::TensorPtrList Reducer::find_unused_parameters(const tensor::TensorPtrList &outputs) {
  unused_params = autograd::SearchUnusedParameters(outputs, parameters_);
  MS_LOG(DEBUG) << "unused param count:" << unused_params.size() << std::endl;
  return unused_params;
}

std::vector<int> _find_unused_parameters(const tensor::TensorPtrList &outputs,
                                         const tensor::TensorPtrList &parameters_) {
  auto unused_params = autograd::SearchUnusedParameters(outputs, parameters_);
  std::vector<int> ret;

  for (const auto &param : unused_params) {
    for (size_t idx = 0; idx < parameters_.size(); ++idx) {
      if (parameters_[idx] == param) {
        ret.emplace_back(idx);
      }
    }
  }
  MS_LOG(DEBUG) << "unused param count:" << unused_params.size() << " and idx:" << ret << std::endl;
  return ret;
}

void Reducer::rebuild_buckets() {
  // save old bucket info
  if (!should_rebuild_buckets() || rebuilt_params_.empty()) {
    MS_LOG(DEBUG) << "Reducer buckets should not be rebuilt in this iteration with should_rebuild_buckets "
                  << should_rebuild_buckets();
    return;
  }
  auto old_buckets = std::move(buckets_);
  auto old_bucket_indices = std::move(bucket_indices);
  MS_LOG(DEBUG) << "Reducer buckets have been rebuilt in this iteration.";

  // reinit buckets
  MS_LOG(DEBUG) << "rebuilding: rebuilt_params size " << rebuilt_params_.size() << " parameter size "
                << parameters_.size();
  for (auto x : rebuilt_params_) {
    MS_LOG(DEBUG) << x->ToString() << " ";
  }

  initialize_buckets(rebuilt_params_);
  initialize_bucket_views();

  // copy old buffers to rebuilt buffers
  for (size_t old_bucket_idx = 0; old_bucket_idx < old_buckets.size(); old_bucket_idx++) {
    auto &old_bucket = old_buckets[old_bucket_idx];
    auto &old_indice = old_bucket_indices[old_bucket_idx];
    MS_LOG(DEBUG) << "rebuilding: old_bucket_idx" << old_bucket_idx;
    for (size_t old_inner_idx = 0; old_inner_idx < old_bucket.parameters.size(); old_inner_idx++) {
      size_t variable_idx = old_indice[old_inner_idx];
      MS_LOG(DEBUG) << "rebuilding: variable idx: " << variable_idx;
      auto [new_bucket_idx, new_inner_idx] = variable_locator[variable_idx];
      MS_LOG(DEBUG) << "rebuilding: inplace copy old gradient data"
                    << buckets_[new_bucket_idx].bucket_views[new_inner_idx]->shape() << " "
                    << old_bucket.bucket_views[old_inner_idx]->shape();
      auto res =
        kernel::pyboost::inplace_copy(buckets_[new_bucket_idx].bucket_views[new_inner_idx],
                                      old_bucket.bucket_views[old_inner_idx], std::make_shared<BoolImm>(false));
    }
  }
  bucket_rebuilt = true;
}

}  // namespace distributed
}  // namespace pynative
}  // namespace mindspore
