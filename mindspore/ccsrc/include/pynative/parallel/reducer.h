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

#ifndef MINDSPORE_CCSRC_PYNATIVE_PARALLEL_REDUCER_H_
#define MINDSPORE_CCSRC_PYNATIVE_PARALLEL_REDUCER_H_

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "pynative/utils/base.h"
#include "include/utils/pynative//variable.h"
#include "include/backend/visible.h"
#include "include/pynative/utils/pyboost/comm_handle.h"

namespace mindspore {
namespace pynative {
namespace distributed {
PYNATIVE_EXPORT std::vector<int> _find_unused_parameters(const tensor::TensorPtrList &outputs,
                                                         const tensor::TensorPtrList &parameters_);
class PYNATIVE_EXPORT Reducer {
 public:
  // Init
  Reducer() = delete;
  ~Reducer() = default;
  Reducer(tensor::TensorPtrList parameters, std::string process_group, size_t bucket_cap_mb, bool grad_reduce_in_fp32,
          bool average_in_collective, bool enable_mem_align, bool static_graph_);

  // main API func
  void prepare_for_backward(const tensor::TensorPtrList &outputs);
  void prepare_for_forward();

  bool should_rebuild_buckets();
  void rebuild_buckets();
  tensor::TensorPtrList find_unused_parameters(const tensor::TensorPtrList &outputs);
  void zero_grad();
  tensor::TensorPtrList get_bucket_for_debug();
  std::vector<std::vector<size_t>> bucket_indices;  // list that describes the buckets and the containing param idxs

 private:
  // inner member
  using TensorPtrSet = std::unordered_set<tensor::TensorPtr>;
  struct Bucket {
    tensor::TensorPtrList parameters;
    TensorPtrSet ready_params;
    std::vector<size_t> offsets;
    tensor::TensorPtrList bucket_views;
    tensor::TensorPtr gradients;
    size_t bucket_size;
    std::tuple<tensor::TensorPtr, kernel::pyboost::CommHandlePtr> comm_handle;
  };

  tensor::TensorPtrList parameters_;
  tensor::TensorPtrList unused_params;
  std::vector<Bucket> buckets_;
  using BucketIndexer = std::pair<size_t, size_t>;
  std::vector<BucketIndexer> variable_locator;                   // map param global idx -> (bucket_idx, inner_idx)
  std::unordered_map<tensor::TensorPtr, size_t> global_locator;  // map param ptr to its global idx

  tensor::TensorPtrList rebuilt_params_;
  tensor::TensorPtr gradient_scaling_factor;
  std::string process_group_;
  device::DeviceType device_target_;
  std::int64_t world_size_;
  size_t buckets_pending;
  size_t bucket_cap_mb_;
  bool grad_reduce_in_fp32_;
  bool average_in_collective_;
  bool bucket_rebuilt;
  bool static_graph_;
  bool find_unused_parameters_;
  bool has_marked_unused_params_;
  bool expect_comm_reduce;

  // comm related func
  void initialize_bucket_views();
  void mark_bucket_ready(size_t bucket_index);
  void all_reduce_bucket(Bucket *bucket);

  // inner helper func
  void register_backward_hooks();
  void initialize_buckets(tensor::TensorPtrList &);
  void PrepareOpStatus();
};

}  // namespace distributed
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_PARALLEL_REDUCER_H_
