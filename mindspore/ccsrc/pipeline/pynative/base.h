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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_BASE_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_BASE_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <set>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/anf.h"
#include "pybind_api/ir/primitive_py.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "abstract/abstract_value.h"
#include "include/common/utils/stub_tensor.h"
#include "include/common/utils/tensor_future.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;
const size_t kDefaultContainerSize = 5000;

struct BaseOpRunInfo {
  uint64_t py_prim_id_{0};
  bool has_dynamic_output = false;
  bool is_mixed_precision_cast = false;
  bool use_dynamic_shape_process = false;
  bool need_earse_cache = false;
  std::string op_name;
  std::string next_op_name;
  std::string device_target = "Unknown";
#if defined(__APPLE__)
  int next_input_index = 0;
#else
  size_t next_input_index = 0;
#endif
  std::vector<tensor::TensorPtr> input_tensor;
  std::vector<int64_t> input_mask;
  AbstractBasePtr abstract;
  std::vector<size_t> output_indexes;
  std::vector<int64_t> dyn_input_sizes;
  std::vector<tensor::TensorPtr> output_tensors;
};

struct AsyncStatus {
  bool disable_mix_precision{false};
  bool is_jit_compiling{false};
  size_t custom_bprop_cell_count{0};
};

struct OpGradInfo {
  PrimitivePtr op_prim{nullptr};
  abstract::AbstractBasePtrList input_abs{};
  abstract::AbstractBasePtr out_abs{nullptr};
  std::vector<ValuePtr> input_value{};
  ValuePtr out_value{nullptr};
  std::vector<TensorGradType> input_value_grad_type{};
  // Currently only packfunc will use the grad_graph_id, and it will not be used in other scenarios.
  // Since the current grad process uses the prim in FrontendOpRunInfo, not the prim in BackendOpRunInfo,
  // the grad_graph_id cannot be placed in the prim attr during the async run,
  // and the grad_graph_id will be able to the prim attr later.
  int64_t grad_graph_id{-1};
};
using OpGradInfoPtr = std::shared_ptr<OpGradInfo>;

struct GradParam {
  GradParam(OpGradInfoPtr op_grad_info, bool use_dynamic_shape_process)
      : op_grad_info(op_grad_info), use_dynamic_shape_process(use_dynamic_shape_process) {
    input_size = op_grad_info->input_value.size();
  }

  OpGradInfoPtr op_grad_info;

  // Dynamic shape or dynamic structure
  bool use_dynamic_shape_process{false};

  // For other used
  bool out_used_in_bporp_graph{false};
  bool is_control_flow{false};
  size_t input_size{0};

  // For jit domain
  bool has_added_v{false};
  bool is_jit_graph{false};
  bool is_jit_self_dynamic_shape{false};

  // For KPynativeWithFProp used
  FuncGraphPtr fg{nullptr};
  // grad func graph for jit or fg
  FuncGraphPtr source_fg{nullptr};
  // Op forward output used in bprop graph
  std::string graph_cache_key;
  // Used for pyexecute
  CNodePtr cnode;
};

using GradParamPtr = std::shared_ptr<GradParam>;

struct FrontendOpRunInfo {
  FrontendOpRunInfo() { op_grad_info = std::make_shared<OpGradInfo>(); }
  OpGradInfoPtr op_grad_info;

  BaseOpRunInfo base_op_run_info;
  bool run_in_vm = false;
  bool requires_grad = false;
  bool output_get_by_infer_value = false;
  bool should_be_cache = false;
  bool is_jit_input = false;
  bool is_view_op = false;
  int mix_type{0};
  size_t input_size = 0;
  // real_out return to python; out_value in OpGradInfo may be fake value;
  ValuePtr real_out{nullptr};
  std::string op_info;
  std::string out_value_id;
  std::string cell_obj_id;
  std::vector<bool> input_unused_in_bprop{};
  // Hold tensorGradType
  std::vector<std::string> input_value_id{};
  stub::StubNodePtr stub_output{nullptr};
  std::vector<Signature> signatures{};
  AsyncStatus async_status;
  mindspore::HashSet<size_t> input_to_attr{};
  std::vector<DeviceAddressPromisePtr> device_sync_promises;
};
using FrontendOpRunInfoPtr = std::shared_ptr<FrontendOpRunInfo>;

struct InputArgsInfo {
  InputArgsInfo() = default;
  ~InputArgsInfo() = default;
  InputArgsInfo(bool is_grad_topest_cell, bool is_high_order_top_cell, bool grad_is_running, size_t obj_order)
      : is_grad_topest_cell(is_grad_topest_cell),
        is_high_order_top_cell(is_high_order_top_cell),
        grad_is_running(grad_is_running),
        obj_order(obj_order) {}

  bool is_grad_topest_cell;
  bool is_high_order_top_cell;
  bool grad_is_running;
  size_t obj_order;

  bool has_custom_bprop{false};
  bool has_sens{false};
  PrimitivePyPtr custom_bprop_prim{nullptr};
  ValuePtr out_value{nullptr};
  std::string obj_id;
  std::string cell_id;
  std::string already_run_cell_id;
  std::string input_args_id;
  // Cell unique id, cell_id + cell_order;
  std::string obj_order_id;
  size_t input_size = 0;
  size_t grad_order = 0;
  std::vector<std::string> input_arg_id_vec;
  std::vector<ValuePtr> input_arg_value_vec;
  // Used for dynamic shape auto detect
  std::vector<abstract::BaseShapePtr> input_arg_base_shape_vec;

  // Free memory
  void Reset() {
    custom_bprop_prim = nullptr;
    out_value = nullptr;
    input_arg_value_vec.clear();
  }
};
using InputArgsInfoPtr = std::shared_ptr<InputArgsInfo>;

class FastValue {
 public:
  FastValue() = default;
  ~FastValue() = default;

  explicit FastValue(const int64_t &v) : int_value_(v), is_int_{true} {}
  explicit FastValue(std::vector<int64_t> v) : vec_value_(std::move(v)), is_int_{false} {}

  bool is_int() const { return is_int_; }
  int64_t int_value() const { return int_value_; }
  const std::vector<int64_t> &vec_value() const { return vec_value_; }

 private:
  int64_t int_value_;
  std::vector<int64_t> vec_value_;
  bool is_int_{false};
};
using FastValuePtr = std::shared_ptr<FastValue>;

struct SliceOpInfo {
  SliceOpInfo() = default;
  ~SliceOpInfo() = default;
  std::string slice_op_name;
  std::vector<size_t> data_indexs;
  std::vector<FastValuePtr> slice_index_inputs;
};
using SliceOpInfoPtr = std::shared_ptr<SliceOpInfo>;

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_BASE_H_
