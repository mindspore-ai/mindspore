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
#include "pipeline/jit/parse/parse.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace pynative {
namespace py = pybind11;

struct BaseOpRunInfo {
  bool has_dynamic_input = false;
  bool has_dynamic_output = false;
  bool is_mixed_precision_cast = false;
  bool lazy_build = false;
  std::string op_name;
  std::string next_op_name;
  std::string graph_info;
  std::string device_target = "Unknown";
#if defined(__APPLE__)
  int next_input_index = 0;
#else
  size_t next_input_index = 0;
#endif
  std::vector<tensor::TensorPtr> input_tensor;
  std::vector<int64_t> input_mask;
  AbstractBasePtr abstract;
};

struct FrontendOpRunInfo {
  BaseOpRunInfo base_op_run_info;
  bool run_in_vm = false;
  bool is_nop_prim = false;
  bool output_get_by_infer_value = false;
  int mix_type{0};
  uint64_t py_prim_id{0};
  PrimitivePyPtr op_prim;
  std::string op_info;
  // Tensor input and with its value
  std::vector<std::pair<size_t, ValuePtr>> index_with_value;
  std::vector<AbstractBasePtr> input_abs;
  std::vector<ValuePtr> input_value;
  py::list op_inputs;
};
using FrontendOpRunInfoPtr = std::shared_ptr<FrontendOpRunInfo>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_BASE_H_
