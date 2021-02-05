/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_GRU_H_
#define MINDSPORE_CORE_OPS_GRU_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameGRU = "GRU";
class GRU : public PrimitiveC {
 public:
  GRU() : PrimitiveC(kNameGRU) {
    InitIOName({"x", "weight_input", "weight_hidden", "bias_input", "bias_hidden", "seq_length", "init_h"},
               {"output", "output_h", "update", "reset", "new", "hidden_new"});
  }
  ~GRU() = default;
  MS_DECLARE_PARENT(GRU, PrimitiveC);
  void Init(const bool bidirectional = false, const int64_t cell_depth = 1, const float keep_prob = 1.0,
            const float cell_clip = -1.0, const int64_t num_proj = 0, const bool time_major = true,
            const bool reset_after = true, const bool is_training = true,
            const ActivationType activation = ActivationType::TANH,
            const GateOrderMode gate_order = GateOrderMode::RZH);

  void set_bidirectional(const bool bidirectional);
  void set_cell_depth(const int64_t cell_depth);
  void set_keep_prob(const float keep_prob);
  void set_cell_clip(const float cell_clip);
  void set_num_proj(const int64_t num_proj);
  void set_time_major(const bool time_major);
  void set_reset_after(const bool reset_after);
  void set_is_training(const bool is_training);
  void set_activation(const ActivationType activation);
  void set_gate_order(const GateOrderMode gate_order);

  bool get_bidirectional() const;
  int64_t get_cell_depth() const;
  float get_keep_prob() const;
  float get_cell_clip() const;
  int64_t get_num_proj() const;
  bool get_time_major() const;
  bool get_reset_after() const;
  bool get_is_training() const;
  ActivationType get_activation() const;
  GateOrderMode get_gate_order() const;
};

AbstractBasePtr GRUInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args);
using PrimGRUPtr = std::shared_ptr<GRU>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_GRU_H_
