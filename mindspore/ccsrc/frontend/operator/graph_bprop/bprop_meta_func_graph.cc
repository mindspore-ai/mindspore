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

#include "frontend/operator/graph_bprop/bprop_meta_func_graph.h"
#include "frontend/operator/graph_bprop/bprop_expander_meta_func_graph.h"

namespace mindspore {
namespace graph_bprop {
PrimitiveBpropImplMap *GetPrimitiveBpropImplMapPtr() {
  static PrimitiveBpropImplMap prim_bprop_impl_map{};
  return &prim_bprop_impl_map;
}

const PrimitiveBpropImplMap &GetPrimitiveBpropImplMap() { return *GetPrimitiveBpropImplMapPtr(); }

void RegBpropMetaFuncGraph() {
  RegArrayOps();
  RegMathOps();
  RegNNOps();
  RegBpropExpanderOps();
}
}  // namespace graph_bprop
}  // namespace mindspore
