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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_AD_ADJOINT_H_
#define MINDSPORE_CCSRC_OPTIMIZER_AD_ADJOINT_H_

#include <memory>
#include <utility>
#include <vector>

#include "ir/anf.h"
#include "optimizer/opt.h"

namespace mindspore {
namespace ad {
class Adjoint {
 public:
  Adjoint(const AnfNodePtr &primal, const AnfNodePtr &k, const FuncGraphPtr &caller);
  ~Adjoint() = default;
  AnfNodePtr primal();
  AnfNodePtr k();
  void UpdateK(const AnfNodePtr &k);
  void RegisterKUser(const CNodePtr &user, size_t index);
  AnfNodePtr dout();
  void AccumulateDout(const AnfNodePtr &dout_factor);
  void RegisterDoutUser(const CNodePtr &user, size_t index);
  void CallDoutHole();

 private:
  AnfNodePtr primal_;
  FuncGraphPtr caller_;
  // For ```def f(x): return expr```, The representation graph k is ```def kf(kx): return expr, bprop{expr}```.
  AnfNodePtr k_;
  std::vector<std::pair<CNodePtr, size_t>> k_user_;
  AnfNodePtr dout_;
  AnfNodePtr dout_hole_;
  std::vector<std::pair<CNodePtr, size_t>> dout_user_;
};

using AdjointPtr = std::shared_ptr<Adjoint>;
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_AD_ADJOINT_H_
