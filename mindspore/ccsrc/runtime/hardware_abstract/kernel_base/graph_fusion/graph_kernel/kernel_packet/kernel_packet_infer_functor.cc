/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_infer_functor.h"
#include <memory>
#include <utility>
#include <mutex>
#include "mindspore/ccsrc/include/utils/symbol_engine/symbol_engine_impl.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_kernel_mod.h"

namespace mindspore::kernel {
BaseShapePtr KernelPacketInfer::InferShape(const AbstractBasePtrList &args) {
  MS_EXCEPTION_IF_NULL(engine_);
  auto engine = engine_->cast_ptr<symshape::SymbolEngineImpl>();
  MS_EXCEPTION_IF_NULL(engine);
  std::lock_guard<std::mutex> infer_lock_guard(*(engine->GetInferMutex()));
  auto ret = SymbolEngineInfer::InferShape(args);
  SaveSymbolicValue();
  return ret;
}

void KernelPacketInfer::SaveSymbolicValue() {
  for (size_t i = 0; i < inner_inputs_abstract_.size(); i++) {
    if (inner_inputs_abstract_[i] == nullptr) {
      continue;
    }
    auto abs = inner_inputs_abstract_[i];
    auto symbolic_value = abs->GetSymbolicValue();
    MS_EXCEPTION_IF_NULL(symbolic_value);
    kernel_mod_holder_.CacheHostValue(i, symbolic_value->ToValueOf(abs->GetType()));
  }
}
}  // namespace mindspore::kernel
