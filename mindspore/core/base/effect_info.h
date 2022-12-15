/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_EFFECT_INFO_H_
#define MINDSPORE_CORE_EFFECT_INFO_H_

namespace mindspore {
struct EffectInfo {
  enum State : unsigned char {
    kUnknown = 0,
    kDetecting = 1,
    kDetected = 2,
  };
  State state = kUnknown;  // effect info state.
  bool memory = false;     // memory side effects, e.g., access global variable.
  bool io = false;         // IO side effects, e.g., print message.
  bool load = false;       // load value from global variable, e.g. add(self.para, x).
  bool back_mem = false;

  void Merge(const EffectInfo &info) {
    if (info.state != EffectInfo::kDetected) {
      state = EffectInfo::kDetecting;
    }
    memory = memory || info.memory;
    io = io || info.io;
    load = load || info.load;
    back_mem = back_mem || info.back_mem;
  }
};

// EffectInfoHolder as base class for effect info holders, such as CNode, FuncGraph, etc.
class EffectInfoHolder {
 public:
  // Gets effect info.
  const EffectInfo &GetEffectInfo() const { return effect_info_; }

  // Set effect info.
  void SetEffectInfo(const EffectInfo &info) { effect_info_ = info; }

  virtual ~EffectInfoHolder() = default;

 protected:
  EffectInfo effect_info_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_EFFECT_INFO_H_
