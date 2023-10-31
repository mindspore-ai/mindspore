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

#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/auto_generate/baddbmm.h"
#include "kernel/pyboost/auto_generate/add_ext.h"
#include "kernel/pyboost/auto_generate/mul.h"
#include "kernel/pyboost/auto_generate/batch_matmul.h"
#include "kernel/pyboost/auto_generate/matmul.h"
#include "kernel/pyboost/auto_generate/bias_add.h"
#include "kernel/pyboost/auto_generate/linear.h"
#include "kernel/pyboost/auto_generate/square.h"
#include "kernel/pyboost/auto_generate/transpose.h"
#include "kernel/pyboost/auto_generate/view.h"
#include "kernel/pyboost/auto_generate/bmm.h"
#include "kernel/pyboost/auto_generate/exp.h"
#include "kernel/pyboost/auto_generate/erf.h"
#include "kernel/pyboost/auto_generate/silu.h"
#include "kernel/pyboost/auto_generate/sin.h"
#include "kernel/pyboost/auto_generate/cos.h"
#include "kernel/pyboost/auto_generate/cast.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

template <typename T>
OpFactory<T> &OpFactory<T>::Get() {
  static OpFactory<T> instance;
  return instance;
}

template <typename T>
std::shared_ptr<T> OpFactory<T>::Create(const string &name, const string &device) {
  auto iter = op_creater_.find(device);
  if (iter == op_creater_.end()) {
    MS_LOG(EXCEPTION) << "Not found op " << name << " on device " << device;
  }
  return iter->second();
}

template class OpFactory<Baddbmm>;
template class OpFactory<AddExt>;
template class OpFactory<Mul>;
template class OpFactory<BatchMatmul>;
template class OpFactory<BiasAdd>;
template class OpFactory<Matmul>;
template class OpFactory<Linear>;
template class OpFactory<Square>;
template class OpFactory<Transpose>;
template class OpFactory<View>;
template class OpFactory<Bmm>;
template class OpFactory<Exp>;
template class OpFactory<Erf>;
template class OpFactory<SiLU>;
template class OpFactory<Cos>;
template class OpFactory<Sin>;
template class OpFactory<Cast>;
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
