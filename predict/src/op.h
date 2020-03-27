/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_SRC_OP_H_
#define PREDICT_SRC_OP_H_

#include <string>
#include <vector>
#include "include/context.h"
#include "include/tensor.h"
#include "include/errorcode.h"
#include "schema/inner/ms_generated.h"

#define MSPREDICT_API __attribute__((visibility("default")))

namespace mindspore {
namespace predict {
enum OP_ARCH { X86_FP32, X86_INT8, ARM_FP32, ARM_FP16, ARM_INT8, GPU };

struct MSPREDICT_API OpDesc {
  OP_ARCH arch;
  OpT type;

  bool operator<(const OpDesc &dst) const { return (arch < dst.arch) || (type < dst.type); }
};

class MSPREDICT_API OpBase {
 public:
  OpBase();
  virtual ~OpBase();

  virtual int Execute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) = 0;
  virtual int Init(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) = 0;

 protected:
  const OpDesc *desc;
  std::string name;
};

typedef OpBase *(*OpCreator)(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                             const OpDef &opDef, const Context &ctx, const OpDesc &desc);
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_OP_H_
