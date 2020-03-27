/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this ${file} except in compliance with the License.
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

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "src/api/tvm_op_module.h"
#include "common/op_utils.h"
#include "lite/api/km_api.h"
#include "src/op.h"
namespace mindspore {
namespace predict {
using OpFunc = std::function<int(const std::vector<DLTensor *> &)>;

class TVMOperator : public OpBase {
 public:
  explicit TVMOperator(OpFunc func) : opfunc(std::move(func)) {}
  ~TVMOperator() override = default;
  int Init(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override { return 0; }
  int Execute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
    std::vector<DLTensor *> dlT;
    for (auto input : inputs) {
      MS_ASSERT(input != nullptr);
      dlT.push_back(input->GetDLTensor());
    }

    for (auto output : outputs) {
      MS_ASSERT(output != nullptr);
      dlT.push_back(output->GetDLTensor());
    }
    return opfunc(dlT);
  }

  static OpBase *CreateOp(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const OpDef &opDef,
                          const Context &ctx, const OpDesc &desc) {
    std::vector<DLTensor *> dlT;
    for (auto input : inputs) {
      MS_ASSERT(input != nullptr);
      dlT.push_back(input->GetDLTensor());
    }

    for (auto output : outputs) {
      MS_ASSERT(output != nullptr);
      dlT.push_back(output->GetDLTensor());
    }

    KernelOption option;
    option.numThreads = ctx.threadNum;
    OpFunc opFunc = GetKernel(opDef, dlT, option);
    if (opFunc != nullptr) {
      auto op = std::unique_ptr<TVMOperator>(new (std::nothrow) TVMOperator(opFunc));
      if (op == nullptr) {
        MS_LOGE("new TVMOperator failed");
      }
      return op.release();
    }
    return nullptr;
  }

 private:
  OpFunc opfunc;
  std::vector<DLTensor *> dltensors;
};

TVMOpRegistry::TVMOpRegistry() = default;

mindspore::predict::OpCreator TVMOpRegistry::GetOpCreator(const mindspore::predict::OpDesc &desc) {
  return TVMOperator::CreateOp;
}

OpRegistry *TVMOpModule::GetInstance() {
  static TVMOpRegistry tvmOpRegistry;
  return &tvmOpRegistry;
}

static TVMOpModule tvmOpModule;

static ModuleRegistrar<Module<OpRegistry>> g_tvmOpReg(MODULE_REG_NAME_OP_REGISTRY, tvmOpModule);
}  // namespace predict
}  // namespace mindspore
