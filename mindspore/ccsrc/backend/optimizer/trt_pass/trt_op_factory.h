/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_PASS_OP_FACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_PASS_OP_FACTORY_H_

#include <functional>
#include <unordered_map>
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <NvInfer.h>
#include "base/base.h"
#include "ir/anf.h"

namespace mindspore {
namespace opt {
class LayerInput;
class TrtConverterContext;
using ConvertResult = std::pair<bool, std::vector<nvinfer1::ITensor *>>;
using ConvertFunc = std::function<ConvertResult(AnfNodePtr, std::shared_ptr<TrtConverterContext>)>;

class TrtOpFactory {
 public:
  static TrtOpFactory &GetInstance() {
    static TrtOpFactory instance;
    return instance;
  }

  void Register(const std::string &op_name, const ConvertFunc &func) {
    if (op_convert_map_.count(op_name)) {
      MS_LOG(EXCEPTION) << "Operator: " << op_name << " re-registered.";
    }
    op_convert_map_.insert(std::make_pair(op_name, func));
  }

  ConvertFunc GetConvertFunc(const std::string &op_name) const {
    auto iter = op_convert_map_.find(op_name);
    if (iter == op_convert_map_.end()) {
      MS_LOG(WARNING) << "Operator: " << op_name << " not support.";
      return nullptr;
    }
    return iter->second;
  }

 private:
  TrtOpFactory() = default;
  ~TrtOpFactory() = default;
  DISABLE_COPY_AND_ASSIGN(TrtOpFactory)

  std::unordered_map<std::string, ConvertFunc> op_convert_map_;
};

class TrtOpRegister {
 public:
  TrtOpRegister(const std::string &op_name, ConvertFunc func) { TrtOpFactory::GetInstance().Register(op_name, func); }
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTITIMIZER_TRT_PASS_OP_FACTORY_H_
