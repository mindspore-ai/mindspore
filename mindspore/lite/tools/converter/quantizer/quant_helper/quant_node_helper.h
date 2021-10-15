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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PREPROCESSOR_QUANT_NODE_PREPROCESSOR_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PREPROCESSOR_QUANT_NODE_PREPROCESSOR_H

#include <unordered_map>
#include <memory>
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore::lite {
class QuantNodeBase {
 public:
  void UpdateQuantParamsNum(const schema::MetaGraphT &graph, const schema::CNodeT &node);

 protected:
  size_t input_inited_quant_params_ = 0;
  size_t output_inited_quant_params_ = 0;
};

class QuantParamPropogator : public QuantNodeBase {
 public:
  virtual STATUS PropogateQuantParams(schema::MetaGraphT *graph, const schema::CNodeT &node) { return RET_OK; }
};

class QuantTypeDeterminer : public QuantNodeBase {
 public:
  virtual bool DetermineQuantAll(const schema::MetaGraphT &graph, schema::CNodeT *node);
  virtual bool DetermineQuantWeight(const schema::MetaGraphT &graph, schema::CNodeT *node);
};

class QuantNodeHelper {
 public:
  int NodeQuantPreprocess(schema::MetaGraphT *graph, schema::CNodeT *node);
  QuantNodeHelper(std::shared_ptr<QuantParamPropogator> quant_param_propogator,
                  std::shared_ptr<QuantTypeDeterminer> quant_type_determiner) {
    quant_param_propogator_ = quant_param_propogator;
    quant_type_determiner_ = quant_type_determiner;
  }
  virtual ~QuantNodeHelper() = default;

 protected:
  std::shared_ptr<QuantParamPropogator> quant_param_propogator_;
  std::shared_ptr<QuantTypeDeterminer> quant_type_determiner_;
};

class QuantHelperRegister {
 public:
  virtual ~QuantHelperRegister();
  QuantNodeHelper *GetQuantHelper(schema::PrimitiveType op_type);
  static QuantHelperRegister *GetInstance();

 private:
  QuantHelperRegister();
  std::unordered_map<schema::PrimitiveType, QuantNodeHelper *> register_map_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PREPROCESSOR_QUANT_NODE_PREPROCESSOR_H
