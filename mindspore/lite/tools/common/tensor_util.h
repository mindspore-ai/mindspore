/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_TENSOR_UTIL_H
#define MINDSPORE_LITE_TOOLS_COMMON_TENSOR_UTIL_H

#include <cmath>
#include <unordered_map>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "ir/dtype/type_id.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
using schema::CNodeT;
using schema::Format;
using schema::FusedBatchNormT;
using schema::MetaGraphT;
using schema::QuantParamT;
using schema::TensorT;
using schema::Format::Format_NCHW;
using schema::Format::Format_NHWC;

std::unique_ptr<QuantParamT> GetTensorQuantParam(const std::unique_ptr<TensorT> &tensor);

size_t GetElementSize(const TensorT &tensor);

size_t GetElementSize(const TypeId &dataType);

size_t GetShapeSize(const TensorT &tensor);

size_t GetShapeSize(const std::vector<int32_t> &shape);

std::unique_ptr<TensorT> CopyTensorDefT(const std::unique_ptr<TensorT> &);

size_t GetRefCount(schema::MetaGraphT *graphT, uint32_t tensorIdx);

std::unique_ptr<schema::QuantParamT> CopyQuantParamT(const std::unique_ptr<schema::QuantParamT> &srcQuantParam);

std::unique_ptr<schema::QuantParamT> CopyQuantParamArrayT(
  const std::unique_ptr<schema::QuantParamT> &srcQuantParamArray);

enum Category { CONST = 0, GRAPH_INPUT = 1, OP_OUTPUT = 2, TF_CONST = 3 };

class TensorCache {
 public:
  TensorCache() = default;

  ~TensorCache() { tensors.clear(); }

  int AddTensor(const std::string &name, TensorT *tensor, int Category) {
    index++;
    if (Category == CONST || Category == TF_CONST || Category == GRAPH_INPUT) {
      tensor->refCount = 1;
      tensor->nodeType = NodeType_ValueNode;
    } else {
      tensor->nodeType = NodeType_Parameter;
    }
    tensor->name = name;
    tensors.push_back(tensor);

    if (Category == GRAPH_INPUT) {
      graphInputs.push_back(index);
    }

    if (Category == GRAPH_INPUT || Category == OP_OUTPUT || Category == TF_CONST) {
      UpdateTensorIndex(name, index);
    }
    return index;
  }

  // find the name index
  int FindTensor(const std::string &name) {
    auto iter = tensorIndex.find(name);
    if (iter != tensorIndex.end()) {
      return iter->second;
    }
    return -1;
  }

  void UpdateTensorIndex(const std::string &name, int idx) {
    auto iter = tensorIndex.find(name);
    if (iter != tensorIndex.end()) {
      tensorIndex[name] = idx;
    } else {
      tensorIndex.insert(make_pair(name, idx));
    }
  }

  // return allTensors
  const std::vector<TensorT *> &GetCachedTensor() const { return tensors; }

  const std::vector<int> &GetGraphInputs() const { return graphInputs; }

 private:
  std::vector<TensorT *> tensors;
  std::unordered_map<std::string, int> tensorIndex;
  std::vector<int> graphInputs;
  int index = -1;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_COMMON_TENSOR_UTIL_H
