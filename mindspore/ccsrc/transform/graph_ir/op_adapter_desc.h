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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_DESC_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_DESC_H_

#include <memory>
#include "transform/graph_ir/op_adapter.h"

namespace mindspore {
namespace transform {
class OpAdapterDesc {
 public:
  OpAdapterDesc() : train_(nullptr), infer_(nullptr) {}

  OpAdapterDesc(const OpAdapterPtr &train, const OpAdapterPtr &infer) : train_(train), infer_(infer) {}

  explicit OpAdapterDesc(const OpAdapterPtr &common) : train_(common), infer_(common) {}

  OpAdapterDesc(const OpAdapterDesc &desc) {
    this->train_ = desc.train_;
    this->infer_ = desc.infer_;
  }

  OpAdapterDesc(OpAdapterDesc &&desc) {
    this->train_ = desc.train_;
    this->infer_ = desc.infer_;
    desc.train_ = nullptr;
    desc.infer_ = nullptr;
  }

  ~OpAdapterDesc() = default;

  OpAdapterPtr Get(bool train) const { return train ? train_ : infer_; }

  OpAdapterDesc &operator=(const OpAdapterDesc &desc) {
    if (this != &desc) {
      this->train_ = desc.train_;
      this->infer_ = desc.infer_;
    }
    return *this;
  }

  OpAdapterDesc &operator=(OpAdapterDesc &&desc) {
    if (this != &desc) {
      this->train_ = desc.train_;
      this->infer_ = desc.infer_;
      desc.train_ = nullptr;
      desc.infer_ = nullptr;
    }
    return *this;
  }

 private:
  OpAdapterPtr train_;
  OpAdapterPtr infer_;
};

using OpAdapterDescPtr = std::shared_ptr<OpAdapterDesc>;
}  // namespace transform
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_DESC_H_
