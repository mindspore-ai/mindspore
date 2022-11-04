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

#include <string>
#include <memory>
#include <algorithm>
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "checker/op_checker.h"

namespace mindspore {
namespace dpico {
OpCheckerRegistry::~OpCheckerRegistry() {
  for (auto ite : checkers) {
    if (ite.second != nullptr) {
      delete ite.second;
      ite.second = nullptr;
    }
  }
}

OpCheckerRegistry *OpCheckerRegistry::GetInstance() {
  static OpCheckerRegistry instance;
  return &instance;
}

OpChecker *OpCheckerRegistry::GetOpChecker(const std::string &type) {
  auto it = checkers.find(type);
  if (it != checkers.end()) {
    return it->second;
  }
  return nullptr;
}

STATUS GetWidth(const std::vector<int64_t> &shape, mindspore::Format format, int64_t *width) {
  if (width == nullptr) {
    MS_LOG(ERROR) << "width is nullptr.";
    return RET_ERROR;
  }
  if (shape.size() == kDims4) {
    if (format == mindspore::Format::NCHW) {
      *width = shape.at(kInputIndex3);
    } else if (format == mindspore::Format::NHWC) {
      *width = shape.at(kInputIndex2);
    } else {
      MS_LOG(ERROR) << "format should be NCHW or NHWC";
      return RET_ERROR;
    }
  } else {
    *width = shape.back();
  }
  return RET_OK;
}

STATUS GetTensorChannel(const std::vector<int64_t> &shape, mindspore::Format format, int64_t *channel) {
  if (channel == nullptr) {
    MS_LOG(ERROR) << "channel is nullptr.";
    return RET_ERROR;
  }
  if (shape.size() != kDims4) {
    MS_LOG(ERROR) << "shape size should be 4, but is " << shape.size();
    return RET_ERROR;
  } else {
    if (format == mindspore::Format::NCHW) {
      *channel = shape.at(kInputIndex1);
    } else if (format == mindspore::Format::NHWC) {
      *channel = shape.at(kInputIndex3);
    } else {
      MS_LOG(ERROR) << "format should be NCHW or NHWC";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS GetVectorChannel(const std::vector<int64_t> &shape, int64_t *channel) {
  if (channel == nullptr) {
    MS_LOG(ERROR) << "channel is nullptr.";
    return RET_ERROR;
  }
  if (shape.size() != kDims2) {
    MS_LOG(ERROR) << "shape size should be 2, but is " << shape.size();
    return RET_ERROR;
  }
  *channel = shape.back();
  return RET_OK;
}

bool HasOfflineData(const api::AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is nullptr.";
    return false;
  }
  auto param = node->cast<api::ParameterPtr>();

  return param != nullptr && param->has_default();
}

bool CheckInputW(const api::CNodePtr &op, size_t index, mindspore::Format format, int limit_w) {
  if (index >= op->inputs().size()) {
    MS_LOG(ERROR) << "index:" << index << " is greater than " << op->fullname_with_scope()
                  << " inputs size:" << op->inputs().size();
    return false;
  }
  std::vector<int64_t> input_shape;
  if (GetInputShapeFromCNode(op, index, &input_shape) == RET_OK && !input_shape.empty()) {
    int64_t input_w;
    if (GetWidth(input_shape, format, &input_w) != RET_OK) {
      MS_LOG(ERROR) << "get input_w failed " << op->fullname_with_scope();
      return false;
    }
    if (input_shape.size() == kDims4 && input_w > limit_w) {
      MS_LOG(INFO) << op->fullname_with_scope() << "'s input_w:" << input_w << " exceed the maximum limit " << limit_w;
      return false;
    }
  }
  return true;
}
}  // namespace dpico
}  // namespace mindspore
