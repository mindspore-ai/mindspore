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

#include "parser/caffe/caffe_binary_math_parser.h"
#include <memory>
#include <vector>
#include "ops/custom.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/maximum.h"
#include "ops/minimum.h"
#include "ops/squared_difference.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeBinaryMathParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  const caffe::BinaryMathParameter &binary_math_param = proto.binary_math_param();
  BaseOperatorPtr prim;
  if (binary_math_param.has_operation()) {
    auto operation = binary_math_param.operation();
    switch (operation) {
      case caffe::BinaryMathParameter_BinaryMathOp_ADD:
        prim = std::make_shared<ops::AddFusion>();
        break;
      case caffe::BinaryMathParameter_BinaryMathOp_SUB:
        prim = std::make_shared<ops::SubFusion>();
        break;
      case caffe::BinaryMathParameter_BinaryMathOp_MUL:
        prim = std::make_shared<ops::MulFusion>();
        break;
      case caffe::BinaryMathParameter_BinaryMathOp_DIV:
        prim = std::make_shared<ops::DivFusion>();
        break;
      case caffe::BinaryMathParameter_BinaryMathOp_MAX:
        prim = std::make_shared<ops::Maximum>();
        break;
      case caffe::BinaryMathParameter_BinaryMathOp_MIN:
        prim = std::make_shared<ops::Minimum>();
        break;
      case caffe::BinaryMathParameter_BinaryMathOp_SQUARE_DIFF:
        prim = std::make_shared<ops::SquaredDifference>();
        break;
      case caffe::BinaryMathParameter_BinaryMathOp_X_DIV_Y: {
        prim = std::make_shared<ops::Custom>();
        if (prim == nullptr) {
          MS_LOG(ERROR) << "prim is nullptr. " << proto.name();
          return nullptr;
        }
        static_cast<ops::Custom *>(prim.get())->set_type("X_DIV_Y");
        break;
      }
      case caffe::BinaryMathParameter_BinaryMathOp_X_LOG_Y: {
        prim = std::make_shared<ops::Custom>();
        if (prim == nullptr) {
          MS_LOG(ERROR) << "prim is nullptr. " << proto.name();
          return nullptr;
        }
        static_cast<ops::Custom *>(prim.get())->set_type("X_LOG_Y");
        break;
      }
      default:
        MS_LOG(ERROR) << "unsupported binary math op type. " << operation;
        return nullptr;
    }
  } else {
    prim = std::make_shared<ops::AddFusion>();
  }
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr. " << proto.name();
    return nullptr;
  }
  return prim;
}

CaffeNodeRegistrar g_caffeBinaryMathParser("BinaryMath", new CaffeBinaryMathParser());
}  // namespace lite
}  // namespace mindspore
