/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/weight_quantizer.h"
#include <list>
#include <string>
#include "src/common/common.h"
#include "ir/dtype/type_id.h"

using std::string;
using std::vector;

namespace mindspore {
namespace lite {
namespace quant {

WeightQuantizer::WeightQuantizer(FuncGraphPtr graph, const string &weightSize,
                                 const std::string &convWeightChannelThreshold, const std::string &bitNum)
    : Quantizer(graph) {
  auto quantSize = static_cast<size_t>(std::stoull(weightSize));
  this->bitNum = static_cast<size_t>(std::stoull(bitNum));
  auto convQuantWeightChannelThreshold = static_cast<size_t>(std::stoull(convWeightChannelThreshold));
  // TODO(...): update stractory
  mStrategy.reset(new QuantStrategy(quantSize, convQuantWeightChannelThreshold));
}

// uint32_t GetConvChannel(TensorDefT *weight) {
//   uint32_t channel = 0;
//   const vector<int> dims = weight->dims;

//   switch (weight->format) {
//     case Format_NCHW:
//     case Format_KCHW:
//     case Format_NC4HW4:
//       channel = static_cast<uint32_t>(dims[NCHW_N]);
//       break;
//     case Format_NHWC:
//     case Format_HWKC:
//       channel = static_cast<uint32_t>(dims[NHWC_N]);
//       break;
//     case Format_HWCK:
//       channel = static_cast<uint32_t>(dims[HWCK_K]);
//       break;
//     case Format_CKHW:
//       channel = static_cast<uint32_t>(dims[CKHW_K]);
//       break;
//     default:
//       MS_LOGE("Unsupported format: %d", weight->format);
//       return 0;
//   }
//   return channel;
// }

STATUS WeightQuantizer::DoConvQuantize(const std::list<CNodePtr> &nodes) {
    for (auto &cnode : nodes) {
        if (!mStrategy->CanConvOpQuantized(cnode)) {
            continue;
        }

        auto inputNode = cnode->input(2);
        if (!inputNode->isa<Parameter>()) {
            return RET_ERROR;
        }

        auto paramNode = inputNode->cast<ParameterPtr>();
        if (!paramNode->has_default()) {
            return RET_ERROR;
        }

        ParamValueLitePtr paramValue = std::static_pointer_cast<ParamValueLite>(paramNode->default_param());
        auto status = QuantFilter(paramValue, QuantType_WeightQuant, 127, -128, bitNum);
        if (status != RET_OK) {
            MS_LOG(ERROR) << "QuantFilter failed : " << status;
            return status;
        }
    }

    return RET_OK;
}

STATUS WeightQuantizer::DoMulQuantize(const std::list<CNodePtr> &nodes) {
    for (auto &node : nodes) {
        if (!mStrategy->CanMulOpQuantized(node)) {
            continue;
        }

        ParamValueLitePtr paramValue = nullptr;
        for (size_t i = 1; i < node->size(); i++) {
            auto inputNode = node->input(i);
            if (inputNode->isa<Parameter>() == true) {
                auto paramNode = inputNode->cast<ParameterPtr>();
                if ((paramNode != nullptr) && (paramNode->has_default() == true)) {
                    paramValue = std::static_pointer_cast<ParamValueLite>(paramNode->default_param());
                    if ((paramValue == nullptr) || (paramValue->tensor_size() == 0)
                        || (paramValue->tensor_shape().size() != 4)
                        || (paramValue->tensor_addr() == nullptr)
                        || (paramValue->tensor_type() != mindspore::kNumberTypeFloat32)) {
                            paramValue = nullptr;
                            continue;
                    } else {
                        break;
                    }
                }
            }
        }
        if (paramValue == nullptr) {
            MS_LOG(ERROR) << "No valid input param node !";
            continue;
        }
        auto status = QuantFilter(paramValue, QuantType_WeightQuant, 127, -128, bitNum);
        if (status != RET_OK) {
            MS_LOG(ERROR) << "QunatFilter failed" << status;
            return RET_ERROR;
        }
    }

    return RET_OK;
}

STATUS WeightQuantizer::DoQuantize(FuncGraphPtr funcGraph) {
  auto ret = RET_OK;
  auto cnodes = funcGraph->GetOrderedCnodes();
  ret = DoConvQuantize(cnodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoConvQuantize failed :" << ret;
    return ret;
  }
  ret = DoMulQuantize(cnodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoMulQuantize failed :" << ret;
    return ret;
  }
  return ret;
}
}  // namespace quant
}  // namespace lite
}  // namespace mindspore

