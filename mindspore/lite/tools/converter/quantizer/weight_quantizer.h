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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H_

#include <future>
#include <memory>
#include <map>
#include <list>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <algorithm>
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quant_strategy.h"
#include "tools/converter/preprocess/preprocess_param.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "abstract/dshape.h"
#include "src/litert/lite_session.h"
#include "src/common/quant_utils.h"

namespace mindspore::lite::quant {
class WeightQuantizer : public Quantizer {
 public:
  WeightQuantizer() {
    quant_min_ = QuantMin(bit_num_, false, false);
    quant_max_ = QuantMax(bit_num_, false);
    symmetric_quant_min_ = QuantMin(bit_num_, false, true);
    symmetric_quant_max_ = QuantMax(bit_num_, false);
    // parse type_id_
    MS_ASSERT(bit_num_ > 0 && bit_num_ <= k16Bit);
    if (bit_num_ > 0 && bit_num_ <= k8Bit) {
      type_id_ = kNumberTypeInt8;
    } else if (bit_num_ <= k16Bit) {
      type_id_ = kNumberTypeInt16;
    }
  }

  explicit WeightQuantizer(const std::shared_ptr<ConverterPara> &param, double init_scale = 0) : Quantizer(param) {
    bit_num_ = param_->commonQuantParam.bit_num;
    enable_encode_ = param_->commonQuantParam.enable_encode;
    if (bit_num_ == 0) {
      type_id_ = kNumberTypeInt16;
      is_mixed_bit_ = true;
      mixed_bit_init_scale_ = param_->mixedBitWeightQuantParam.init_scale;
      is_auto_tune_ = param_->mixedBitWeightQuantParam.auto_tune;
    }
    // parse param for fixed bit quant.
    if (!is_mixed_bit_) {
      quant_min_ = QuantMin(bit_num_, false, false);
      quant_max_ = QuantMax(bit_num_, false);
      symmetric_quant_min_ = QuantMin(bit_num_, false, true);
      symmetric_quant_max_ = QuantMax(bit_num_, false);
      // parse type_id_
      MS_ASSERT(bit_num_ > 0 && bit_num_ <= k16Bit);
      if (bit_num_ > 0 && bit_num_ <= k8Bit) {
        type_id_ = kNumberTypeInt8;
      } else if (bit_num_ <= k16Bit) {
        type_id_ = kNumberTypeInt16;
      }
    }
    quant_strategy_ = std::make_unique<QuantStrategy>(param_->commonQuantParam.min_quant_weight_size,
                                                      param_->commonQuantParam.min_quant_weight_channel,
                                                      param_->commonQuantParam.skip_quant_node);
    if (init_scale > 0) {
      mixed_bit_init_scale_ = init_scale;
    }
    if (!param_->commonQuantParam.skip_quant_node.empty()) {
      std::copy(param_->commonQuantParam.skip_quant_node.cbegin(), param_->commonQuantParam.skip_quant_node.cend(),
                std::inserter(skip_quant_node_, skip_quant_node_.begin()));
    }
    quant_type_ = param_->commonQuantParam.quant_type;
    dequant_strategy_ = param_->weightQuantParam.dequant_strategy;
    max_segments_ = param_->weightQuantParam.max_segments;
    if (param_->weightQuantParam.dequant_strategy == ON_THE_FLY) {
      weight_quant_type_ = WeightQuantType::FIXED_BIT_PER_LAYER;
    }
  }

  ~WeightQuantizer() override = default;

  int DoQuantize(FuncGraphPtr func_graph) override;

  int WeightQuant(const FuncGraphPtr &func_graph, const std::set<PrimitivePtr> &support_weight_quant_types,
                  const std::set<PrimitivePtr> &per_layer_types, const std::set<PrimitivePtr> &symmetric_types,
                  bool compression = true);

  std::set<tensor::TensorPtr> GetWeightQuantizedTensors() { return this->weight_quantized_tensors_; }

 private:
  int WeightQuantPerCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                          const std::set<PrimitivePtr> &support_weight_quant_types,
                          const std::set<PrimitivePtr> &per_layer_types, const std::set<PrimitivePtr> &symmetric_types,
                          bool compression = true);
  int LinearQuant(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::set<PrimitivePtr> &per_layer_types,
                  const std::set<PrimitivePtr> &symmetric_types, const std::vector<int> &weight_indices,
                  bool compression = true);
  int MarkGraphWeightQuantType(const FuncGraphPtr &func_graph);
  int MarkCNodeWeightQuantType(const CNodePtr &cnode);
  int DoCompression(const CNodePtr &cnode, const ParameterPtr &parameter, int idx);
  int DoMixBitQuant(const CNodePtr &cnode, const ParameterPtr &parameter, int idx, const tensor::TensorPtr &tensor_info,
                    int preferred_dim, WeightQuantType weight_quant_type, bool symmetric = true);
  int InsertDequantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const ParameterPtr &parameter, int idx,
                        const tensor::TensorPtr &tensor_info);

 private:
  bool is_auto_tune_{false};
  bool is_mixed_bit_{false};
  bool linear_quant_{true};
  size_t bit_num_{8};
  double mixed_bit_init_scale_ = 0.02;
  int quant_min_{-128};
  int quant_max_{127};
  int symmetric_quant_min_{-127};
  int symmetric_quant_max_{127};
  TypeId type_id_{kNumberTypeInt8};
  std::set<std::string> skip_quant_node_;
  std::unique_ptr<QuantStrategy> quant_strategy_;
  quant::QuantType quant_type_{quant::QUANT_WEIGHT};
  bool enable_encode_{true};
  WeightQuantType weight_quant_type_ = WeightQuantType::FIXED_BIT_PER_CHANNEL;
  DequantStrategy dequant_strategy_ = DEFAULT;
  int max_segments_{1};
  // Support for mark shared weight node.
  std::set<tensor::TensorPtr> weight_quantized_tensors_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H_
