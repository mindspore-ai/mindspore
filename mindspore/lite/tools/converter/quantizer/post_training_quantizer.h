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

#ifndef POSTRAINING_QUANTIZER_H
#define POSTRAINING_QUANTIZER_H

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cfloat>
#include <map>
#include "src/lite_session.h"
#include "tools/converter/quantizer/quantizer.h"
#include "src/ir/primitive_t_value.h"
#include "tools/converter/converter.h"
#include "include/ms_tensor.h"

namespace mindspore {
namespace lite {
namespace quant {
class Calibrator;

struct MaxMin {
 public:
  float min;
  float max;
};

enum ImageFormat {
  RGB = 0,
  GRAY = 1,
  BGR = 2,
};

const char kMethodMaxMin[] = "MAX_MIN";
const char kMethodKL[] = "KL";

struct ConfigParam {
  // ImageFormat imageFormat;
  std::string image_path;
  uint32_t batch_count{100};
  std::string method_x{kMethodKL};
  uint32_t thread_num;
};

class PostTrainingQuantizer : public Quantizer {
 public:
  PostTrainingQuantizer(FuncGraphPtr graph, std::string path, int bit_num, TypeId target_type = kNumberTypeInt8,
                        bool per_channel = false);

  STATUS DoQuantize(FuncGraphPtr funcGraph) override;

  size_t bit_num;
  int quant_max{127};
  int quant_min{-128};

 private:
  bool per_channel_;

  TypeId target_type_{kNumberTypeInt8};

  std::unique_ptr<Calibrator> calibrator_;

  mindspore::lite::LiteSession *session_;

  STATUS PreProcess();

  STATUS CheckTensorVec(const std::string &nodeName, const std::vector<mindspore::tensor::MSTensor *> &tensorVec) const;

  STATUS DoInference();

  STATUS UpdateDivergInverval();

  STATUS CollectDataFrequency();

  STATUS ComputeThreshold();

  STATUS QuantNode();

  //    STATUS reformatConvWeight(GraphDefT *graph);

  STATUS DoQuantInput(double scale, int32_t zeropoint, struct MaxMin *max_min, std::shared_ptr<PrimitiveTValue>);
  STATUS DoQuantOutput(double scale, int32_t zeropoint, struct MaxMin *max_min, std::shared_ptr<PrimitiveTValue>);

  STATUS DoWeightQuant(AnfNodePtr node);

  STATUS DoBiasQuant(std::shared_ptr<PrimitiveTValue> input, AnfNodePtr weight, AnfNodePtr bias);
};

struct DivergInfo;

class Calibrator {
 public:
  explicit Calibrator(std::string path, size_t quant_size, int quant_max, int quant_msin);

  ~Calibrator() = default;

  STATUS ReadConfig();

  STATUS CollectImages();

  STATUS GenerateInputData(int index, mindspore::tensor::MSTensor *tensor) const;

  size_t GetBatchNum() const { return images_.size(); }

  uint32_t GetThreadNum() const { return config_param_.thread_num; }

  std::string GetMethodX() const { return config_param_.method_x; }

  STATUS AddQuantizedOp(CNodePtr node);

  STATUS RecordMaxValue(std::string opName, std::vector<float> data,
                        std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info);

  STATUS UpdateDivergInverval(std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info);

  STATUS UpdateDataFrequency(std::string op_name, std::vector<float> data, std::vector<int> shape,
                             std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info);
  void Dump();

  STATUS ComputeThreshold();

  std::unordered_map<CNodePtr, float> GetResult(
    std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info);

  std::unordered_map<CNodePtr, int32_t> GetZeropoint(
    std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info);

  std::map<CNodePtr, MaxMin> GetMinMax(std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *diverg_info);

  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *GetInputDivergInfo();

  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> *GetOutputDivergInfo();

 private:
  std::vector<std::string> images_;

  std::string config_path_;

  ConfigParam config_param_;

  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> input_diverg_info_;

  std::unordered_map<std::string, std::unique_ptr<DivergInfo>> output_diverg_info_;

  size_t bit_num_;
  int quant_max_;
  int quant_min_;

  void AddImage(std::string file);
};
}  // namespace quant
}  // namespace lite
}  // namespace mindspore
#endif  // POSTRAINING_QUANTIZER_H
