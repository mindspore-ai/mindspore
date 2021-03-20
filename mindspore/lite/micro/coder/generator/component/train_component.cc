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

#include "coder/generator/component/train_component.h"
#include <string>
#include "coder/utils/type_cast.h"

namespace mindspore::lite::micro {

void CodeTrainParams(std::ofstream &ofs) {
  ofs << "struct TrainParameter {\n"
         "  float beta1_;\n"
         "  float beta2_;\n"
         "  float epsilon_;\n"
         "};\n"
         "\n"
         "enum EarlyStopType {\n"
         "  Diff = 0,\n"
         "  WeigthDiff = 1,\n"
         "  Abs = 2,\n"
         "};\n"
         "\n"
         "struct EarlyStop {\n"
         "  enum EarlyStopType type;\n"
         "  float tolerate;\n"
         "};\n\n";
}

void CodeFeaturesState(std::ofstream &ofs) {
  ofs << "/**\n"
         " *\n"
         " * @param size, return the number of features\n"
         " * @return, the address of features\n"
         " */\n"
      << "FeatureParam *GetFeatures(int *size);\n\n";
  ofs << "/**\n"
         " *\n"
         " * @param features, the address of features\n"
         " * @param size, the number of features\n"
         " * @return, status\n"
         " */\n"
      << "int UpdateFeatures(FeatureParam *features, int size);\n\n";
}

void CodeFeaturesImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  size_t features_num = 0;
  ofs << "static FeatureParam feature_params[] = {\n";
  for (const auto &item : ctx->saved_weights()) {
    std::string addr = item.first;
    Tensor *tensor = item.second;
    if (tensor->tensor_name().empty()) {
      MS_LOG(ERROR) << "exist empty feature";
      continue;
    }
    ofs << "\t{\"" << tensor->tensor_name() << "\", " << addr << ", " << tensor->ElementsNum() << ", "
        << EnumMicroTensorDataType(tensor->data_type()) << "}, \n";
    features_num++;
  }
  ofs << "};\n";

  ofs << "FeatureParam *GetFeatures(int *size) {\n"
      << "  *size = " << features_num << ";\n"
      << "  return feature_params;\n"
         "}\n\n";

  ofs << "int "
      << "UpdateFeatures(FeatureParam *features, int size) {\n"
      << "  for (int i = 0; i < size; ++i) {\n"
         "    FeatureParam *src = features + i;\n"
         "    FeatureParam dst;\n"
         "    // find the dst feature\n"
         "    bool is_find = false;\n"
      << "    for (int j = 0; j < " << features_num << "; ++j) {\n"
      << "      if (strcmp(src->name, feature_params[j].name) == 0) {\n"
         "        dst = feature_params[j];\n"
         "        is_find = true;\n"
         "        break;\n"
         "      }\n"
         "    }\n"
         "    if (!is_find) {\n"
         "      MICRO_ERROR(\"invalid feature param: %s\", src->name);\n"
         "      return RET_ERROR;\n"
         "    }\n"
         "    if (src->elenums != dst.elenums) {\n"
         "      MICRO_ERROR(\"feature %s elenums is mismatch, src: %lu, dst: %lu\", src->name, src->elenums, "
         "dst.elenums);\n"
         "      return RET_ERROR;\n"
         "    }\n"
         "    memcpy(dst.data, src->data, src->elenums * sizeof(float));\n"
         "  }\n"
         "  MICRO_INFO(\"update features map success\");\n"
         "  return RET_OK;\n"
         "}\n\n";
}

void CodeTrainState(std::ofstream &ofs) {
  ofs
    << "/**\n"
       " * Train Function\n"
       " * @param epoch, the train epoch\n"
       " * @param iterations, which is equal to batch_num, the number of iterations of each epoch\n"
       " * @param use_train_param, default parameters already exists, such as the momentum, user can update these\n"
       " * parameters to improve the accuracy\n"
       " * @param parameter, the TrainParameter contains epsilon/beta1/beta2\n"
       " * @return status\n"
       " */\n"
    << "int Train(const int epoch, const int iterations, bool use_train_param, const struct TrainParameter *parameter, "
       "const struct EarlyStop *early_stop);\n\n";
}

void CodeTrainImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  size_t inputs_num = inputs.size();
  auto inputs_tostring = [&]() {
    std::string result;
    result += "{";
    for (size_t i = 0; i < inputs.size(); ++i) {
      result += ctx->input_name() + std::to_string(i) + ", ";
    }
    result += "}";
    return result;
  };
  auto wrap = [](size_t i) { return "[" + std::to_string(i) + "]"; };
  auto offset_inputs = [&]() {
    std::string src = "origin_inputs";
    std::string dst = "input_ptr";
    std::string result;
    for (size_t i = 0; i < inputs.size(); ++i) {
      result += dst + wrap(i) += " = " + src + wrap(i) + " + j * " + std::to_string(inputs[i]->Size()) + ";\n";
    }
    return result;
  };
  auto varify_inputs = [&]() {
    std::string result;
    for (size_t i = 0; i < inputs.size(); ++i) {
      result += "origin_input" + wrap(i) + " + iterations * " + std::to_string(inputs[i]->Size()) + " == NULL";
      i < inputs.size() - 1 ? result += " || " : result += "";
    }
    return result;
  };
  ofs << "int Train(const int epoch, const int iterations, bool use_train_param, const struct TrainParameter "
         "*parameter, const struct EarlyStop *early_stop) {\n"
         "  if (iterations <= 0 || epoch <= 0) {\n"
         "    MICRO_ERROR(\"error iterations or epoch!, epoch:%d, iterations:%d\", epoch, iterations);\n"
         "    return RET_ERROR;\n"
         "  }\n"
         "  MICRO_INFO(\"train epoch: %d, batch_num: %d\", epoch, iterations);\n"
      << "  const void *origin_input[] = " << inputs_tostring() << ";\n";
  ofs << "  if (" << varify_inputs() << ") {\n"
      << "    MICRO_ERROR(\"input data is invalid, epoch: %d, iterations: %d\", epoch, iterations);\n"
         "    return RET_ERROR;\n"
         "  }\n";
  ofs << "  for (int i = 0; i < epoch; ++i) {\n"
      << "    const void *input_ptr[" << inputs_num << "];\n"
      << "    float loss = 0;\n"
      << "    for (int j = 0; j < iterations; ++j) {\n"
      << "      " << offset_inputs() << "\n"
      << "      "
      << "_SetInputs(input_ptr, " << inputs_num << ");\n"
      << "      "
      << "_Inference();\n"
      << "      loss = "
      << "ComputeLossAndGradient();\n"
      << "    }\n"
         "  }\n"
         "  return RET_OK;\n"
         "};\n\n";
}
}  // namespace mindspore::lite::micro
