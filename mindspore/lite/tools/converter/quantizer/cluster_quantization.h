/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLUSTER_QUANTIZATION_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLUSTER_QUANTIZATION_H

#include <vector>
#include "ir/func_graph.h"
#include "tools/converter/quantizer/quantize_util.h"
namespace mindspore::lite::quant {
class ClusterQuantization {
 public:
  ClusterQuantization() = default;

  ~ClusterQuantization() = default;

  int KMeansQuantization(const CNodePtr &cnode, const std::vector<int> &weight_indices);

 private:
  int KMeans(const float *data, size_t elem_count, size_t k, size_t max_epochs, double tol_error,
             std::vector<int8_t> *clusters, std::vector<float> *cluster_centroid);
  std::vector<float> LinearInit(const float *data, size_t elem_count, size_t k);
  std::vector<float> KMeansPlusPlusInit(const float *data, size_t elem_count, size_t k);
  void SelectClusterCentroid(const float *data, size_t elem_count, const std::vector<float> &clusters,
                             std::vector<int8_t> *clusters_index, std::vector<std::vector<float>> *clusters_data);

  size_t k_ = 256;
  size_t max_epochs_ = 64;
  double tol_error_ = 0.000;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_CLUSTER_QUANTIZATION_H
