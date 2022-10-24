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

#include "tools/converter/quantizer/cluster_quantization.h"
#include <set>
#include <vector>
#include <algorithm>
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "include/errorcode.h"
#include "tools/converter/quantizer/fse_encoder.h"

namespace mindspore::lite::quant {
namespace {
constexpr int kPowExponent = 2;
constexpr int kMinSize = 2;
constexpr int kAverage = 2;
}  // namespace
std::vector<float> ClusterQuantization::KMeansPlusPlusInit(const float *data, size_t elem_count, size_t k) {
  std::vector<float> data_vector(data, data + elem_count);
  std::sort(data_vector.begin(), data_vector.end());
  data_vector.erase(std::unique(data_vector.begin(), data_vector.end()), data_vector.end());

  if (data_vector.size() < k) {
    return {};
  }

  std::vector<float> clusters{};
  auto min_max = GetMinMaxValue(data_vector.data(), data_vector.size());
  auto mean = (min_max.second - min_max.first) / 2;
  clusters.push_back(mean);

  for (size_t i = 1; i < k; i++) {
    auto max_distance = -1;
    auto max_index = 0;
    for (size_t j = 0; j < data_vector.size(); j++) {
      float distance = 0;
      for (auto cluster : clusters) {
        auto cur_distance = std::pow(data_vector[j] - cluster, kPowExponent);
        distance += cur_distance;
      }
      if (distance > max_distance) {
        max_distance = distance;
        max_index = j;
      }
    }
    clusters.push_back(data_vector[max_index]);
    data_vector.erase(data_vector.begin() + max_index);
  }
  std::sort(clusters.begin(), clusters.end());
  return clusters;
}

std::vector<float> ClusterQuantization::LinearInit(const float *data, size_t elem_count, size_t k) {
  MS_ASSERT(data != nullptr);
  std::set<float> set_unique{};
  for (size_t i = 0; i < elem_count; i++) {
    set_unique.emplace(data[i]);
  }
  std::vector<float> data_unique;
  data_unique.assign(set_unique.begin(), set_unique.end());
  std::vector<float> clusters{};
  if (set_unique.size() < k) {
    return clusters;
  }
  // Linear division into K classes.
  MS_ASSERT(k >= kMinSize);
  float stride = static_cast<float>(data_unique.size()) / (k - 1);
  std::sort(data_unique.begin(), data_unique.end());
  for (size_t i = 0; i < k; i++) {
    size_t index = std::floor(i * stride);
    if (i * stride - index > 0) {
      // Average cluster centroid.
      auto cluster_centroid = (data_unique[index] + data_unique[index + 1]) / kAverage;
      clusters.emplace_back(cluster_centroid);
    } else {
      clusters.emplace_back(data_unique[index]);
    }
  }
  return clusters;
}

void ClusterQuantization::SelectClusterCentroid(const float *data, size_t elem_count,
                                                const std::vector<float> &clusters, std::vector<int8_t> *clusters_index,
                                                std::vector<std::vector<float>> *clusters_data) {
  for (size_t i = 0; i < elem_count; i++) {
    size_t index = 0;
    double euclidean_distance = std::pow(data[i] - clusters.at(0), kPowExponent);
    double min_distance = euclidean_distance;
    for (size_t j = 1; j < clusters.size(); j++) {
      if (pow(data[i] - clusters.at(j), kPowExponent) < min_distance) {
        min_distance = std::pow(data[i] - clusters.at(j), kPowExponent);
        index = j;
      }
    }
    clusters_index->at(i) = index + INT8_MIN;
    clusters_data->at(index).push_back(data[i]);
  }
}

void UpdateClusterCentroid(const std::vector<std::vector<float>> &clusters_data, std::vector<float> *clusters) {
  for (size_t j = 0; j < clusters->size(); j++) {
    if (!clusters_data[j].empty()) {
      clusters->at(j) =
        std::accumulate(clusters_data[j].begin(), clusters_data[j].end(), 0.0) / clusters_data[j].size();
    }
  }
}

int ClusterQuantization::KMeans(const float *data, size_t elem_count, size_t k, size_t max_epochs, double tol_error,
                                std::vector<int8_t> *clusters, std::vector<float> *cluster_centroid) {
  CHECK_NULL_RETURN(data);
  CHECK_LESS_RETURN(elem_count, 1);
  CHECK_LESS_RETURN(k, kMinSize);
  CHECK_LESS_RETURN(max_epochs, 1);
  std::vector<float> centroid = LinearInit(data, elem_count, k);
  if (centroid.size() < kMinSize) {
    MS_LOG(INFO) << "centroid size is " << centroid.size() << ", so KMeans function is not executed. ";
    return RET_NO_CHANGE;
  }
  auto min_error = DBL_MAX;

  for (size_t epoch = 0; epoch < max_epochs; epoch++) {
    std::vector<int8_t> cur_clusters_index(elem_count);
    std::vector<std::vector<float>> clusters_data(centroid.size());

    // Calculate the distance and select the cluster centroid.
    SelectClusterCentroid(data, elem_count, centroid, &cur_clusters_index, &clusters_data);
    // Update cluster centroid.
    UpdateClusterCentroid(clusters_data, &centroid);

    // Calculate Error(MSE)
    double cur_error = 0;
    for (size_t j = 0; j < elem_count; j++) {
      auto real_index = cur_clusters_index.at(j) - INT8_MIN;
      cur_error += std::pow(data[j] - centroid.at(real_index), kPowExponent);
    }
    cur_error = std::sqrt(cur_error / elem_count);
    MS_LOG(INFO) << "current epoch is " << epoch << ", error is " << cur_error
                 << " , centroid size is:" << centroid.size();

    if (cur_error <= tol_error) {
      MS_LOG(INFO) << "current min_error is " << cur_error << " <= tolerance min_error " << tol_error;
      break;
    }

    if (cur_error == min_error) {
      MS_LOG(INFO) << "The cluster center has not changed, stop the iteration.";
      break;
    }

    if (cur_error < min_error) {
      clusters->assign(cur_clusters_index.begin(), cur_clusters_index.end());
      cluster_centroid->assign(centroid.begin(), centroid.end());
      min_error = cur_error;
    }
  }
  return RET_OK;
}

int ClusterQuantization::KMeansQuantization(const CNodePtr &cnode, const std::vector<int> &weight_indices) {
  for (auto idx : weight_indices) {
    auto input = cnode->input(idx);
    ParameterPtr parameter;
    tensor::TensorPtr tensor_info;
    GetLiteParameter(input, &parameter, &tensor_info);
    if (parameter == nullptr || tensor_info == nullptr || tensor_info->data_type() != TypeId::kNumberTypeFloat32 ||
        tensor_info->compression_type() != mindspore::kNoCompression) {
      MS_LOG(INFO) << "This op " << cnode->fullname_with_scope() << " dont need quant weight";
      continue;
    }
    if (tensor_info->shape_c().size() == 1) {
      MS_LOG(INFO) << "This op " << parameter->fullname_with_scope() << " is bias";
      continue;
    }
    auto data = static_cast<float *>(tensor_info->data().data());
    std::vector<float> cluster_centroid;
    std::vector<int8_t> clusters;
    auto ret = KMeans(data, tensor_info->DataSize(), k_, max_epochs_, tol_error_, &clusters, &cluster_centroid);
    if (ret == RET_NO_CHANGE) {
      continue;
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << input->fullname_with_scope() << " execute KMeans failed.";
      return ret;
    }

    UpdateTensorDataAndSize(parameter, tensor_info, clusters.data(), clusters.size(), kNumberTypeInt8);
    // Optimize with Share Weight
    auto quant_param_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(quant_param_holder);
    quant_param_holder->SetQuantClusters(idx - kPrimOffset, cluster_centroid);
    quant_param_holder->set_quant_type(schema::QuantType_QUANT_WEIGHT);

    FSEEncoder fse_encoder;
    ret = fse_encoder.Compress(parameter, {}, mindspore::kFSEInt);
    if (ret == RET_OK) {
      MS_LOG(INFO) << "Execute FSE compression success.";
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
