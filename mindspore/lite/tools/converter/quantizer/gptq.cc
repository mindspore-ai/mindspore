/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/gptq.h"
#include <memory>
#include <cmath>
#include "tools/converter/quantizer/eigen_util.h"
#include "tools/common/statistic_utils.h"
#include "src/common/quant_utils.h"

namespace mindspore::lite::quant {
int Gptq::Init() {
  // find_params
  if (FindQuantParams(weight_data_, weight_tensor_->shape(), weight_tensor_->ElementsNum(), prefer_dim_,
                      quant_params_) != RET_OK) {
    MS_LOG(ERROR) << "find quant params failed.";
    return RET_ERROR;
  }

  // initialize data
  CHECK_NULL_RETURN(hessian_data_);
  for (int32_t i = 0; i < hessian_length_; i++) {
    // dead = torch.diag(H) == 0
    if (*(hessian_data_ + i * hessian_length_ + i) == 0) {
      // H[dead, dead] = 1
      *(hessian_data_ + i * hessian_length_ + i) = 1;
      for (int32_t k = 0; k < rows_; k++) {
        // W[:, dead] = 0
        *(weight_data_ + k * deep_ + i) = 0;
      }
    }
  }
  return RET_OK;
}

int Gptq::FindQuantParams(const float *weight_data, std::vector<int> dims, int element_count, int prefer_dim,
                          std::vector<schema::QuantParamT> *quant_params) {
  CHECK_NULL_RETURN(weight_data);
  if (dims.size() != 2) {
    MS_LOG(ERROR) << "Invalid shape.";
    return RET_ERROR;
  }

  std::map<int, MinMax> perchannel_min_max;
  for (int i = 0; i < dims[prefer_dim]; i++) {
    perchannel_min_max.insert({i, {FLT_MAX, -FLT_MAX}});
  }
  if (prefer_dim == 0) {
    auto bucket_size = element_count / dims[prefer_dim];
    for (int i = 0; i < dims[prefer_dim]; ++i) {
      if ((weight_data + i * bucket_size) == nullptr) {
        return RET_ERROR;
      }
      auto mim_max = GetFloatMinMaxValue(weight_data + i * bucket_size, bucket_size);
      auto iter = perchannel_min_max.find(i);
      MS_CHECK_TRUE_RET(iter != perchannel_min_max.end(), RET_ERROR);
      iter->second.min = mim_max.first;
      iter->second.max = mim_max.second;
    }
  } else {
    for (int i = 0; i < element_count; ++i) {
      auto bucket_index = GetBucketIndex(dims, prefer_dim, i);
      auto iter = perchannel_min_max.find(bucket_index);
      MS_CHECK_TRUE_RET(iter != perchannel_min_max.end(), RET_ERROR);
      iter->second.min = std::min(iter->second.min, weight_data[i]);
      iter->second.max = std::max(iter->second.max, weight_data[i]);
    }
  }
  for (auto min_max_map : perchannel_min_max) {
    float min = min_max_map.second.min;
    float max = min_max_map.second.max;
    schema::QuantParamT quant_param;
    if (CalQuantizationParams(&quant_param, min, max, bit_num_, quant_min_, quant_max_, symmetric_, narrow_range_) !=
        RET_OK) {
      MS_LOG(ERROR) << "Cal quantization params failed.";
      return RET_ERROR;
    }
    quant_params->emplace_back(quant_param);
  }
  return RET_OK;
}

/**
 * damp = percdamp * torch.mean(torch.diag(H))
 * diag = torch.arange(self.columns, device=self.dev)
 * H[diag, diag] += damp
 * H = torch.linalg.cholesky(H)
 * H = torch.cholesky_inverse(H)
 * H = torch.linalg.cholesky(H, upper=True)
 * Hinv = H
 * */
int Gptq::CalculateHessianInv(float *hessian_data, float *hessian_inv_data, int hessian_length, float percdamp) {
  CHECK_NULL_RETURN(hessian_inv_data);
  float mean = 0.0;
  for (int32_t i = 0; i < hessian_length; i++) {
    for (int32_t j = 0; j < hessian_length; j++) {
      mean += *(hessian_data + i * hessian_length + j);
    }
  }
  mean = mean / (hessian_length * hessian_length);
  float damp = percdamp * mean;
  for (int32_t i = 0; i < hessian_length; i++) {
    *(hessian_data + i * hessian_length + i) += damp;
  }
  float *hessian_data_tmp = reinterpret_cast<float *>(malloc(hessian_length * hessian_length * sizeof(float)));
  MS_CHECK_TRUE_MSG(hessian_data_tmp != nullptr, false, "Malloc hessian data failed.");
  std::vector<int64_t> dims{hessian_length, hessian_length};
  if (quant::CalculateCholesky<float>(hessian_data, hessian_data_tmp, dims) != RET_OK) {
    MS_LOG(ERROR) << "Calculate Hessian matrix cholesky failed";
    return RET_ERROR;
  }
  if (quant::CalculateCholeskyInverse<float>(hessian_data_tmp, hessian_data, hessian_length) != RET_OK) {
    MS_LOG(ERROR) << "Calculate Hessian matrix cholesky inverse failed.";
    return RET_ERROR;
  }
  if (quant::CalculateCholesky<float>(hessian_data, hessian_inv_data, dims, true) != RET_OK) {
    MS_LOG(ERROR) << "Calculate Hessian matrix cholesky failed.";
    return RET_ERROR;
  }
  free(hessian_data_tmp);
  return RET_OK;
}

// Clone matrix, src[i1:i2, j1:j2] -> dest[r, c]
int Gptq::CloneMatrix(float *dest, int dst_rows, int dst_columns, const float *src, int src_rows, int src_columns,
                      int i1, int i2, int j1, int j2) {
  CHECK_NULL_RETURN(dest);
  CHECK_NULL_RETURN(src);
  MS_CHECK_TRUE_MSG(i1 >= 0 && i1 < src_rows, RET_ERROR, "Index i1 out-of-range.");
  MS_CHECK_TRUE_MSG(i2 >= 0 && i2 <= src_rows, RET_ERROR, "Index i2 out-of-range.");
  MS_CHECK_TRUE_MSG(j1 >= 0 && j1 < src_columns, RET_ERROR, "Index j1 out-of-range.");
  MS_CHECK_TRUE_MSG(j2 >= 0 && j2 <= src_columns, RET_ERROR, "Index j2 out-of-range.");
  int r = i2 - i1;
  int c = j2 - j1;
  MS_CHECK_TRUE_MSG(r >= 0 && r <= dst_rows, RET_ERROR, "Index (i2 - i1) out-of-range.");
  MS_CHECK_TRUE_MSG(c >= 0 && c <= dst_columns, RET_ERROR, "Index (j2 - j1) out-of-range.");
  for (int i = i1; i < i2; i++) {
    for (int j = j1; j < j2; j++) {
      *(dest + (i - i1) * c + (j - j1)) = *(src + i * src_columns + j);
    }
  }
  return RET_OK;
}

int Gptq::QuantizePerBlock(std::vector<float> *weight_data, std::vector<int> *quant_data, std::vector<float> *error,
                           std::vector<float> *loss, const std::vector<float> *hinv, int count) {
  CHECK_NULL_RETURN(weight_data);
  CHECK_NULL_RETURN(quant_data);
  CHECK_NULL_RETURN(error);
  CHECK_NULL_RETURN(loss);
  CHECK_NULL_RETURN(hinv);
  for (int i = 0; i < count; i++) {
    float d = hinv->at(i * count + i);
    // formula: Q1[:, i] = q
    for (int r = 0; r < rows_; r++) {
      float raw_data = weight_data->at(r * count + i);
      MS_CHECK_GT(static_cast<int>(quant_params_->size()), r, RET_ERROR);
      auto quant_param = quant_params_->at(r);
      auto data = QuantizeData<int8_t>(raw_data, &quant_param, quant_max_, quant_min_);
      quant_data->at(r * count + i) = data;
      float dequant_data = quant_param.scale * (data - quant_param.zeroPoint);
      // formula: err1 = (w - q) / d
      error->at(r * count + i) = (raw_data - dequant_data) / d;
      // formula: Losses1[:, i] = (w - q) ** 2 / d ** 2
      loss->at(r * count + i) = std::pow((raw_data - dequant_data), 2.0) / std::pow(d, 2.0);
    }
    // formula: W1[:,i:] -= Err1[:,i] * Hinv1[i, i:]
    for (int j = i; j < count; j++) {
      float hinv_data = hinv->at(i * count + j);
      for (int r = 0; r < rows_; r++) {
        weight_data->at(r * count + j) -= error->at(r * count + i) * hinv_data;
      }
    }
  }
  return RET_OK;
}

/**
 * 1 calculate inverse Hessian
 * 2 quantize per block
 * 3 calculate error
 * 4 update weights unquantized
 * */
int Gptq::DoQuantize() {
  if (Init() != RET_OK) {
    MS_LOG(ERROR) << "Initialize weight data failed.";
    return RET_ERROR;
  }
  size_t elements_num = static_cast<size_t>(weight_tensor_->ElementsNum());
  std::vector<float> loss(elements_num, 0);
  std::vector<float> hinv(hessian_length_ * hessian_length_, 0);
  if (CalculateHessianInv(hessian_data_, hinv.data(), hessian_length_, percdamp_) != RET_OK) {
    MS_LOG(ERROR) << "Calculate hessian inverse failed, tensor: " << weight_tensor_->tensor_name();
    return RET_ERROR;
  }
  // quantize per block
  for (int i1 = 0; i1 < columns_; i1 += block_size_) {
    int i2 = MSMIN(i1 + block_size_, columns_);
    int count = i2 - i1;

    elements_num = static_cast<size_t>(rows_ * count);
    std::vector<float> local_weight_data(elements_num, 0);
    std::vector<int> local_quant_data(elements_num, 0);
    std::vector<float> local_error(elements_num, 0);
    std::vector<float> local_loss(elements_num, 0);
    std::vector<float> local_hinv(count * count, 0);
    if (CloneMatrix(local_weight_data.data(), rows_, count, weight_data_, rows_, columns_, 0, rows_, i1, i2) !=
        RET_OK) {
      MS_LOG(ERROR) << "Clone weight data failed.";
      return RET_ERROR;
    }
    if (CloneMatrix(local_hinv.data(), count, count, hinv.data(), hessian_length_, hessian_length_, i1, i2, i1, i2) !=
        RET_OK) {
      MS_LOG(ERROR) << "Clone hessian inv data failed.";
      return RET_ERROR;
    }
    if (QuantizePerBlock(&local_weight_data, &local_quant_data, &local_error, &local_loss, &local_hinv, count) !=
        RET_OK) {
      MS_LOG(ERROR) << "QuantizePerBlock failed.";
      return RET_ERROR;
    }

    // updata quant_data_ and loss
    for (int i = i1; i < i2; i++) {
      for (int r = 0; r < rows_; r++) {
        // formula: Q[:, i1:i2] = Q1
        *(quant_data_ + r * columns_ + i) = local_quant_data[r * count + (i - i1)];
        // formula: Losses[:, i1:i2] = Losses1 / 2
        loss[r * columns_ + i] = local_loss[r * count + (i - i1)] / 2;
      }
    }

    elements_num = static_cast<size_t>(rows_ * (columns_ - i2));
    std::vector<float> delta(elements_num, 0);
    // Err1[:,:].matmul(Hinv[i1:i2, i2:])
    for (int r = 0; r < rows_; r++) {
      for (int c = 0; c < (columns_ - i2); c++) {
        float tmp = 0;
        for (int i = 0; i < count; i++) {
          tmp += local_error[r * count + i] * hinv[(i + i1) * hessian_length_ + (c + i2)];
        }
        delta[r * (columns_ - i2) + c] = tmp;
      }
    }

    // update weights unquantized
    // W[:, i2:] -= Err1[:,:].matmul(Hinv[i1:i2, i2:])
    for (int i = i2; i < columns_; i++) {
      for (int r = 0; r < rows_; r++) {
        *(weight_data_ + r * columns_ + i) -= delta[r * (columns_ - i2) + (i - i2)];
      }
    }
  }
  float loss_tmp = 0;
  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < columns_; c++) {
      loss_tmp += loss[r * columns_ + c];
    }
  }
  MS_LOG(WARNING) << weight_tensor_->tensor_name() << " gptq quant error: " << loss_tmp;
  return RET_OK;
}
}  // namespace mindspore::lite::quant
