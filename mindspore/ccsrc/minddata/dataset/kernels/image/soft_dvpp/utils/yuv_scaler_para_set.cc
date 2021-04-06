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

#include "minddata/dataset/kernels/image/soft_dvpp/utils/yuv_scaler_para_set.h"
#include <securec.h>
#include <fstream>
#include <sstream>
#include <utility>
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_check.h"
#include "minddata/dataset/kernels/image/soft_dvpp/utils/soft_dp_tools.h"

pthread_mutex_t YuvScalerParaSet::g_mutex_ = PTHREAD_MUTEX_INITIALIZER;
YuvWPara *YuvScalerParaSet::g_m_instance_ = nullptr;
YuvScalerParaSet::GarbageCollector YuvScalerParaSet::g_collector_;

const int32_t dpSucc = 0;
const int32_t dpFail = -1;

/*
 * @brief : Replaces the specified symbol in a string with another symbol.
 * @param [in] const string &strSrc : src string.
 * @param [in] const string &strDst : dest string.
 */
void StringReplace(std::string *str_big, const std::string &str_src, const std::string &str_dst) {
  std::string::size_type pos = 0;
  std::string::size_type src_len = str_src.size();
  std::string::size_type dst_len = str_dst.size();
  while ((pos = str_big->find(str_src, pos)) != std::string::npos) {
    str_big->replace(pos, src_len, str_dst);
    pos += dst_len;
  }
}

/*
 * @brief : Parse the data in the character string and transfer the data to the structure.
 * @param [in] string strLine : parsed string.
 * @param [in] int32_t *flagCtl : the number of char.
 * @param [in] int32_t *flagTap : the flag of char.
 * @param [in] YuvWPara *yuvScalerParaSet : yuv scaler param sets.
 * @param [in] ScalerCoefficientIndex *index : scaler index.
 */
void GetParaSet(std::string str_line, int32_t *flag_ctl, int32_t *flag_tap, YuvWPara *yuv_scaler_paraset,
                ScalerCoefficientIndex *index) {
  std::stringstream ss;
  StringReplace(&str_line, ",", " ");  // Replaces commas in a string with spaces.
  ss.str(str_line);
  int32_t cnt = yuv_scaler_paraset->real_count;  // Number of saved arrays.
  const int32_t arrTypeNum = 3;
  const int32_t initBracketNum = 3;

  // {start,end}
  if ((*flag_ctl - initBracketNum) % arrTypeNum == 1) {
    char chTmp;
    ss >> chTmp >> yuv_scaler_paraset->scale[cnt].range.start >> yuv_scaler_paraset->scale[cnt].range.end;

    if (ss.fail()) {  // read failed.
#ifndef DVPP_UTST
      ss.clear();
#endif
    }
  }

  // taps_4, the second character in the square brackets is the start address of the array block.
  if ((*flag_ctl - initBracketNum) % arrTypeNum == 2) {
    while (1) {
      ss >> yuv_scaler_paraset->scale[cnt].taps_4[index->first_index++];
      if (ss.fail()) {  // rerad failed.
        index->first_index = index->first_index - 1;
        ss.clear();
        break;
      }

      if (index->first_index == kScalerCoffNb4) {  // read finish
        index->first_index = 0;
        *flag_tap = 0;
        ss.clear();
        break;
      }
    }
  }

  // taps_6
  if ((*flag_ctl - initBracketNum) % arrTypeNum == 0) {
    while (1) {
      ss >> yuv_scaler_paraset->scale[cnt].taps_6[index->second_index++];
      if (ss.fail()) {  // read failed.
        index->second_index = index->second_index - 1;
        ss.clear();
        break;
      }

      if (index->second_index == kScalerCoffNb6) {  // read finish.
        index->second_index = 0;
        *flag_tap = 0;
        ss.clear();
        ++(yuv_scaler_paraset->real_count);
        *flag_ctl = *flag_ctl - 4;  // The filtering parameter set has four large blocks.
        break;
      }
    }
  }
}

int32_t CheckParamater(std::pair<bool, std::string> rlt, uint32_t i) {
  int32_t ret = dpSucc;
  if (rlt.first == false) {
    API_LOGE("Get real path failed. index = %u", i);
    return dpFail;
  }

  if (IsDirectory(rlt.second)) {
    API_LOGE("It is a directory, not file path. index = %u", i);
    return dpFail;
  }

  return ret;
}

// Read the parameter set file and skip the comments in the file.
int32_t ParseFileToVar(std::string *para_set_name, uint32_t yuv_scaler_paraset_size, YuvWPara *yuv_scaler_paraset) {
  int32_t ret = dpSucc;

  VPC_CHECK_COND_FAIL_RETURN(para_set_name != nullptr, dpFail);
  VPC_CHECK_COND_FAIL_RETURN(yuv_scaler_paraset != nullptr, dpFail);

  uint32_t i = 0;
  while (i < yuv_scaler_paraset_size && i < maxFileCount && (!para_set_name[i].empty())) {
    std::string str_line;

    // Standardize the file path and check whether the path exists.
    std::pair<bool, std::string> rlt = GetRealpath(para_set_name[i]);
    ret = CheckParamater(rlt, i);
    if (ret != dpSucc) {
      return ret;
    }

    std::ifstream inFile(rlt.second);

    int32_t flag_tap = 1;
    int32_t flag_ctl = 0;
    int32_t flag_anno = 0;
    ScalerCoefficientIndex index;
    const int32_t initBracketNum = 3;
    yuv_scaler_paraset[i].real_count = 0;

    while (getline(inFile, str_line)) {  // read each row of data.
      // Skip the comments.
      if (str_line.find("/*") != std::string::npos) {
        flag_anno = 1;
        continue;
      }

      if (flag_anno) {
        if (str_line.find("*/") != std::string::npos) {
          flag_anno = 0;
          continue;
        }
        continue;
      }

      if (str_line.find("//") != std::string::npos) {
        continue;
      }

      // cale the number of "{",check the location of the data.
      if (str_line.find("{") != std::string::npos) {
        flag_ctl++;
        flag_tap = 1;
      }

      if (flag_ctl > initBracketNum && flag_tap == 1) {  // parse params
        GetParaSet(str_line, &flag_ctl, &flag_tap, &yuv_scaler_paraset[i], &index);
      }
    }

    inFile.close();
    ++i;
  }
  return ret;
}

YuvWPara *YuvScalerParaSet::GetInstance(std::string *paraset_name, uint32_t yuv_scaler_paraset_size) {
  if (g_m_instance_ == nullptr) {
    (void)pthread_mutex_lock(&g_mutex_);
    if (g_m_instance_ == nullptr) {
      if (paraset_name == nullptr) {
#ifndef API_MAR_UT
#ifdef DVPP_UTST
        YuvWPara p_tmp[10];  // 10: 滤波参数集最大数
        p_tmp[0] = YUV_W_PARA;
        g_m_instance_ = p_tmp;
#else
        auto p_tmp = static_cast<YuvWPara *>(malloc(sizeof(YuvWPara) * maxFileCount));
        if (p_tmp == nullptr) {
          API_LOGE("malloc YuvWPara fail!");
          g_m_instance_ = nullptr;
          (void)pthread_mutex_unlock(&g_mutex_);
          return g_m_instance_;
        }

        uint32_t ret = memcpy_s(&p_tmp[0], sizeof(p_tmp[0]), &YUV_W_PARA, sizeof(YUV_W_PARA));
        if (ret != EOK) {
          API_LOGE("memcpy_s p_tmp[0] fail!");
          g_m_instance_ = nullptr;
          free(p_tmp);
          p_tmp = nullptr;
          (void)pthread_mutex_unlock(&g_mutex_);
          return g_m_instance_;
        }

        g_m_instance_ = p_tmp;
#endif
#endif
      } else {
        auto p_tmp = static_cast<YuvWPara *>(malloc(sizeof(YuvWPara) * maxFileCount));
        if (p_tmp == nullptr) {
#ifndef DVPP_UTST
          API_LOGE("malloc YuvWPara fail!");
          g_m_instance_ = nullptr;
          (void)pthread_mutex_unlock(&g_mutex_);
          return g_m_instance_;
#endif
        }

        if (ParseFileToVar(paraset_name, yuv_scaler_paraset_size, p_tmp) == -1) {
          free(p_tmp);
          g_m_instance_ = nullptr;
        } else {
          g_m_instance_ = p_tmp;
        }
      }
    }
    (void)pthread_mutex_unlock(&g_mutex_);
  }
  return g_m_instance_;
}

// Searching for the index number of the filtering parameter by using the dichotomy
int32_t GetScalerParameterIndex(uint32_t parameter, YuvWPara *parameterset) {
  int32_t count = parameterset->real_count;
  int32_t left = 0;
  int32_t right = count - 1;
  YuvScalerPara *scaler = parameterset->scale;
  int32_t index = 0;

  if (parameter <= scalerRadio1Time) {
    index = 0;
  } else {
    parameter = parameter >> parameterInterval;
    while (left <= right) {
      index = (left + right) / 2;  // 2-point search
      if (parameter > scaler[index].range.start && parameter <= scaler[index].range.end) {
        break;
      }
      if (parameter > scaler[index].range.end) {
        left = index + 1;
      } else if (parameter <= scaler[index].range.start) {
        right = index - 1;
      }
    }
  }

  if (left > right) {
    index = count - 1;
  }
  return index;
}
