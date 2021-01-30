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

#include <android/log.h>
#include <algorithm>
#include "ssd_util/ssd_util.h"

#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)

SSDModelUtil::~SSDModelUtil(void) {}

/**
 * SSD model util constructor.
 * @param srcImgWidth The width of the original input image.
 * @param srcImgHeight The height of the original input image.
 */
SSDModelUtil::SSDModelUtil(int srcImgWidth, int srcImgHeight) {
    inputImageWidth = srcImgWidth;
    inputImageHeight = srcImgHeight;

    getDefaultBoxes();  // To fill the vectordefaultboxes.
}


std::string SSDModelUtil::getDecodeResult(float *branchScores, float *branchBoxData) {
    std::string result = "";
    NormalBox tmpBox[1917] = {0};
    float mScores[1917][81] = {0};

    float outBuff[1917][7] = {0};

    float scoreWithOneClass[1917] = {0};
    int outBoxNum = 0;
    YXBoxes decodedBoxes[1917] = {0};

    // Copy branch outputs box data to tmpBox.
    for (int i = 0; i < 1917; ++i) {
        tmpBox[i].y = branchBoxData[i * 4 + 0];
        tmpBox[i].x = branchBoxData[i * 4 + 1];
        tmpBox[i].h = branchBoxData[i * 4 + 2];
        tmpBox[i].w = branchBoxData[i * 4 + 3];
    }

    // Copy branch outputs score to mScores.
    for (int i = 0; i < 1917; ++i) {
        for (int j = 0; j < 81; ++j) {
            mScores[i][j] = branchScores[i * 81 + j];
        }
    }

    // NMS processing.
    ssd_boxes_decode(tmpBox, decodedBoxes, 0.1, 0.2, 1917);
    const float nms_threshold = 0.3;
    for (int i = 1; i < 81; i++) {
        std::vector<int> in_indexes;
        for (int j = 0; j < 1917; j++) {
            scoreWithOneClass[j] = mScores[j][i];
            if (mScores[j][i] > g_thres_map[i]) {
                in_indexes.push_back(j);
            }
        }
        if (in_indexes.size() == 0) {
            continue;
        }

        sort(in_indexes.begin(), in_indexes.end(),
             [&](int a, int b) { return scoreWithOneClass[a] > scoreWithOneClass[b]; });
        std::vector<int> out_indexes;

        nonMaximumSuppression(decodedBoxes, scoreWithOneClass, in_indexes, &out_indexes,
                              nms_threshold);
        for (int k = 0; k < out_indexes.size(); k++) {
            // image id
            outBuff[outBoxNum][0] = out_indexes[k];
            // labelid
            outBuff[outBoxNum][1] = i;
            // scores
            outBuff[outBoxNum][2] = scoreWithOneClass[out_indexes[k]];
            outBuff[outBoxNum][3] =
                    decodedBoxes[out_indexes[k]].xmin * inputImageWidth / 300;
            outBuff[outBoxNum][4] =
                    decodedBoxes[out_indexes[k]].ymin * inputImageHeight / 300;
            outBuff[outBoxNum][5] =
                    decodedBoxes[out_indexes[k]].xmax * inputImageWidth / 300;
            outBuff[outBoxNum][6] =
                    decodedBoxes[out_indexes[k]].ymax * inputImageHeight / 300;
            outBoxNum++;
        }
    }
    MS_PRINT("outBoxNum %d", outBoxNum);

    for (int i = 0; i < outBoxNum; ++i) {
        std::string tmpid_str = std::to_string(outBuff[i][0]);
        result += tmpid_str;
        result += "_";
        MS_PRINT("label_classes i %d, outBuff %d", i, (int) outBuff[i][1]);
        tmpid_str = std::to_string(static_cast<int>(outBuff[i][1]));
        // label id
        result += tmpid_str;
        result += "_";
        tmpid_str = std::to_string(outBuff[i][2]);
        // scores
        result += tmpid_str;
        result += "_";
        tmpid_str = std::to_string(outBuff[i][3]);
        // xmin
        result += tmpid_str;
        result += "_";
        tmpid_str = std::to_string(outBuff[i][4]);
        // ymin
        result += tmpid_str;
        result += "_";
        tmpid_str = std::to_string(outBuff[i][5]);
        // xmax
        result += tmpid_str;
        result += "_";
        tmpid_str = std::to_string(outBuff[i][6]);
        // ymax
        result += tmpid_str;
        result += ";";
    }

    return result;
}

void SSDModelUtil::getDefaultBoxes() {
    float fk[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<struct WHBox> all_sizes;
    struct Product mProductData[19 * 19] = {0};

    for (int i = 0; i < 6; i++) {
        fk[i] = config.model_input_height / config.steps[i];
    }
    float scale_rate =
            (config.max_scale - config.min_scale) / (sizeof(config.num_default) / sizeof(int) - 1);
    float scales[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    for (int i = 0; i < sizeof(config.num_default) / sizeof(int); i++) {
        scales[i] = config.min_scale + scale_rate * i;
    }

    for (int idex = 0; idex < sizeof(config.feature_size) / sizeof(int); idex++) {
        float sk1 = scales[idex];
        float sk2 = scales[idex + 1];
        float sk3 = sqrt(sk1 * sk2);
        struct WHBox tempWHBox;

        all_sizes.clear();

        // idex == 0时， len(all_sizes) = 3.
        if (idex == 0) {
            float w = sk1 * sqrt(2);
            float h = sk1 / sqrt(2);

            // all_sizes = [(0.1, 0.1), (w, h), (h, w)]
            tempWHBox.boxw = 0.1;
            tempWHBox.boxh = 0.1;
            all_sizes.push_back(tempWHBox);

            tempWHBox.boxw = w;
            tempWHBox.boxh = h;
            all_sizes.push_back(tempWHBox);

            tempWHBox.boxw = h;
            tempWHBox.boxh = w;
            all_sizes.push_back(tempWHBox);
        } else {
            // len(all_sizes) = 6.
            tempWHBox.boxw = sk1;
            tempWHBox.boxh = sk1;
            all_sizes.push_back(tempWHBox);

            for (int j = 0; j < sizeof(config.aspect_ratios[idex]) / sizeof(int); j++) {
                float w = sk1 * sqrt(config.aspect_ratios[idex][j]);
                float h = sk1 / sqrt(config.aspect_ratios[idex][j]);

                tempWHBox.boxw = w;
                tempWHBox.boxh = h;
                all_sizes.push_back(tempWHBox);

                tempWHBox.boxw = h;
                tempWHBox.boxh = w;
                all_sizes.push_back(tempWHBox);
            }

            tempWHBox.boxw = sk3;
            tempWHBox.boxh = sk3;
            all_sizes.push_back(tempWHBox);
        }

        for (int i = 0; i < config.feature_size[idex]; i++) {
            for (int j = 0; j < config.feature_size[idex]; j++) {
                mProductData[i * config.feature_size[idex] + j].x = i;
                mProductData[i * config.feature_size[idex] + j].y = j;
            }
        }

        int productLen = config.feature_size[idex] * config.feature_size[idex];

        for (int i = 0; i < productLen; i++) {
            for (int j = 0; j < all_sizes.size(); j++) {
                struct NormalBox tempBox;

                float cx = (mProductData[i].y + 0.5) / fk[idex];
                float cy = (mProductData[i].x + 0.5) / fk[idex];

                tempBox.y = cy;
                tempBox.x = cx;
                tempBox.h = all_sizes[j].boxh;
                tempBox.w = all_sizes[j].boxw;

                mDefaultBoxes.push_back(tempBox);
            }
        }
    }
}


void SSDModelUtil::ssd_boxes_decode(const NormalBox *boxes,
                                    YXBoxes *const decoded_boxes, const float scale0,
                                    const float scale1, const int count) {
    if (mDefaultBoxes.size() == 0) {
        MS_PRINT("get default boxes error.");
        return;
    }

    for (int i = 0; i < count; ++i) {
        float cy = boxes[i].y * scale0 * mDefaultBoxes[i].h + mDefaultBoxes[i].y;
        float cx = boxes[i].x * scale0 * mDefaultBoxes[i].w + mDefaultBoxes[i].x;
        float h = exp(boxes[i].h * scale1) * mDefaultBoxes[i].h;
        float w = exp(boxes[i].w * scale1) * mDefaultBoxes[i].w;
        decoded_boxes[i].ymin =
                std::min(1.0f, std::max(0.0f, cy - h / 2)) * config.model_input_height;
        decoded_boxes[i].xmin =
                std::min(1.0f, std::max(0.0f, cx - w / 2)) * config.model_input_width;
        decoded_boxes[i].ymax =
                std::min(1.0f, std::max(0.0f, cy + h / 2)) * config.model_input_height;
        decoded_boxes[i].xmax =
                std::min(1.0f, std::max(0.0f, cx + w / 2)) * config.model_input_width;
    }
}

void SSDModelUtil::nonMaximumSuppression(const YXBoxes *const decoded_boxes,
                                         const float *const scores,
                                         const std::vector<int> &in_indexes,
                                         std::vector<int> *out_indexes_p, const float nmsThreshold,
                                         const int count, const int max_results) {
    int nR = 0;
    std::vector<int> &out_indexes = *out_indexes_p;
    std::vector<bool> del(count, false);
    for (size_t i = 0; i < in_indexes.size(); i++) {
        if (!del[in_indexes[i]]) {
            out_indexes.push_back(in_indexes[i]);
            if (++nR == max_results) {
                break;
            }
            for (size_t j = i + 1; j < in_indexes.size(); j++) {
                const auto boxi = decoded_boxes[in_indexes[i]], boxj = decoded_boxes[in_indexes[j]];
                float a[4] = {boxi.xmin, boxi.ymin, boxi.xmax, boxi.ymax};
                float b[4] = {boxj.xmin, boxj.ymin, boxj.xmax, boxj.ymax};
                if (IOU(a, b) > nmsThreshold) {
                    del[in_indexes[j]] = true;
                }
            }
        }
    }
}

double SSDModelUtil::IOU(float r1[4], float r2[4]) {
    float x1 = std::max(r1[0], r2[0]);
    float y1 = std::max(r1[1], r2[1]);
    float x2 = std::min(r1[2], r2[2]);
    float y2 = std::min(r1[3], r2[3]);
    // if max(min) > min(max), there is no intersection
    if (x2 - x1 + 1 <= 0 || y2 - y1 + 1 <= 0)
        return 0;
    double insect_area = (x2 - x1 + 1) * (y2 - y1 + 1);
    double union_area =
            (r1[2] - r1[0] + 1) * (r1[3] - r1[1] + 1) + (r2[2] - r2[0] + 1) * (r2[3] - r2[1] + 1) -
            insect_area;
    double iou = insect_area / union_area;
    return (iou > 0) ? iou : 0;
}
