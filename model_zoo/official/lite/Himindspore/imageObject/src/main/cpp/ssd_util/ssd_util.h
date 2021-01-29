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

#ifndef HIMINDSPORE_SSD_UTIL_H
#define HIMINDSPORE_SSD_UTIL_H

#include <string>
#include <vector>


class SSDModelUtil {
 public:
    // Constructor.
    SSDModelUtil(int srcImageWidth, int srcImgHeight);

    ~SSDModelUtil();

    /**
     * Return the SSD model post-processing result.
     * @param branchScores
     * @param branchBoxData
     * @return
     */
    std::string getDecodeResult(float *branchScores, float *branchBoxData);

    struct NormalBox {
        float y;
        float x;
        float h;
        float w;
    };

    struct YXBoxes {
        float ymin;
        float xmin;
        float ymax;
        float xmax;
    };

    struct Product {
        int x;
        int y;
    };

    struct WHBox {
        float boxw;
        float boxh;
    };

 private:
    std::vector<struct NormalBox> mDefaultBoxes;
    int inputImageHeight;
    int inputImageWidth;

    void getDefaultBoxes();

    void ssd_boxes_decode(const NormalBox *boxes,
                          YXBoxes *const decoded_boxes,
                          const float scale0 = 0.1, const float scale1 = 0.2,
                          const int count = 1917);

    void nonMaximumSuppression(const YXBoxes *const decoded_boxes, const float *const scores,
                               const std::vector<int> &in_indexes, std::vector<int> *out_indexes_p,
                               const float nmsThreshold = 0.6,
                               const int count = 1917, const int max_results = 100);

    double IOU(float r1[4], float r2[4]);

    // ============= variables =============.
    struct network {
        int model_input_height = 300;
        int model_input_width = 300;

        int num_default[6] = {3, 6, 6, 6, 6, 6};
        int feature_size[6] = {19, 10, 5, 3, 2, 1};
        double min_scale = 0.2;
        float max_scale = 0.95;
        float steps[6] = {16, 32, 64, 100, 150, 300};
        float prior_scaling[2] = {0.1, 0.2};
        float gamma = 2.0;
        float alpha = 0.75;
        int aspect_ratios[6][2] = {{2, 0},
                                   {2, 3},
                                   {2, 3},
                                   {2, 3},
                                   {2, 3},
                                   {2, 3}};
    } config;

    float g_thres_map[81] = {0, 0.635, 0.627, 0.589, 0.585, 0.648, 0.664, 0.655,
                             0.481, 0.529, 0.611, 0.641, 0.774, 0.549, 0.513, 0.652,
                             0.552, 0.590, 0.650, 0.575, 0.583, 0.650, 0.656, 0.696,
                             0.653, 0.438, 0.515, 0.459, 0.561, 0.545, 0.635, 0.540,
                             0.560, 0.721, 0.544, 0.548, 0.511, 0.611, 0.592, 0.542,
                             0.512, 0.635, 0.531, 0.437, 0.525, 0.445, 0.484, 0.546,
                             0.490, 0.581, 0.566, 0.516, 0.445, 0.541, 0.613, 0.560,
                             0.483, 0.509, 0.464, 0.543, 0.538, 0.490, 0.576, 0.617,
                             0.577, 0.595, 0.640, 0.585, 0.598, 0.592, 0.514, 0.397,
                             0.592, 0.504, 0.548, 0.642, 0.581, 0.497, 0.545, 0.154,
                             0.580,
    };
};

#endif
