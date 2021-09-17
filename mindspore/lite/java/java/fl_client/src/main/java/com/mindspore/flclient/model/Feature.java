/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

package com.mindspore.flclient.model;

/**
 * feature class
 *
 * @since v1.0
 */
public class Feature {
    int[] inputIds;
    int[] inputMasks;
    int[] tokenIds;
    int labelIds;
    int seqLen;

    /**
     * constructor
     *
     * @param inputIds input id
     * @param inputMasks input masks
     * @param tokenIds token ids
     * @param labelIds label ids
     * @param seqLen seq len
     */
    public Feature(int[] inputIds, int[] inputMasks, int[] tokenIds, int labelIds, int seqLen) {
        this.inputIds = inputIds;
        this.inputMasks = inputMasks;
        this.tokenIds = tokenIds;
        this.labelIds = labelIds;
        this.seqLen = seqLen;
    }
}
