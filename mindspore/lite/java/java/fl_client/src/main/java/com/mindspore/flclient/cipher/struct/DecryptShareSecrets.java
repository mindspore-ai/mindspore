/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient.cipher.struct;

import com.mindspore.flclient.Common;

import java.util.logging.Logger;

/**
 * class used for set and get decryption shards
 *
 * @since 2021-8-27
 */
public class DecryptShareSecrets {
    private static final Logger LOGGER = Logger.getLogger(DecryptShareSecrets.class.toString());
    private String flID;
    private NewArray<byte[]> sSkVu;
    private NewArray<byte[]> bVu;
    private int sIndex;
    private int indexB;

    /**
     * get flID of client
     *
     * @return flID of this client
     */
    public String getFlID() {
        if (flID == null || flID.isEmpty()) {
            LOGGER.severe(Common.addTag("[DecryptShareSecrets] the parameter of <flID> is null, please set it before " +
                    "use"));
            throw new IllegalArgumentException();
        }
        return flID;
    }

    /**
     * set flID for this client
     *
     * @param flID hash value used for identify client
     */
    public void setFlID(String flID) {
        this.flID = flID;
    }

    /**
     * get secret key shards
     *
     * @return secret key shards
     */
    public NewArray<byte[]> getSSkVu() {
        return sSkVu;
    }

    /**
     * set secret key shards
     *
     * @param sSkVu secret key shards
     */
    public void setSSkVu(NewArray<byte[]> sSkVu) {
        this.sSkVu = sSkVu;
    }

    /**
     * get bu shards
     *
     * @return bu shards
     */
    public NewArray<byte[]> getBVu() {
        return bVu;
    }

    /**
     * set bu shards
     *
     * @param bVu bu shards used for secure aggregation
     */
    public void setBVu(NewArray<byte[]> bVu) {
        this.bVu = bVu;
    }

    /**
     * get index of secret shards
     *
     * @return index of secret shards
     */
    public int getSIndex() {
        return sIndex;
    }

    /**
     * set index of secret shards
     *
     * @param sIndex index of secret shards
     */
    public void setSIndex(int sIndex) {
        this.sIndex = sIndex;
    }

    /**
     * get index of bu shards
     *
     * @return index of bu shards
     */
    public int getIndexB() {
        return indexB;
    }

    /**
     * set index of bu shards
     *
     * @param indexB index of bu shards
     */
    public void setIndexB(int indexB) {
        this.indexB = indexB;
    }
}
