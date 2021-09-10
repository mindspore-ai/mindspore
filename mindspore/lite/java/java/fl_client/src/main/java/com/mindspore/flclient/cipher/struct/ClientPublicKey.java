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
 * public key class of secure aggregation
 *
 * @since 2021-8-27
 */
public class ClientPublicKey {
    private static final Logger LOGGER = Logger.getLogger(ClientPublicKey.class.toString());
    private String flID;
    private NewArray<byte[]> cPK;
    private NewArray<byte[]> sPk;
    private NewArray<byte[]> pwIv;
    private NewArray<byte[]> pwSalt;

    /**
     * get client's flID
     *
     * @return flID of this client
     */
    public String getFlID() {
        if (flID == null || flID.isEmpty()) {
            LOGGER.severe(Common.addTag("[ClientPublicKey] the parameter of <flID> is null, please set it before use"));
            throw new IllegalArgumentException();
        }
        return flID;
    }

    /**
     * set client's flID
     *
     * @param flID hash value used for identify client
     */
    public void setFlID(String flID) {
        this.flID = flID;
    }

    /**
     * get CPK of secure aggregation
     *
     * @return CPK of secure aggregation
     */
    public NewArray<byte[]> getCPK() {
        return cPK;
    }

    /**
     * set CPK of secure aggregation
     *
     * @param cPK public key used for encryption
     */
    public void setCPK(NewArray<byte[]> cPK) {
        this.cPK = cPK;
    }

    /**
     * get SPK of secure aggregation
     *
     * @return SPK of secure aggregation
     */
    public NewArray<byte[]> getSPK() {
        return sPk;
    }

    /**
     * set SPK of secure aggregation
     *
     * @param sPk public key used for encryption
     */
    public void setSPK(NewArray<byte[]> sPk) {
        this.sPk = sPk;
    }

    /**
     * get the IV value used for pairwise encrypt
     *
     * @return the IV value used for pairwise encrypt
     */
    public NewArray<byte[]> getPwIv() {
        return pwIv;
    }

    /**
     * set the IV value used for pairwise encrypt
     *
     * @param pwIv IV value used for pairwise encrypt
     */
    public void setPwIv(NewArray<byte[]> pwIv) {
        this.pwIv = pwIv;
    }

    /**
     * get salt value for secure aggregation
     *
     * @return salt value for secure aggregation
     */
    public NewArray<byte[]> getPwSalt() {
        return pwSalt;
    }

    /**
     * set salt value for secure aggregation
     *
     * @param pwSalt salt value for secure aggregation
     */
    public void setPwSalt(NewArray<byte[]> pwSalt) {
        this.pwSalt = pwSalt;
    }
}
