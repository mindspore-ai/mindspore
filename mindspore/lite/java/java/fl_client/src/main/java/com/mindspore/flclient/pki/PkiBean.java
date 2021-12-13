/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 */

package com.mindspore.flclient.pki;

import java.security.cert.Certificate;

/**
 * PkiBean entity
 *
 * @since 2021-08-25
 */
public class PkiBean {
    private byte[] signData;

    private Certificate[] certificates;

    public PkiBean(byte[] signData, Certificate[] certificates) {
        this.signData = signData;
        this.certificates = certificates;
    }

    public byte[] getSignData() {
        return signData;
    }

    public void setSignData(byte[] signData) {
        this.signData = signData;
    }

    public Certificate[] getCertificates() {
        return certificates;
    }

    public void setCertificates(Certificate[] certificates) {
        this.certificates = certificates;
    }
}
