package com.mindspore.himindspore.base;

import android.app.Activity;
import android.os.Bundle;

public abstract class BaseActivity<T extends BasePresenter> extends Activity {

    protected T presenter;

    @Override
    public void onCreate(Bundle savedInstance) {
        super.onCreate(savedInstance);
        setContentView(getLayout());
        init();
    }

    protected abstract void init();

    public abstract int getLayout();
}
