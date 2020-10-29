package com.mindspore.himindspore.contract;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.mindspore.himindspore.R;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ContractActivity extends AppCompatActivity implements View.OnClickListener {

    private EditText emailEdit;
    private Button submitBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_contract);

        emailEdit = findViewById(R.id.emailEditText);
        submitBtn = findViewById(R.id.submitBtn);
        submitBtn.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {

        if (R.id.submitBtn == view.getId()) {
            String email = emailEdit.getText().toString();
            if (TextUtils.isEmpty(email)) {
                Toast.makeText(ContractActivity.this,"Please input your email!",Toast.LENGTH_LONG).show();
                return;
            }
            if (isEmailFormat(email)){

            }else{
                Toast.makeText(ContractActivity.this,"The email address you enterd is not in the correct format",Toast.LENGTH_LONG).show();
                return;
            }
        }
    }

    private boolean isEmailFormat(String emailAdd) {
        Pattern p = Pattern.compile("^([a-zA-Z0-9_-])+@([a-zA-Z0-9_-])+(\\.([a-zA-Z0-9_-])+)+$");
        Matcher m = p.matcher(emailAdd);
        return m.matches();
    }
}