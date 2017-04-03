package usp.each.ppgsi.helpers;

import twitter4j.JSONArray;
import twitter4j.JSONException;
import twitter4j.JSONObject;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by felipealvesdias on 02/04/17.
 */
public class AccountsReader {
    private static final String accountsFilename = "accounts.json";
    private static AccountsReader instance;

    private AccountsReader() {
    }

    public static Map<String, String> getAccounts() {

        if (instance == null) {
            instance = new AccountsReader();
        }

        InputStream input = AccountsReader.class.getClassLoader().getResourceAsStream(accountsFilename);
        String content = convertStreamToString(input).replace("\n", "");
        Map<String, String> accounts = new HashMap<>();
        try {
            JSONObject jsonObject = new JSONObject(content);
            JSONArray jsonArray = jsonObject.getJSONArray("accounts");
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject account = jsonArray.getJSONObject(i);
                String key = account.keys().next().toString();
                accounts.put(key, account.getString(key));
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return accounts;
    }

    private static String convertStreamToString(java.io.InputStream is) {
        java.util.Scanner s = new java.util.Scanner(is).useDelimiter("\\A");
        return s.hasNext() ? s.next() : "";
    }
}
