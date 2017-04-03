package usp.each.ppgsi.tweets;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONObject;

/**
 * Created by felipealvesdias on 02/04/17.
 */
public class GoogleMapsService {
    public static JSONObject getLatLong(String address) {
        String url = "https://maps.googleapis.com/maps/api/geocode/json?address=" +
                address + "&region=br&key=AIzaSyCh8Hm-P0SV5Fwy0ZY4eZtvXa3mUsMMlwQ";
        HttpResponse<String> response;
        JSONObject body;
        JSONObject result = null;
        try {
            response = Unirest.get(url).header("content-type", "application/json").asString();
            body = new JSONObject(response.getBody());
            if (body.has("results") && body.getString("status").contains("OK")) {
                result = body.getJSONArray("results").getJSONObject(0);
            }
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return result;
    }
}
