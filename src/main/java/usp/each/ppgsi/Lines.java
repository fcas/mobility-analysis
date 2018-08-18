package usp.each.ppgsi;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;

public class Lines {
    public static void main(String[] args) throws UnirestException {
        HttpResponse<String> response = Unirest.get("http://api.olhovivo.sptrans.com.br/v2.1/Linha/Buscar?termosBusca=1012-10")
                .header("Cache-Control", "no-cache")
                .header("Postman-Token", "2ccc790b-232a-efcc-bc26-3a728af20edf")
                .asString();
        System.out.print(response.getBody());
    }
}
