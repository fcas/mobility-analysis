package usp.each.ppgsi.tweets;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by felipealvesdias on 02/04/17.
 */
public class TweetProcessor {

    private static String p1 = "(RUA|Rua|R.|R. Dr.|AVENIDA|Avenida|AV.|Av.|Av. Cel.|TRAVESSA|Travessa|TRAV.|Trav.|Viaduto|Marg.)" +
            " ([a-zA-Z_]+)";
    private static String p2 = "(RUA|Rua|R.|R. Dr.|AVENIDA|Avenida|AV.|Av.|Av. Cel.|TRAVESSA|Travessa|TRAV.|Trav.|Viaduto|Marg.)" +
            " ([a-zA-Z_]+) ([a-zA-Z_]+)";
    private static String p3 = "(RUA|Rua|R.|R. Dr.|AVENIDA|Avenida|AV.|Av.|Av. Cel.|TRAVESSA|Travessa|TRAV.|Trav.|Viaduto|Marg.)" +
            " ([a-zA-Z_]+) ([a-zA-Z_]+) ([a-zA-Z_]+)";
    private static String p4 = "(RUA|Rua|R.|R. Dr.|AVENIDA|Avenida|AV.|Av.|Av. Cel.|TRAVESSA|Travessa|TRAV.|Trav.|Viaduto|Marg.)" +
            " ([a-zA-Z_]+) ([a-zA-Z_]+) ([a-zA-Z_]+) ([a-zA-Z_]+)";
    private static String p5 = "(RUA|Rua|R.|R. Dr.|AVENIDA|Avenida|AV.|Av.|Av. Cel.|TRAVESSA|Travessa|TRAV.|Trav.|Viaduto|Marg.)" +
            " ([a-zA-Z_]+) ([a-zA-Z_]+) ([a-zA-Z_]+) ([a-zA-Z_]+) ([a-zA-Z_]+)";

    private static String findAddress(String pattern, String tweet) {
        String address = "";
        Pattern p = Pattern.compile(pattern);
        Matcher matcher = p.matcher(tweet);
        if (matcher.find()) {
            address = matcher.group(0);
        }
        return address;
    }

    public static String getAddress(String tweet) {
        String address;

        tweet = tweet
                .replace("ã", "a")
                .replace("á", "a")
                .replace("à", "a")
                .replace("â", "a")
                .replace("õ", "o")
                .replace("ó", "o")
                .replace("ô", "o")
                .replace("ç", "c")
                .replace("í", "i")
                .replace("é", "e")
                .replace("ê", "e")
                .replace("ú", "u");

        tweet = tweet.split(" liberada")[0];
        tweet = tweet.split(" sentido")[0];
        tweet = tweet.split(" junto a")[0];
        tweet = tweet.split(" ocupacao total")[0];
        tweet = tweet.split(" em direcao a")[0];
        tweet = tweet.split(" em ambos")[0];
        tweet = tweet.split(" proximo a")[0];
        tweet = tweet.split(" com a")[0];
        tweet = tweet.split("e Rua")[0];
        tweet = tweet.split("com Rua")[0];
        tweet = tweet.split("permanece")[0];
        tweet = tweet.split("liberado.")[0];

        address = findAddress(p5, tweet);

        if (address.isEmpty()) {
            address = findAddress(p4, tweet);
        }

        if (address.isEmpty()) {
            address = findAddress(p3, tweet);
        }

        if (address.isEmpty()) {
            address = findAddress(p3, tweet);
        }

        if (address.isEmpty()) {
            address = findAddress(p2, tweet);
        }

        if (address.isEmpty()) {
            address = findAddress(p1, tweet);
        }

        return address;
    }

    public static void main(String[] args) {
        System.out.println(getAddress("Viaduto do Chá liberado. #ZC"));
    }
}
