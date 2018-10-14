ids = {
    "classified_tweets_CETSP_",
    "classified_tweets_BombeirosPMESP",
    "classified_tweets_PMESP",
    "classified_tweets_governosp",
    "classified_tweets_metrosp_oficial",
    "classified_tweets_TurismoSaoPaulo",
    "classified_tweets_smtsp",
    "classified_tweets_SPCEDEC",
    "classified_tweets_saopaulo_agora",
    "classified_tweets_Policia_Civil",
    "classified_tweets_CPTM_oficial",
    "classified_tweets_sptrans"
}

patterns_to_be_removed_pre_address = [
    r"#(\w+)", r"http\S+", r"http", r"@\S+", r"[0-9]h", r"[0-9]/[0-9]", r"[0-9][0-9]h[0-9][0-9]", r"n° "
    r"[0-9][0-9]:[0-9][0-9]", r"\\\\n", r"\\n", r"\\\\", r"\.", r"/", r",", r";", r"nº", r"n°", r"º", r"n º"
]

patterns_to_be_removed_pos_address = [
    r"[^A-Za-z _]"
]

location_pattern = r"(\brua |\br |\bavenida |\bav |\btravessa |\btrav |\bviaduto |\bmarg |\bmarginal |\btúnel |\blg |" \
                   r"\blargo |\bponte |\bpraça |\bpça ) ([a-zÀ-ÿ_]+).*"
location_pattern_exception = r"(\brua |\br |\bavenida |\bav |\btravessa |\btrav |\bviaduto |\bmarg |\bmarginal " \
                             r"|\btúnel |\blg |\blargo |\bponte |\bpraça |\bpça )([a-zÀ-ÿ_]+).*"

address_padding = "padding padding padding padding padding padding"

address_pre_patterns = [
    "no momento ",
    "na esquina ",
    "na vila",
    "em ambos os sentidos",
    "ambos os sentidos",
    "ambos sentidos",
    "em ambos",
    "foram alv",
    "pista expressa ",
    "pista da ",
    "permanece ",
    "os manifestantes",
    "manifestantes em",
    "para obra",
    "para pedestres",
    "inter",
    "ocupa duas",
    " com r",
    " com a",
    " com destino",
    "e av",
    "e viaduto",
    "e ponte",
    "e rua",
    "x av",
    "e vd",
    "duas faixas",
    "liberad",
    "devido ",
    "e desemboque ",
    "entre a ",
    "junto ",
    "ocupa ",
    "via\\b",
    "altura ",
    "ficou",
    "foi",
    "no sentido ",
    "sentido ",
    "manifestantes",
    "entre r",
    "acesso ao",
    "evite ",
    "veja",
    "tem emboque",
    "x pça",
    "pista local",
    "referente",
    "sent",
    "e término",
    "e o caso",
    "o caso",
    "agradecemos",
    "em direção",
    "agora estão",
    "estará",
    "está",
    "para ",
    "e r ",
    "após",
    "terá",
    "será",
    "próximo",
    "a cet",
    "padding",
    "faco nada",
    "então ",
    "segue ",
]

address_pos_patterns = [
    "defesa civil",
    "promove campanhas",
    "vítima ",
    "apos ponte",
    "pref",
    "subpref",
    "e fizemos a",
    "ao chegar",
    "óbito",
    "linhas",
    "nao ha",
    "\(",
    "\/"
    "houve",
    "-",
    "–",
    "vit",
    "1ª",
    "fogo ",
    "a novembro",
    "infelizmente ",
    "confirmado ",
    "sobre a ",
    "vít",
    "sem vitms",
    "disponib",
    "ocorreu",
    "linhas da",
    "não há",
    "aprox",
    "o cb",
    "def civil",
    "defcivil",
    "defesa"
    "solicitante",
    "\+info",
    "e o",
    "final",
    "conta com",
    "veículo",
    "nas linhas",
    "motivam",
    "interior do",
    "a partir",
    "interdita",
    "acesso da",
    "a primeira",
    "saiba mais",
    "oferece",
    "equip",
    "tem novo",
    "e na ",
    "nos dias ",
    " x ",
    "antes da",
    "mais info",
    "frente ",
    "com ",
    "ref ",
    "a partir",
    "ocorrência ",
    "e remov",
    "porém",
    "nossa inter",
    "afeta ",
    "disponi"
    "entr",
    "prox",
    "fica em",
    "s inf",
    "pequeno ",
    "que atend",
    "e nada",
    "a circ",
    "no loc",
    "aguard",
    "o auto",
    "e for",
    "às",
    "na av",
    "o viad",
    "no aces",
    "provoca ",
    "bombeiros e",
    "de acord",
    "corpo de",
    "local ",
    "percorrido",
    "nada ",
    "e edif",
    "agora ",
    "por ",
    "px ",
    "em pri",
    "área",
    "observ",
    "ligue ",
    "motiva ",
    "é de",
    "parab",
    "neste ",
    "terminal ",
    "polici",
    "bombeir",
    "é um ",
    "já a",
    "ocup fx",
    "ela ganho",
    "informações",
    " datas",
    "a obra",
    "e suas",
    "info de",
    "até a",
    "terão ",
    "comando ",
    "temos ",
    "inform",
    "dados ao",
    "polícia",
    "no mesmo",
    "ajudando",
    "foram ",
    "pista ",
    "não houve",
    "regional ",
    "tráfego ",
    "ñ",
    "cruzamento ",
    "não sabe",
    "concentra",
    "socorrida",
    "esquina ",
    "defronte ",
    "segundo solicitante",
    "segundo o solicitante",
    "evadiu",
    "aproveite",
    "que funciona",
    "e fizemos",
    "atenderá ao público",
    "ao fundo o",
    "serão desviadas",
    "acesse:",
    "barracos queimados",
    "é repleta",
    "subindo a",
    "solicitantes",
    "no domingo",
    "desviam",
    "durante julho",
    "recebe a",
    "atenderá",
    "ganham sinalização",
    "visite ",
    "boa tarde",
    "próx",
    "arquivo",
    "implementado",
    "hoje e amanhã",
    "altera",
    "na virada",
    "na região",
    "é parcialmente",
    "região",
    "funciona",
    "confira",
    ":",
    "desvio",
    "até",
    "e evento",
    "o número",
    "serão ",
    "queda ",
    "o capotamento",
    "vtr",
    "encontrado ",
    "acompanhe",
    "retificando",
    "acesso ",
    "cachorro encontra",
    "justificando",
    "sob elevado",
    "\?",
    "árvore caiu",
    "muito ",
    "duas ",
    "todos os ",
    "detidos "
]

sequential_numbers = list(range(1, 21))
for number in sequential_numbers:
    address_pos_patterns.append("{} vítms".format(number))
    address_pos_patterns.append("{} vít".format(number))
    address_pos_patterns.append("{}vítm".format(number))
    address_pos_patterns.append("{} vit".format(number))
    address_pos_patterns.append("{}vtm".format(number))
    address_pos_patterns.append("{} vtm".format(number))
    address_pos_patterns.append("{} vtr".format(number))
    address_pos_patterns.append("{}vtr".format(number))
    address_pos_patterns.append("{} viatura".format(number))
    address_pos_patterns.append("{} vtrs".format(number))

    address_pos_patterns.append("{num:02d} vítms".format(num=number))
    address_pos_patterns.append("{num:02d}vítm".format(num=number))
    address_pos_patterns.append("{num:02d} vít".format(num=number))
    address_pos_patterns.append("{num:02d} vit".format(num=number))
    address_pos_patterns.append("{num:02d}vtm".format(num=number))
    address_pos_patterns.append("{num:02d} vtm".format(num=number))
    address_pos_patterns.append("{num:02d} vtr".format(num=number))
    address_pos_patterns.append("{num:02d}vtr".format(num=number))
    address_pos_patterns.append("{num:02d} viatura".format(num=number))
    address_pos_patterns.append("{num:02d} vtrs".format(num=number))


address_anti_patterns = [
    "viaduto ligando",
    "ruas em ",
    "ruas e avenidas",
    "túnel entre ",
    "ponte devem",
    "ponte que ",
    "viaduto de ",
    "viaduto no ",
    "rua e ",
    "rua começou!",
    "r n o g o o g l e",
    "r i o s",
    "r e c i s a",
    "marginal segura",
    "praça é ",
    "rua mais animado",
    "ponte do…",
    "rua é ",
    "rua em ",
    "rua se lem",
    "rua ministram",
    "viaduto do chá no centro da cidade?",
    "ponte de embarque",
    "rua lixo",
    "avenida que ",
    "praça inativo",
    "rua verde\"",
    "rua p reativar",
    "largo são francisco terão",
    "ainda existe conheça sua história",
    "ponte também é",
    "é uma das mais famosas ",
    "uma nova ponte",
    "avenida é o lugar ideal",
    "rua ou ",
    "av __",
    "av citada",
    "rua acontece amanhã",
    "ponte sobre…",
    "rua na z",
    "rua já começou na cidade",
    "rua completa ",
    "rua do concorra a inscrições!",
    "rua sim!",
    "rua cidadã",
    "praça confira mais",
    "rua leva atrações",
    "rua verde\"",
    "rua consagrados",
    "rua ministram",
    "rua e bandas",
    "rua do concorra",
    "viaduto ligando",
    "túnel entre",
    "ponte que liga",
    "viaduto de retorno",
    "viaduto no km",
    "túnel inaugura",
    "viaduto ações",
    "túnel na",
    "rua precisando",
    "rua saindo",
    "rua esquentam",
    "rua vai agitar",
    "avenida que é",
    "ponte quanto",
    "ponte é ",
    "rua numa noite",
    "rua se lembra ",
    "intenerário",
    "linha ",
    "inaugura ponte ",
    " x ponte preta",
    "lixo sob viaduto"
]

google_geolocation_url = "https://maps.googleapis.com/maps/api/geocode/json?address={}&bounds={}&region={}&" \
                         "location_type={}&key={}"
google_geolocation_bounds = "-23.6815315,-46.8754817|-23.5505199,-46.6333094"
google_geolocation_region = "br"
google_geolocation_location_type = "GEOMETRIC_CENTER"
model_classes = ['Irrelevant', 'Social Event', 'Urban Event', 'Accident', 'Natural Disaster']
input_headers = ["_id", "address", "dateTime", "lat", "lng", "text", "label", "class_label"]