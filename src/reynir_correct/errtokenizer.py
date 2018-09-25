"""

    Reynir: Natural language processing for Icelandic

    Error-correcting tokenization layer

    Copyright (C) 2018 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.


    This module adds layers to the bintokenizer.py module in ReynirPackage.
    These layers add token-level error corrections and recommendation flags
    to the token stream.

"""

from reynir import TOK
from reynir.bintokenizer import DefaultPipeline


# Set of word forms that are allowed to appear more than once in a row
ALLOWED_MULTIPLES = frozenset(
    [
        "af",
        "auður",
        "að",
        "bannið",
        "bara",
        "bæði",
        "efni",
        "eftir",
        "eftir ",
        "eigi",
        "eigum",
        "eins",
        "ekki",
        "er",
        "falla",
        "fallið",
        "ferð",
        "festi",
        "flokkar",
        "flæði",
        "formið",
        "fram",
        "framan",
        "frá",
        "fylgi",
        "fyrir",
        "fyrir ",
        "fá",
        "gegn",
        "gerði",
        "getum",
        "hafa",
        "hafi",
        "hafið",
        "haft",
        "halla",
        "heim",
        "hekla",
        "heldur",
        "helga",
        "helgi",
        "hita",
        "hjá",
        "hjólum",
        "hlaupið",
        "hrætt",
        "hvort",
        "hæli",
        "inn ",
        "inni",
        "kanna",
        "kaupa",
        "kemba",
        "kira",
        "koma",
        "kæra",
        "lagi",
        "lagið",
        "leik",
        "leikur",
        "leið",
        "liðið",
        "lækna",
        "lögum",
        "löngu",
        "manni",
        "með",
        "milli",
        "minnst",
        "mun",
        "myndir",
        "málið",
        "móti",
        "mörkum",
        "neðan",
        "niðri",
        "niður",
        "niður ",
        "næst",
        "ofan",
        "opnir",
        "orðin",
        "rennur",
        "reynir",
        "riðlar",
        "riðli",
        "ráðum",
        "rétt",
        "safnið",
        "sem",
        "sett",
        "skipið",
        "skráðir",
        "spenna",
        "standa",
        "stofna",
        "streymi",
        "strokið",
        "stundum",
        "svala",
        "sæti",
        "sé",
        "sér",
        "síðan",
        "sótt",
        "sýna",
        "talið",
        "til",
        "tíma",
        "um",
        "undan",
        "undir",
        "upp",
        "valda",
        "vanda",
        "var",
        "vega",
        "veikir",
        "vel",
        "velta",
        "vera",
        "verið",
        "vernda",
        "verða",
        "verði",
        "verður",
        "veður",
        "vikum",
        "við",
        "væri",
        "yfir",
        "yrði",
        "á",
        "átta",
        "í",
        "ó",
        "ómar",
        "úr",
        "út",
        "úti",
        "þegar",
        "þjóna",
    ]
)

# Words incorrectly written as one word
NOT_COMPOUNDS = {
    "afhverju": ("af", "hverju"),
    "aftanfrá": ("aftan", "frá"),
    "afturábak": ("aftur", "á", "bak"),
    "afturí": ("aftur", "í"),
    "afturúr": ("aftur", "úr"),
    "afþví": ("af", "því"),
    "afþvíað": ("af", "því", "að"),
    "allajafna": ("alla", "jafna"),
    "allajafnan": ("alla", "jafnan"),
    "allrabest": ("allra", "best"),
    "allrafyrst": ("allra", "fyrst"),
    "allsekki": ("alls", "ekki"),
    "allskonar": ("alls", "konar"),
    "allskostar": ("alls", "kostar"),
    "allskyns": ("alls", "kyns"),
    "allsstaðar": ("alls", "staðar"),
    "allstaðar": ("alls", "staðar"),
    "alltsaman": ("allt", "saman"),
    "alltíeinu": ("allt", "í", "einu"),
    "alskonar": ("alls", "konar"),
    "alskyns": ("alls", "kyns"),
    "alstaðar": ("alls", "staðar"),
    "annarhver": ("annar", "hver"),
    "annarhvor": ("annar", "hvor"),
    "annarskonar": ("annars", "konar"),
    "annarslags": ("annars", "lags"),
    "annarsstaðar": ("annars", "staðar"),
    "annarstaðar": ("annars", "staðar"),
    "annarsvegar": ("annars", "vegar"),
    "annartveggja": ("annar", "tveggja"),
    "annaðslagið": ("annað", "slagið"),
    "austanfrá": ("austan", "frá"),
    "austanmegin": ("austan", "megin"),
    "austantil": ("austan", "til"),
    "austureftir": ("austur", "eftir"),
    "austurfrá": ("austur", "frá"),
    "austurfyrir": ("austur", "fyrir"),
    "bakatil": ("baka", "til"),
    "báðumegin": ("báðum", "megin"),
    "eftirað": ("eftir", "að"),
    "eftirá": ("eftir", "á"),
    "einhverjusinni": ("einhverju", "sinni"),
    "einhverntíma": ("einhvern", "tíma"),
    "einhverntímann": ("einhvern", "tímann"),
    "einhvernveginn": ("einhvern", "veginn"),
    "einhverskonar": ("einhvers", "konar"),
    "einhversstaðar": ("einhvers", "staðar"),
    "einhverstaðar": ("einhvers", "staðar"),
    "einskisvirði": ("einskis", "virði"),
    "einskonar": ("eins", "konar"),
    "einsog": ("eins", "og"),
    "einusinni": ("einu", "sinni"),
    "eittsinn": ("eitt", "sinn"),
    "endaþótt": ("enda", "þótt"),
    "enganveginn": ("engan", "veginn"),
    "ennfrekar": ("enn", "frekar"),
    "ennfremur": ("enn", "fremur"),
    "ennþá": ("enn", "þá"),
    "fimmhundruð": ("fimm", "hundruð"),
    "fimmtuhlutar": ("fimmtu", "hlutar"),
    "fjórðuhlutar": ("fjórðu", "hlutar"),
    "fjögurhundruð": ("fjögur", "hundruð"),
    "framaf": ("fram", "af"),
    "framanaf": ("framan", "af"),
    "frameftir": ("fram", "eftir"),
    "framhjá": ("fram", "hjá"),
    "frammí": ("frammi", "í"),
    "framundan": ("fram", "undan"),
    "framundir": ("fram", "undir"),
    "framvið": ("fram", "við"),
    "framyfir": ("fram", "yfir"),
    "framá": ("fram", "á"),
    "framávið": ("fram", "á", "við"),
    "framúr": ("fram", "úr"),
    "fulltaf": ("fullt", "af"),
    "fyrirfram": ("fyrir", "fram"),
    "fyrren": ("fyrr", "en"),
    "fyrripartur": ("fyrr", "partur"),
    "heilshugar": ("heils", "hugar"),
    "helduren": ("heldur", "en"),
    "hinsvegar": ("hins", "vegar"),
    "hinumegin": ("hinum", "megin"),
    "hvarsem": ("hvar", "sem"),
    "hvaðaner": ("hvaðan", "er"),
    "hvaðansem": ("hvaðan", "sem"),
    "hvaðeina": ("hvað", "eina"),
    "hverjusinni": ("hverju", "sinni"),
    "hverskonar": ("hvers", "konar"),
    "hverskyns": ("hvers", "kyns"),
    "hversvegna": ("hvers", "vegna"),
    "hvertsem": ("hvert", "sem"),
    "hvortannað": ("hvort", "annað"),
    "hvorteðer": ("hvort", "eð", "er"),
    "hvortveggja": ("hvort", "tveggja"),
    "héreftir": ("hér", "eftir"),
    "hérmeð": ("hér", "með"),
    "hérnamegin": ("hérna", "megin"),
    "hérumbil": ("hér", "um", "bil"),
    "innanfrá": ("innan", "frá"),
    "innanum": ("innan", "um"),
    "inneftir": ("inn", "eftir"),
    "innivið": ("inni", "við"),
    "innvið": ("inn", "við"),
    "inná": ("inn", "á"),
    "innávið": ("inn", "á", "við"),
    "inní": ("inn", "í"),
    "innúr": ("inn", "úr"),
    "lítilsháttar": ("lítils", "háttar"),
    "margskonar": ("margs", "konar"),
    "margskyns": ("margs", "kyns"),
    "meirasegja": ("meira", "að", "segja"),
    "meiraðsegja": ("meira", "að", "segja"),
    "meiriháttar": ("meiri", "háttar"),
    "meðþvíað": ("með", "því", "að"),
    "mikilsháttar": ("mikils", "háttar"),
    "minniháttar": ("minni", "háttar"),
    "minnstakosti": ("minnsta", "kosti"),
    "mörghundruð": ("mörg", "hundruð"),
    "neinsstaðar": ("neins", "staðar"),
    "neinstaðar": ("neins", "staðar"),
    "niðreftir": ("niður", "eftir"),
    "niðrá": ("niður", "á"),
    "niðrí": ("niður", "á"),
    "niðureftir": ("niður", "eftir"),
    "niðurfrá": ("niður", "frá"),
    "niðurfyrir": ("niður", "fyrir"),
    "niðurá": ("niður", "á"),
    "niðurávið": ("niður", "á", "við"),
    "nokkrusinni": ("nokkru", "sinni"),
    "nokkurntíma": ("nokkurn", "tíma"),
    "nokkurntímann": ("nokkurn", "tímann"),
    "nokkurnveginn": ("nokkurn", "veginn"),
    "nokkurskonar": ("nokkurs", "konar"),
    "nokkursstaðar": ("nokkurs", "staðar"),
    "nokkurstaðar": ("nokkurs", "staðar"),
    "norðanfrá": ("norðan", "frá"),
    "norðanmegin": ("norðan", "megin"),
    "norðantil": ("norðan", "til"),
    "norðaustantil": ("norðaustan", "til"),
    "norðureftir": ("norður", "eftir"),
    "norðurfrá": ("norður", "frá"),
    "norðurúr": ("norður", "úr"),
    "norðvestantil": ("norðvestan", "til"),
    "norðvesturtil": ("norðvestur", "til"),
    "níuhundruð": ("níu", "hundruð"),
    "núþegar": ("nú", "þegar"),
    "ofanaf": ("ofan", "af"),
    "ofaná": ("ofan", "á"),
    "ofaní": ("ofan", "í"),
    "ofanúr": ("ofan", "úr"),
    "oní": ("ofan", "í"),
    "réttumegin": ("réttum", "megin"),
    "réttummegin": ("réttum", "megin"),
    "samskonar": ("sams", "konar"),
    "seinnipartur": ("seinni", "partur"),
    "semsagt": ("sem", "sagt"),
    "sexhundruð": ("sex", "hundruð"),
    "sigrihrósandi": ("sigri", "hrósandi"),
    "sjöhundruð": ("sjö", "hundruð"),
    "sjöttuhlutar": ("sjöttu", "hlutar"),
    "smámsaman": ("smám", "saman"),
    "sumsstaðar": ("sums", "staðar"),
    "sumstaðar": ("sums", "staðar"),
    "sunnanað": ("sunnan", "að"),
    "sunnanmegin": ("sunnan", "megin"),
    "sunnantil": ("sunnan", "til"),
    "sunnanvið": ("sunnan", "við"),
    "suðaustantil": ("suðaustan", "til"),
    "suðuraf": ("suður", "af"),
    "suðureftir": ("suður", "eftir"),
    "suðurfrá": ("suður", "frá"),
    "suðurfyrir": ("suður", "fyrir"),
    "suðurí": ("suður", "í"),
    "suðvestantil": ("suðvestan", "til"),
    "svoað": ("svo", "að"),
    "svokallaður": ("svo", "kallaður"),
    "svosem": ("svo", "sem"),
    "svosemeins": ("svo", "sem", "eins"),
    "svotil": ("svo", "til"),
    "tilbaka": ("til", "baka"),
    "tilþessað": ("til", "þess", "að"),
    "tvennskonar": ("tvenns", "konar"),
    "tvöhundruð": ("tvö", "hundruð"),
    "tvöþúsund": ("tvö", "þúsund"),
    "umfram": ("um", "fram"),
    "undanúr": ("undan", "úr"),
    "undireins": ("undir", "eins"),
    "uppaf": ("upp", "af"),
    "uppað": ("upp", "að"),
    "uppeftir": ("upp", "eftir"),
    "uppfrá": ("upp", "frá"),
    "uppundir": ("upp", "undir"),
    "uppá": ("upp", "á"),
    "uppávið": ("upp", "á", "við"),
    "uppí": ("upp", "í"),
    "uppúr": ("upp", "úr"),
    "utanaf": ("utan", "af"),
    "utanað": ("utan", "að"),
    "utanfrá": ("utan", "frá"),
    "utanmeð": ("utan", "með"),
    "utanum": ("utan", "um"),
    "utanundir": ("utan", "undir"),
    "utanvið": ("utan", "við"),
    "utaná": ("utan", "á"),
    "vegnaþess": ("vegna", "þess"),
    "vestantil": ("vestan", "til"),
    "vestureftir": ("vestur", "eftir"),
    "vesturyfir": ("vestur", "yfir"),
    "vesturúr": ("vestur", "úr"),
    "vitlausumegin": ("vitlausum", "megin"),
    "viðkemur": ("við", "kemur"),
    "viðkom": ("við", "kom"),
    "viðkæmi": ("við", "kæmi"),
    "viðkæmum": ("við", "kæmum"),
    "víðsfjarri": ("víðs", "fjarri"),
    "víðsvegar": ("víðs", "vegar"),
    "yfirum": ("yfir", "um"),
    "ámeðal": ("á", "meðal"),
    "ámilli": ("á", "milli"),
    "áttahundruð": ("átta", "hundruð"),
    "áðuren": ("áður", "en"),
    "öðruhverju": ("öðru", "hverju"),
    "öðruhvoru": ("öðru", "hvoru"),
    "öðrumegin": ("öðrum", "megin"),
    "úrþvíað": ("úr", "því", "að"),
    "útaf": ("út", "af"),
    "útfrá": ("út", "frá"),
    "útfyrir": ("út", "fyrir"),
    "útifyrir": ("út", "fyrir"),
    "útivið": ("út", "við"),
    "útundan": ("út", "undan"),
    "útvið": ("út", "við"),
    "útá": ("út", "á"),
    "útávið": ("út", "á", "við"),
    "útí": ("út", "í"),
    "útúr": ("út", "úr"),
    "ýmiskonar": ("ýmiss", "konar"),
    "ýmisskonar": ("ýmiss", "konar"),
    "þangaðsem": ("þangað", "sem"),
    "þarafleiðandi": ("þar", "af", "leiðandi"),
    "þaraðauki": ("þar", "að", "auki"),
    "þareð": ("þar", "eð"),
    "þarmeð": ("þar", "með"),
    "þarsem": ("þar", "sem"),
    "þarsíðasta": ("þar", "síðasta"),
    "þarsíðustu": ("þar", "síðustu"),
    "þartilgerður": ("þar", "til", "gerður"),
    "þeimegin": ("þeim", "megin"),
    "þeimmegin": ("þeim", "megin"),
    "þessháttar": ("þess", "háttar"),
    "þesskonar": ("þess", "konar"),
    "þesskyns": ("þess", "kyns"),
    "þessvegna": ("þess", "vegna"),
    "þriðjuhlutar": ("þriðju", "hlutar"),
    "þrjúhundruð": ("þrjú", "hundruð"),
    "þrjúþúsund": ("þrjú", "þúsund"),
    "þvíað": ("því", "að"),
    "þvínæst": ("því", "næst"),
    "þínmegin": ("þín", "megin"),
    "þóað": ("þó", "að"),
}

SPLIT_COMPOUNDS = {
    ("afbragðs", "fagur"): "afbragðsfagur",
    ("afbragðs", "góður"): "afbragðsgóður",
    ("afbragðs", "maður"): "afbragðsmaður",
    ("afburða", "árangur"): "afburðaárangur",
    ("aftaka", "veður"): "aftakaveður",
    ("al", "góður"): "algóður",
    ("all", "góður"): "allgóður",
    ("allsherjar", "atkvæðagreiðsla"): "allsherjaratkvæðagreiðsla",
    ("allsherjar", "breyting"): "allsherjarbreyting",
    ("allsherjar", "neyðarútkall"): "allsherjarneyðarútkall",
    ("and", "stæðingur"): "andstæðingur",
    ("auka", "herbergi"): "aukaherbergi",
    ("auð", "sveipur"): "auðsveipur",
    ("aðal", "inngangur"): "aðalinngangur",
    ("aðaldyra", "megin"): "aðaldyramegin",
    ("bakborðs", "megin"): "bakborðsmegin",
    ("bakdyra", "megin"): "bakdyramegin",
    ("blæja", "logn"): "blæjalogn",
    ("brekku", "megin"): "brekkumegin",
    ("bílstjóra", "megin"): "bílstjóramegin",
    ("einskis", "verður"): "einskisverður",
    ("endur", "úthluta"): "endurúthluta",
    ("farþega", "megin"): "farþegamegin",
    ("fjölda", "margir"): "fjöldamargir",
    ("for", "maður"): "formaður",
    ("forkunnar", "fagir"): "forkunnarfagur",
    ("frum", "stæður"): "frumstæður",
    ("full", "mikill"): "fullmikill",
    ("furðu", "góður"): "furðugóður",
    ("gagn", "stæður"): "gagnstæður",
    ("gegn", "drepa"): "gegndrepa",
    ("ger", "breyta"): "gerbreyta",
    ("gjalda", "megin"): "gjaldamegin",
    ("gjör", "breyta"): "gjörbreyta",
    ("heildar", "staða"): "heildarstaða",
    ("hlé", "megin"): "hlémegin",
    ("hálf", "undarlegur"): "hálfundarlegur",
    ("hálfs", "mánaðarlega"): "hálfsmánaðarlega",
    ("hálftíma", "gangur"): "hálftímagangur",
    ("innvortis", "blæðing"): "innvortisblæðing",
    ("jafn", "framt"): "jafnframt",
    ("jafn", "lyndur"): "jafnlyndur",
    ("jafn", "vægi"): "jafnvægi",
    ("karla", "megin"): "karlamegin",
    ("klukkustundar", "frestur"): "klukkustundarfrestur",
    ("kring", "um"): "kringum",
    ("kvenna", "megin"): "kvennamegin",
    ("lang", "stærstur"): "langstærstur",
    ("langtíma", "aukaverkun"): "langtímaaukaverkun",
    ("langtíma", "lán"): "langtímalán",
    ("langtíma", "markmið"): "langtímamarkmið",
    ("langtíma", "skuld"): "langtímaskuld",
    ("langtíma", "sparnaður"): "langtímasparnaður",
    ("langtíma", "spá"): "langtímaspá",
    ("langtíma", "stefnumörkun"): "langtímastefnumörkun",
    ("langtíma", "þróun"): "langtímaþróun",
    ("lágmarks", "aldur"): "lágmarksaldur",
    ("lágmarks", "fjöldi"): "lágmarksfjöldi",
    ("lágmarks", "gjald"): "lágmarksgjald",
    ("lágmarks", "kurteisi"): "lágmarkskurteisi",
    ("lágmarks", "menntun"): "lágmarksmenntun",
    ("lágmarks", "stærð"): "lágmarksstærð",
    ("lágmarks", "áhætta"): "lágmarksáhætta",
    ("lítils", "verður"): "lítilsverður",
    ("marg", "oft"): "margoft",
    ("megin", "atriði"): "meginatriði",
    ("megin", "forsenda"): "meginforsenda",
    ("megin", "land"): "meginland",
    ("megin", "markmið"): "meginmarkmið",
    ("megin", "orsök"): "meginorsök",
    ("megin", "regla"): "meginregla",
    ("megin", "tilgangur"): "megintilgangur",
    ("megin", "uppistaða"): "meginuppistaða ",
    ("megin", "viðfangsefni"): "meginviðfangsefni",
    ("megin", "ágreiningur"): "meginágreiningur",
    ("megin", "ákvörðun"): "meginákvörðun",
    ("megin", "áveitukerfi"): "megináveitukerfi",
    ("mest", "allt"): "mestallt",
    ("mest", "allur"): "mestallur",
    ("meðal", "aðgengi"): "meðalaðgengi",
    ("meðal", "biðtími"): "meðalbiðtími",
    ("meðal", "ævilengd"): "meðalævilengd",
    ("mis", "bjóða"): "misbjóða",
    ("mis", "breiður"): "misbreiður",
    ("mis", "heppnaður"): "misheppnaður",
    ("mis", "lengi"): "mislengi",
    ("mis", "mikið"): "mismikið",
    ("mis", "stíga"): "misstíga",
    ("miðlungs", "beiskja"): "miðlungsbeiskja",
    ("myndar", "drengur"): "myndardrengur",
    ("næst", "bestur"): "næstbestur",
    ("næst", "komandi"): "næstkomandi",
    ("næst", "síðastur"): "næstsíðastur",
    ("næst", "verstur"): "næstverstur",
    ("sam", "skeyti"): "samskeyti",
    ("saman", "stendur"): "samanstendur",
    ("sjávar", "megin"): "sjávarmegin",
    ("skammtíma", "skuld"): "skammtímaskuld",
    ("skammtíma", "vistun"): "skammtímavistun",
    ("svo", "kallaður"): "svokallaður",
    ("sér", "framboð"): "sérframboð",
    ("sér", "herbergi"): "sérherbergi",
    ("sér", "inngangur"): "sérinngangur",
    ("sér", "kennari"): "sérkennari",
    ("sér", "staða"): "sérstaða",
    ("sér", "stæði"): "sérstæði",
    ("sér", "vitringur"): "sérvitringur",
    ("sér", "íslenskur"): "séríslenskur",
    ("sér", "þekking"): "sérþekking",
    ("sér", "þvottahús"): "sérþvottahús",
    ("sí", "felldur"): "sífelldur",
    ("sólar", "megin"): "sólarmegin",
    ("tor", "læs"): "torlæs",
    ("undra", "góður"): "undragóður",
    ("uppáhalds", "bragðtegund"): "uppáhaldsbragðtegund",
    ("uppáhalds", "fag"): "uppáhaldsfag",
    ("van", "megnugur"): "vanmegnugur",
    ("van", "virða"): "vanvirða",
    ("vel", "ferð"): "velferð",
    ("vel", "kominn"): "velkominn",
    ("vel", "megun"): "velmegun",
    ("vel", "vild"): "velvild",
    ("ágætis", "maður"): "ágætismaður",
    ("áratuga", "reynsla"): "áratugareynsla",
    ("áratuga", "skeið"): "áratugaskeið",
    ("óhemju", "illa"): "óhemjuilla",
    ("óhemju", "vandaður"): "óhemjuvandaður",
    ("óskapa", "hiti"): "óskapahiti",
    ("óvenju", "góður"): "óvenjugóður",
    ("önd", "verður"): "öndverður",
    ("ör", "magna"): "örmagna",
    ("úrvals", "hveiti"): "úrvalshveiti",
    # Split into 3 words
    # ("heils", "dags", "starf") : "heilsdagsstarf",
    # ("heils", "árs", "vegur") : "heilsársvegur",
    # ("hálfs", "dags", "starf") : "hálfsdagsstarf",
    # ("marg", "um", "talaður") : "margumtalaður",
    # ("sama", "sem", "merki") : "samasemmerki",
    # ("því", "um", "líkt") : "þvíumlíkt",
}


class CorrectToken:

    """ This class sneakily replaces the tokenizer.Tok tuple in the tokenization
        pipeline. When applying a CorrectionPipeline (instead of a DefaultPipeline,
        as defined in binparser.py in ReynirPackage), tokens get translated to
        instances of this class in the correct() phase. This works due to Python's
        duck typing, because a CorrectToken class instance is able to walk and quack
        - i.e. behave - like a tokenizer.Tok tuple. It adds an _err attribute to hold
        information about spelling and grammar errors, and some higher level functions
        to aid in error reporting and correction. """

    def __init__(self, kind, txt, val):
        self.kind = kind
        self.txt = txt
        self.val = val
        self._err = None

    def __getitem__(self, index):
        """ Support tuple-style indexing, as raw tokens do """
        return (self.kind, self.txt, self.val)[index]

    @classmethod
    def from_token(cls, token):
        """ Wrap a raw token in a CorrectToken """
        return cls(token.kind, token.txt, token.val)

    @classmethod
    def word(cls, txt, val=None):
        """ Create a wrapped word token """
        return cls(TOK.WORD, txt, val)

    def __repr__(self):
        return (
            "<CorrectToken(kind: {0}, txt: '{1}', val: {2})>"
            .format(TOK.descr[self.kind], self.txt, self.val)
        )

    __str__ = __repr__

    def set_error(self, err):
        """ Associate an Error class instance with this token """
        self._err = err

    def copy_error(self, other):
        """ Copy the error field from another CorrectToken instance """
        if isinstance(other, CorrectToken):
            self._err = other._err

    @property
    def error_description(self):
        """ Return the description of an error associated with this token, if any """
        return "" if self._err is None else self._err.description


class Error:

    """ Base class for spelling and grammar errors, warnings and recommendations.
        An Error has a code and can provide a description of itself. """

    def __init__(self, code):
        self._code = code

    @property
    def code(self):
        return self._code
    
    @property
    def description(self):
        """ Should be overridden """
        raise NotImplementedError


class CompoundError(Error):

    """ A CompoundError is an error where words are duplicated, split or not
        split correctly. """

    def __init__(self, code, txt):
        # Compound error codes start with "C"
        super().__init__("C" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


class UnknownWordError(Error):

    """ An UnknownWordError is an error where the given word form does not
        exist in BÍN or additional vocabularies, and cannot be explained as
        a compound word. """

    def __init__(self, code, txt):
        # Unknown word error codes start with "U"
        super().__init__("U" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


def parse_errors(token_stream):

    """ This tokenization phase is done before BÍN annotation
        and before static phrases are identified. It finds duplicated words,
        and words that have been incorrectly split or should be split. """

    def get():
        """ Get the next token in the underlying stream and wrap it
            in a CorrectToken instance """
        return CorrectToken.from_token(next(token_stream))

    token = None
    try:
        # Maintain a one-token lookahead
        token = get()
        while True:
            next_token = get()
            # Make the lookahead checks we're interested in

            # Word duplication; GrammCorr 1B
            if (
                token.txt
                and next_token.txt
                and token.txt.lower() == next_token.txt.lower()
                and token.txt.lower() not in ALLOWED_MULTIPLES
                and token.kind == TOK.WORD
            ):
                # Step to next token
                next_token = CorrectToken.word(token.txt)
                next_token.set_error(
                    CompoundError(
                        "001", "Endurtekið orð ('{0}') var fellt burt"
                        .format(token.txt)
                    )
                )
                token = next_token
                continue

            # Splitting wrongly compounded words; GrammCorr 1A
            if token.txt and token.txt.lower() in NOT_COMPOUNDS:
                for phrase_part in NOT_COMPOUNDS[token.txt.lower()]:
                    new_token = CorrectToken.word(phrase_part)
                    new_token.set_error(
                        CompoundError(
                            "002", "Orðinu '{0}' var skipt upp"
                            .format(token.txt)
                        )
                    )
                    yield new_token
                token = next_token
                continue

            # Unite wrongly split compounds; GrammCorr 1X
            if (token.txt, next_token.txt) in SPLIT_COMPOUNDS:
                first_txt = token.txt
                token = CorrectToken.word(token.txt + next_token.txt)
                token.set_error(
                    CompoundError(
                        "003", "Orðin '{0} {1}' voru sameinuð í eitt"
                        .format(first_txt, next_token.txt)
                    )
                )
                continue

            # Yield the current token and advance to the lookahead
            yield token
            token = next_token

    except StopIteration:
        # Final token (previous lookahead)
        if token:
            yield token


def lookup_unknown_words(db, token_stream):
    """ Try to identify unknown words in the token stream, for instance
        as spelling errors (character juxtaposition, deletion, insertion...) """
    for token in token_stream:
        if token.kind == TOK.WORD and not token.val:
            # Mark the token as an unknown word
            token.set_error(
                UnknownWordError(
                    "001", "Óþekkt orð: '{0}'".format(token.txt)
                )
            )
        yield token


class CorrectionPipeline(DefaultPipeline):

    """ Override the default tokenization pipeline defined in binparser.py
        in ReynirPackage, adding a correction phase """

    def __init__(self, text, auto_uppercase=False):
        super().__init__(text, auto_uppercase)

    def word_token_ctor(self, txt, val=None, token=None):
        """ Use our own CorrectToken class for word token instances """
        ct = CorrectToken.word(txt, val)
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, which might have had an associated error:
            # make sure that it is preserved
            ct.copy_error(token)
        return ct

    def correct(self, stream):
        """ Add a correction pass just before BÍN annotation """
        return parse_errors(stream)

    def lookup_unknown_words(self, stream):
        """ Attempt to resolve unknown words """
        return lookup_unknown_words(self._db, stream)


def tokenize(text, auto_uppercase=False):
    """ Tokenize text using the correction pipeline, overriding a part
        of the default tokenization pipeline """
    pipeline = CorrectionPipeline(text, auto_uppercase)
    return pipeline.tokenize()
