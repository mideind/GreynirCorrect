from typing import Tuple

import tokenizer

from reynir_correct import readability

hard_text1 = "Í Ritgerð í átt að nýrri kenningu um sjón (An essay towards a new theory of vision)2 sem kom út 1709 tekur Berkeley sérstaklega fram að þær lögunarhugmyndir sem skynjaðar eru með sjóninni séu aðrar en þær sem skynjaðar eru með snertiskyninu og að þessi tvö skynfæri eigi engar sameiginlegar hugmyndir; við sjáum ekki það sem við snertum, eða öfugt. Berkeley telur, líkt og Locke, að sjónskynið eitt og óstutt gefi okkur ekki þrívíddarhugmyndir þar sem dýptar- og fjarlægðarskyn komi ekki til nema með reynslu. Þeir sem hafa reynslu af því að sjá og snerta hluti hafa lært að tilteknar sjónrænar hugmyndir og tilteknar snertihugmyndir eiga það til að fylgjast að og þess vegna tengjum við þær saman. En maður sem skyndilega fær sjónina hefur ekki þessa reynslu til að byggja á og dregur því enga ályktun um snertihugmyndir á grundvelli þess sem fyrir augu hans ber. Hið neikvæða svar Berkeleys við spurningu Molyneux sem hann ræðir stuttlega síðar í þessu riti sínu um sjónina liggur því beint við."

hard_text2 = "Ábyrgðaraðila staðarins er skylt að tilkynna heilbrigðisnefnd um viðburði þar sem búast má við að hljóðstig verði hærra en tilgreint er í 1. mgr. með hæfilegum fyrirvara og ber hann kostnað við eftirlit á tónleikunum þ.m.t. hljóðmælingum skv. gjaldskrá. Heilbrigðisnefnd getur beint þeim tilmælum til ábyrgðaraðila staðarins að hann bjóði gestum eyrnatappa og hengi upp sérstök viðvörunarskilti um hátt hljóðstig á áberandi hátt, setji aldurstakmörk fyrir gesti, geri grein fyrir staðsetningu hátalara og öðru sem heilbrigðisnefnd telur þurfa til að koma í veg fyrir heilsuspillandi hávaða. Brot gegn ákvæðum 6.-10. gr. reglugerðar þessarar varða sektum hvort sem þau eru framin af ásetningi eða stórfelldu gáleysi. Sé um stórfelld eða ítrekuð ásetningsbrot að ræða skulu þau að auki varða fangelsi allt að fjórum árum. Sektir má ákvarða lögaðila þó að sök verði ekki sönnuð á fyrirsvarsmenn eða starfsmenn hans eða aðra þá einstaklinga sem í þágu hans starfa, enda hafi brotið orðið eða getað orðið til hagsbóta fyrir lögaðilann. Þó skal lögaðili ekki sæta refsingu ef um óhapp er að ræða. Einnig má, með sama skilorði, gera lögaðila sekt ef fyrirsvarsmenn eða starfsmenn hans eða aðrir einstaklingar sem í þágu hans starfa gerast sekir um brot."

medium_hard_text = "Íslandsbanki lítur misferli mjög alvarlegum augum. Við hvetjum þig til að láta vita ef þú hefur vitneskju eða grun um hugsanlegt misferli sem tengist starfsemi bankans á einhvern hátt. Þar með aðstoðar þú við að upplýsa um brot sem valdið geta viðskiptavinum, almenningi, bankanum og atvinnulífi miklu tjóni. Tilkynning um misferli þarf að byggja á rökstuddum grun, sem þarf þó ekki að vera hafinn yfir allan vafa. Móttaka misferlistilkynninga fer í gegnum kerfið WhistleB sem er sérstaklega aðgreint frá öðrum upplýsingakerfum bankans og vistað utan hans. Kerfið er sérstaklega hannað með tilliti til verndunar tilkynnanda og þeirra tilkynninga sem þangað berast, en veitir jafnframt möguleika á áframhaldandi samskiptum við þá sem kjósa að njóta nafnleyndar. Kerfið er mjög einfalt í notkun og leiðir tilkynnanda áfram. Til að tryggja nafnleynd er starfsmönnum bent á að skrá ekki tilkynningar í gegnum tölvubúnað sem tengist netkerfi bankans. Einungis misferlisteymi innri endurskoðunar bankans mun hafa aðgang að skráðum málum."

medium_text1 = "Verðbólgan er komin í 10,2%. Það er allt of mikið og hefur heilmikil áhrif á okkur öll. Í Korni Greiningar Íslandsbanka er greint frá ástæðum þessarar miklu verðbólgu og spáð fyrir um þróunina á næstunni. Slíkt efni má nálgast frá okkur reglulega með skráningu á póstlistann okkar. Hér langar okkur þó að stíga eitt skref til baka og rýna í þessar verðbólgutölur með öðrum hætti. Lítum á 5 áhugaverðar staðreyndir um þessa miklu verðbólgu í dag. Hvað hefur hækkað mest? Ef þér finnst allt vera að hækka ertu með puttann nokkuð vel á verðbólgupúlsinum. Verðhækkanirnar sem búa til þessa miklu verðbólgu koma úr ýmsum áttum og má sem dæmi nefna að matvörur hafa margar hverjar hækkað mikið í verði undanfarið ár. Þá hefur flutningsverð og kostnaður við ferðalög rokið upp sem og húsnæðisverð, þó svo húsnæðismarkaðurinn sé að kólna ansi hratt þessa dagana. Ef við lítum á nokkur dæmi um hækkanir vekur kannski athygli að loksins virðist gamli góði kálböggullinn vera orðinn lúxusvara. Eða þangað virðist hann í það minnsta stefna því kál hefur hækkað um heilt 31% í verði frá því á sama tíma í fyrra og munar um minna. Það er orðið talsvert dýrara að ferðast (og að verja tíma erlendis sömuleiðis) og loks þarf þessi væna flís af feitum sauð sem ég sá fyrir mér um helgina að vera aðeins minni því lambakjötið okkar er 19% dýrara en í fyrra."

medium_text2 = "Tónlistarmaðurinn Daði Freyr sendi frá sér sína fyrstu plötu í fullri lengd í ágúst. Fyrsta lag hans kom út fyrir sex árum og síðan hefur fylgt hellingur af lögum og þó nokkrar þröngskífur og alls konar stemmning eins og alþjóð veit. Það er kannski hálfskrítið að tónlistarmaður eins og Daði Freyr sé að senda frá sér sína fyrstu plötu í lok ágúst árið 2023 en það þýðir nú ekki að hann sé búinn að vera latur. Eins og fólk veit hefur hann verið að taka þátt í Söngvakeppni evrópskra sjónvarpsstöðva, skemmta í sömu keppni og svo var covid og alls konar dæmi sem tekur tíma. Nóg um það, nýja platan er tekin upp á heimili kappans í Berlín og smíði hennar hófst fyrir tveimur árum. Daði segir plötuna á persónulegum nótum, meðganga og fæðing auk heimsfaraldurs hafi litað sköpunina ásamt þessum venjulegu áhrifavöldum eins og tónlist sem hann hefur verið að hlusta á, vinum og stórfjölskyldu. Plata vikunnar að þessu sinni er I Made An Album, fyrsta plata Daða Freys, og er aðgengileg í heild sinni ásamt kynningum hans í spilara RÚV."

easy_text1 = "Þegar ég fæddist fékk ég stimpilinn „stelpa“. Ég valdi ekki þennan stimpil, enda hafði ég hvorki vit né skoðun á málinu þegar ég hlaut þennan titil. Ég ólst upp með þennan stimpil, en það er ýmislegt sem fylgir honum. Ég fékk alveg að vera ég sjálfur þannig, en ég vissi samt að ég ætti að haga mér og klæða mig á ákveðinn hátt. Það að eiga að vera stelpa truflaði mig ekki, ég var töffari („tomboy“) og gerði að mestu bara það sem ég vildi. Ég óskaði þess samt alltaf að hafa fæðst sem strákur, alveg frá því að ég var krakki og langt fram eftir aldri, fannst það passa mér betur, vera meira spennandi. Þegar ég varð unglingur fór ég oftar að reyna að passa í kassann, svo að segja. Ég reyndi m.a. að vera pæja, en það entist aldrei lengi, ég varð mikill femínisti og það hjálpaði mér að halda áfram að gera það sem ég vildi. Ég fór ekki að pæla í kyninu mínu af alvöru fyrr en ég var orðinn 19 ára. Ég horfði á myndband, og fattaði allt í einu að ég gæti verið trans, að ég mætti vera strákur, að það væri hægt. Það var eins og það hefði kviknað ljós í höfðinu á mér. Ég hugsaði um lítið annað næstu mánuðina. Í hvert skipti sem ég þurfti að kynja mig í hugsunum mínum, sem er rosalega oft ef þið hugsið út í það, fór ég að hugsa um kyn mitt: „Ég er svöng … svangur? … svangt? Hvað er ég?“ Ég sagði kærustunni minni eiginlega strax hvað ég var að hugsa og hún bauðst til að nota karlkyn um mig þegar við værum ein. Eftir eina viku var jafn skrítið að hún notaði karlkyn eins og að allir aðrir notuðu kvenkyn, eins og þau höfðu gert í meira en 19 ár."

easy_text2 = "Ernir eru stórir ránfuglar. Það er ein tegund af örnum á Íslandi, haförn. Haförn er næst-stærsta tegund af fuglum á Íslandi, á eftir álft eða svani. En örnum hefur ekki gengið vel á Íslandi. Á síðustu öld var hætta á að þeir myndu deyja út. Þá voru bara um 20 pör af örnum á Íslandi, eða um 40 ernir. Þá var Fuglaverndarfélagið stofnað til að hjálpa örnum. Félagið lét banna að eitra fyrir refum. Margir ernir dóu nefnilega út af refaeitri. Eftir það fór örnum að ganga betur. Þeim fjölgaði hægt og rólega. Núna eru um 90 pör af örnum á Íslandi, eða um 180 ernir. Fuglafræðingur segir að örnum hafi gengið vel að eignast unga á þessu ári. Um 45 pör eru núna með unga. Það hefur sjaldan gengið betur. Ef þetta heldur áfram munu ernir aftur dreifa sér um Ísland. Þá fara ernir aftur að sjást á fleiri stöðum. Núna eru ernir aðallega á Vesturlandi. Einu sinni voru þeir um allt land, nema á hálendinu."

numbers = "123456677, 2434,5, 5 23 233 , ! ???' % #$43234654 1 23 45667 7, 2434,5, 5 23 233 , ! ?  ?' % #$ 432 34 654 12345 6677, 243 4,5, 5 23 233 , ! ???' % #$43 23 4654 123456677, 2434,5, 5 23 233 , ! ???' % #$43234654 123456677. 2434,5, 5 23 233 , ! ??.?' % #$43.2 3465 4 12 345 6677, 2434,5, 5 23 23 3 , ! ???' % #$43 234 654 ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? 12345.6677, 2.434,5, 5 23 233 ., ! ???' % #$43234654 1 23 45667 7, 2434,5, 5 23 233 , ! ?  ?'. % #$ 432 34 654 12345 6677, 243 4,5, 5 23 233 , ! ???' % #$43 23 4654 123456677, 2434,5, 5 23 233 , ! ???' % #$43234.654 123456677, 2434,5, 5 23 233 , ! ???' % #$43.2 3465 4 12 345 6677, .2434,5, 5 23 23 3 , ! ???' % #$43 234 654 ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?"

single_syllables = "Ég á nú að fá ís í búð í dag. Þú ert frá þér af sorg af því þú drakkst ei nóg af bjór."

many_syllables = "Ævinlega þegar Elísabet vaknaði lagaði hún rótsterkan kaffibolla handa sjálfri sér og svefndrukknum eiginmanninum, honum Engilbert. Aukinheldur íklæddist hún föðurlandi og fjólublárri herðaslá."

many_numbers_and_such = "Hafðu samband við ráðgjafaver Íslandsbanka í síma 440 4000, sem er opið í síma frá klukkan 9-16 og netspjalli frá 9-17 alla virka daga. Utan opnunartíma er hægt að hafa samband við neyðarþjónustu Íslandsbanka í síma 440 4000 eða á islandsbanki.is. Árið 1944 þann 17. júní fengu Íslendingar sjálfstæði frá Dönum. 37°C er algengur líkamshiti. Fólk í Grænumýri 11 lenti í 4. lekanum í gær, 30. ágúst."


no_text = ""
# text and expected range:
text_dict = {
    easy_text1: (70, 90),
    easy_text2: (70, 100),
    medium_text2: (50, 70),
    medium_text1: (50, 70),
    medium_hard_text: (40, 60),
    hard_text1: (20, 50),
    hard_text2: (20, 50),
}

not_real_texts = {
    numbers: (100, 150),  # no text
    single_syllables: (90, 130),  # very improbable text
    many_syllables: (-10, 10),  # very improbable text
    many_numbers_and_such: (50, 100),
}


def get_sentence_word_syllable_counts(text: str) -> Tuple[int, int, int]:
    """Get the number of sentences, words and syllables in a text."""
    stream = tokenizer.tokenize(text)
    return readability.FleschKincaidScorer.get_counts_from_stream(stream)


def test_count_syllables():
    words_and_syllables = [
        ("hús", 1),
        ("húsið", 2),
        ("húsin", 2),
        ("húsinu", 3),
        ("ae_iouyáéíóúýöæ", 14),  # vowels
        ("eieyau", 3),  # diphtongs
        ("bcdfghjklmnpqrstvwxzðþ", 0),  # consonants
        ("áin þessi", 4),  # spaces
        ("", 0),
        (" ", 0),
        ("1999", 0),
    ]
    for word, num_syllables in words_and_syllables:
        found_syllables = readability.FleschKincaidScorer.count_syllables_in_word(word)
        assert (
            found_syllables == num_syllables
        ), f"Expected {num_syllables} syllables in {word} but got {found_syllables}"


def test_count_sentences_words_syllables():
    strings_and_words = [
        ("já", (1, 1, 1)),
        ("já já", (1, 2, 2)),
        ("já     já\tjá", (1, 3, 3)),
        ("Þetta. Eru. Fjórar. Setningar", (4, 4, 9)),
    ]
    for string, (num_sents, num_words, num_syllables) in strings_and_words:
        assert get_sentence_word_syllable_counts(string) == (
            num_sents,
            num_words,
            num_syllables,
        ), f"Expected {num_syllables} syllables, {num_words} words and {num_sents} sentences in {string}"


def test_flesch_score_range():
    for text in text_dict:
        assert readability.FleschKincaidScorer.get_score_from_text(text) > 0
        assert readability.FleschKincaidScorer.get_score_from_text(text) < 100

    assert readability.FleschKincaidScorer.get_score_from_text(
        easy_text1
    ) > readability.FleschKincaidScorer.get_score_from_text(
        medium_text1
    ), "Expected easy text to be easier than medium text"
    assert readability.FleschKincaidScorer.get_score_from_text(
        medium_text1
    ) > readability.FleschKincaidScorer.get_score_from_text(
        medium_hard_text
    ), "Expected medium text to be easier than medium hard text"
    assert readability.FleschKincaidScorer.get_score_from_text(
        medium_text2
    ) > readability.FleschKincaidScorer.get_score_from_text(
        medium_hard_text
    ), "Expected medium text to be easier than medium hard text"
    assert readability.FleschKincaidScorer.get_score_from_text(
        medium_hard_text
    ) > readability.FleschKincaidScorer.get_score_from_text(
        hard_text1
    ), "Expected medium hard text to be easier than hard text"


def test_flesch_score_against_estimate():
    for text in text_dict:
        flesch_score = readability.FleschKincaidScorer.get_score_from_text(text)
        feedback = readability.FleschKincaidScorer.get_feedback(flesch_score)

        assert (
            flesch_score > text_dict[text][0]
        ), f"Expected text to be easier than estimate: {text_dict[text][0]}, was {flesch_score}. Feedback: {feedback}. First sentence: {text.split('.')[0]}"

        assert (
            flesch_score < text_dict[text][1]
        ), f"Expected text to be harder than estimate: {text_dict[text][1]}, was {flesch_score}. Feedback: {feedback}. First sentence: {text.split('.')[0]}"

    for text in not_real_texts:
        flesch_score = readability.FleschKincaidScorer.get_score_from_text(text)
        feedback = readability.FleschKincaidScorer.get_feedback(flesch_score)

        assert (
            flesch_score > not_real_texts[text][0]
        ), f"Expected text to be easier than estimate: {not_real_texts[text][0]}, was {flesch_score}. Feedback: {feedback}. Feedback: {feedback}. First sentence: {text.split('.')[0]}"

        assert (
            flesch_score < not_real_texts[text][1]
        ), f"Expected text to be harder than estimate: {not_real_texts[text][1]}, was {flesch_score}. Feedback: {feedback}. Feedback: {feedback}. First sentence: {text.split('.')[0]}"
