from .pattern import PatternMatcher
from reynir.simpletree import SimpleTree
from .annotation import Annotation
from reynir_correct.errtokenizer import emulate_case


def add_extra_patterns(matcher: PatternMatcher) -> None:
    def ferli(match: SimpleTree) -> None:
        vp = match.first_match("VP > { 'koma' }")
        if vp is None:
            vp = match.first_match("VP >> { 'koma' }")
        pp = match.first_match("PP > 'í'")
        if pp is None:
            pp = match.first_match("PP >> 'í' ")
        # Find the attached nominal phrase
        np = match.first_match("NP > 'ferli' ")
        if np is None:
            np = match.first_match("NP >> 'ferli' ")
        if vp is None or pp is None or np is None:
            return
        pp_i = pp.first_match('"í"')
        if pp_i is None:
            return
        start, end = pp.span[0], pp.span[1]
        for ann in matcher._ann:
            if ann.code == "V002" and ann.start == start and ann.end == end:
                # We have already annotated the error, no need to do it twice
                return
        suggest = "leysa úr"
        text = "Betra er að tala um að 'leysa úr málunum' heldur en að 'koma í ferli'."
        matcher._ann.append(
            Annotation(
                start=start,
                end=end,
                code="V002",
                text=text,
                original="ferli",
                suggest=suggest,
                is_warning=True,
            )
        )

    def til_skoðunar(match: SimpleTree) -> None:
        pp = match.first_match("PP > 'til'")
        if pp is None:
            return
        start, end = pp.span[0], pp.span[1]
        for ann in matcher._ann:
            if ann.code == "V002" and ann.start == start and ann.end == end:
                # We have already annotated the error, no need to do it twice
                return
        suggest = "leysa úr"
        text = "Betra er að tala um að 'leysa úr málunum' heldur en að 'taka til skoðunar'."
        matcher._ann.append(
            Annotation(
                start=start,
                end=end,
                code="V002",
                text=text,
                original="skoðunar",
                suggest=suggest,
                is_warning=True,
            )
        )

    def starfsmenn(match: SimpleTree) -> None:
        # TODO: change pattern to need just one call to first_match
        c = match.first_match("NP >> { 'starfsmaður' }")
        if c is None:
            return
        start, end = c.span[0], c.span[1]
        for ann in matcher._ann:
            if ann.code == "V002" and ann.start == start and ann.end == end:
                # We have already annotated the error, no need to do it twice
                return
        original = "starfsmenn"
        suggest = "starfsfólk"
        text = f"Betra er að tala um {suggest} en {original}'."
        matcher._ann.append(
            Annotation(
                start=start,
                end=end,
                code="V002",
                text=text,
                original=original,
                suggest=emulate_case(suggest, template=text),
                is_warning=True,
            )
        )

    def greiðslubyrði(match: SimpleTree) -> None:
        c = match.first_match("NP >> { 'greiðslubyrði' }")
        if c is None:
            return
        start, end = c.span[0], c.span[1]
        for ann in matcher._ann:
            if ann.code == "V002" and ann.start == start and ann.end == end:
                # We have already annotated the error, no need to do it twice
                return
        original = "greiðslubyrði"
        suggest = "mánaðarlegar greiðslur"
        text = f"Betra er að tala um {suggest} en {original}'."
        matcher._ann.append(
            Annotation(
                start=start,
                end=end,
                code="V002",
                text=text,
                original=original,
                suggest=emulate_case(suggest, template=text),
                is_warning=True,
            )
        )

    # add the instance functions to the matcher object
    setattr(matcher, "ferli", ferli)
    setattr(matcher, "til_skoðunar", til_skoðunar)
    setattr(matcher, "starfsmenn", starfsmenn)
    setattr(matcher, "greiðslubyrði", greiðslubyrði)

    matcher.add_pattern(
        (
            "ferli",  # Trigger lemma for this pattern
            "IP > { VP > { VP > { 'koma' } PP > { P > { 'í' } NP > { 'ferli' } } } }",
            matcher.ferli,  # type: ignore
            None,
        )
    )

    matcher.add_pattern(
        (
            "skoðun",  # Trigger lemma for this pattern
            "PP > { 'til' NP > { 'skoðun' } }",
            matcher.til_skoðunar,  # type: ignore
            None,
        )
    )

    matcher.add_pattern(
        (
            "starfsmaður",  # Trigger lemma for this pattern
            " NP >> { 'starfsmaður' no_ft } ",
            matcher.starfsmenn,  # type: ignore
            None,
        )
    )

    matcher.add_pattern(
        (
            "greiðslubyrði",  # Trigger lemma for this pattern
            " NP >> { 'greiðslubyrði' } ",
            matcher.greiðslubyrði,  # type: ignore
            None,
        )
    )
