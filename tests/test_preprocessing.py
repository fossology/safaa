# SPDX-FileCopyrightText: © 2026 Fossology contributors
# SPDX-License-Identifier: LGPL-2.1-only
"""The preprocessing rules, one behaviour per test. No model is loaded."""

import pytest

from safaa.Safaa import (
    CODE_LABEL_RE,
    COPYRIGHT_MARKER_RE,
    POINTER_RE,
    EMAIL_RE,
    LEGAL_FORM_TAIL_RE,
    PLACEHOLDER_HOLDER_RE,
    PLACEHOLDER_YEAR_RE,
    YEAR_RE,
    _fallback_holder_span,
    _names_someone,
    context_dependent,
    looks_binary,
)


@pytest.mark.parametrize("text", ["2019", "1995-2024", "(c) 2003, 2004 FSF"])
def test_years_are_dates(text):
    assert YEAR_RE.search(text)


@pytest.mark.parametrize("text", ["build 20231105", "v12345", "port 8080", "1850"])
def test_other_numbers_are_not_years(text):
    assert not YEAR_RE.search(text)


@pytest.mark.parametrize("text", ["jane.doe@siemens.com", "user+tag@sub.example.org", "kłoczko@pld.org.pl"])
def test_emails(text):
    assert EMAIL_RE.fullmatch(text)


@pytest.mark.parametrize("text", [
    "Copyright (C) <year> <name of author>", "Copyright (c) [year] [copyright holders]",
    "Copyright 2001 M. Y. Name", "Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER",
    "copyright (c) [XXXX-]YYYY, <OWNER>", "Copyright (C) @var{year} @var{your name}",
])
def test_placeholder_holders(text):
    assert PLACEHOLDER_HOLDER_RE.search(text)


@pytest.mark.parametrize("text", ["Copyright 2019 Jane Doe", "Copyright Holder: Apache Software Foundation"])
def test_real_holders_are_not_placeholders(text):
    assert not PLACEHOLDER_HOLDER_RE.search(text)


@pytest.mark.parametrize("text", ["YEAR", "<year>", "[YEAR]", "{{ year }}", "$YEAR", "yyyy", "@var{year}"])
def test_placeholder_years(text):
    assert PLACEHOLDER_YEAR_RE.search(text)


@pytest.mark.parametrize("text", ["copyright year list to include the current year.", "copyright years (90, 1991)"])
def test_the_word_year_in_prose_is_not_a_placeholder(text):
    assert not PLACEHOLDER_YEAR_RE.search(text)


@pytest.mark.parametrize("text,expected", [
    ("Copyright (C) YEAR Free Software Foundation, Inc.", True),   # real holder, placeholder year
    ("Copyright 1995-$YEAR The OpenSSL Project Authors", True),
    ("Copyright (c) {{ year }} Inria", True),
    ("Copyright (C) <year> <name of author>", False),              # template: placeholder holder
    ("Copyright (c) 2019 Jane Doe", False),
])
def test_context_dependent(text, expected):
    assert context_dependent(text) is expected


@pytest.mark.parametrize("text", [
    "Copyright ¡ 19q8 Unicodlfmlym aypostmly\x06y mÿaml;ym\x06ramlym\x0eym",
    "Copyright`&c) 2018, ÿÿÿÿÿÿÿ", "\ufffd\ufffd\tO*Fb@\ufffd\ufffdmox\ufffd\ufffd\ufffd\ufffdT",
    "© eS3n\x98WH\x0fd\x82\x13/þó\x05l¡;T",
])
def test_binary_text(text):
    assert looks_binary(text)


@pytest.mark.parametrize("text", [
    "Copyright 2007 - 2013, Nicolas François", "Copyright (C) 2020 华为技术有限公司", "© 2019 Kłoczko Tomasz",
    "Copyright (c) 2014 Cenk Alt\ufffd\ufffd", "Copyright \ufffd\ufffd 2015 Docker, Inc.",  # decoding damage
])
def test_text_with_accents_or_non_latin_is_not_binary(text):
    assert not looks_binary(text)


@pytest.mark.parametrize("text,holder", [
    ("Copyright (C) HAW Hamburg", "HAW Hamburg"),
    ("Copyright 2025 u-blox AG", "u-blox AG"),
    ("Copyright Canonical Ltd", "Canonical Ltd"),
])
def test_fallback_holder(text, holder):
    span = _fallback_holder_span(text)
    assert span and text[span[0]:span[1]] == holder


@pytest.mark.parametrize("text", [
    "Copyright Treaty of 1996", "Copyright Status", "Copyright Notice", "COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE",
    "copyright notice, this list of conditions", "Copyright Act of 1998",
])
def test_no_fallback_holder_in_prose(text):
    assert _fallback_holder_span(text) is None


@pytest.mark.parametrize("tail", [" A/S", " ASA", " GmbH", ", Inc.", " Ltd.", " e.V.", " B.V.", " LLC"])
def test_legal_form_tail(tail):
    assert LEGAL_FORM_TAIL_RE.match(tail)


@pytest.mark.parametrize("tail", [" Assembly", " Agents", " Incredible"])
def test_words_starting_like_legal_forms_are_not_legal_forms(tail):
    assert not LEGAL_FORM_TAIL_RE.match(tail)


@pytest.mark.parametrize("text", ["Copyright", "(c)", "©", "copyrighted", "COPYRIGHTS"])
def test_copyright_markers(text):
    assert COPYRIGHT_MARKER_RE.search(text)


@pytest.mark.parametrize("text", ["Copyright: see file License: MIT", "Copyright: See individual files",
                                  "copyright holders (see COPYING).", "Copyright (C) 2011 Red Hat see file 'COPYING'"])
def test_pointers(text):
    assert POINTER_RE.search(text)


@pytest.mark.parametrize("text", ["see you at the conference", "Copyright 2019 Seeburger AG"])
def test_not_pointers(text):
    assert not POINTER_RE.search(text)


@pytest.mark.parametrize("entity,expected", [
    ("Block Header", False), ("Status", False), ("Copyright Notice", False), ("¦", False), ("*", False), ("###", False),
    ("华为技术有限公司", True),
    ("Acme Corp", True), ("Free Software Foundation, Inc.", True), ("OCI Contributors", True), ("IBM", True),
])
def test_names_someone(entity, expected):
    assert _names_someone(entity) is expected


@pytest.mark.parametrize("text", [
    'Copyright", DlgDocPreview : "Vista Previa"', 'Copyright", //MISSING DlgDocPreview : "Preview"',
    "Copyright').value = GetMetadata( 'copyright' ) ;", "copyright' , GetE('txtMetaCopyright').value ) ;",
    "Copyright'.", 'Copyright", copyright); if (x)', "copyright_re = 'Copyright';",
])
def test_code_labels(text):
    assert CODE_LABEL_RE.search(text)


@pytest.mark.parametrize("text", [  # a key followed by its value: the value is the statement
    'VALUE "LegalCopyright", "Copyright (C) 2013-2020, Yann Collet"', 'Copyright",\t"(C) 1995-2022 Jean-loup Gailly',
    'Copyright": "Guido Bertoni, Joan Daemen"', 'copyright"] = "(C) 2013-2022 Niels Lohmann";',
    'copyright"] == "(C) 2013-2022 Niels Lohmann");', 'Copyright"]=> string(24) "Eric Stewart, Hex Editor"',
    'copyright = "2014-2018, Anaconda, Inc."', "Copyright (c) 2019 Jane Doe",
])
def test_key_with_value_is_not_a_code_label(text):
    assert not CODE_LABEL_RE.search(text)
