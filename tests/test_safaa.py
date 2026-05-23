# SPDX-FileCopyrightText: © 2026 RAJVEER42 <irajveer.bishnoi2310@gmail.com>
#
# SPDX-License-Identifier: LGPL-2.1-only

import pytest

from safaa.Safaa import SafaaAgent


@pytest.fixture(scope="module")
def agent():
    return SafaaAgent()


# ---------------------------------------------------------------------------
# _perform_text_substitutions — email replacement
# ---------------------------------------------------------------------------

class TestPerformTextSubstitutionsEmail:

    def test_simple_email_replaced(self, agent):
        result = agent._perform_text_substitutions(["contact john@example.com"])
        tokens = result[0].split()
        assert "email" in tokens
        assert "john" not in tokens
        assert "example" not in tokens

    def test_email_with_dots_in_local_part(self, agent):
        result = agent._perform_text_substitutions(["user.name@example.com"])
        tokens = result[0].split()
        assert "email" in tokens
        assert "user" not in tokens
        assert "name" not in tokens

    def test_email_with_plus_tag(self, agent):
        result = agent._perform_text_substitutions(["user+tag@example.com"])
        tokens = result[0].split()
        assert "email" in tokens
        assert "user" not in tokens
        assert "tag" not in tokens

    def test_email_with_subdomain(self, agent):
        result = agent._perform_text_substitutions(["user@mail.sub.example.co.uk"])
        tokens = result[0].split()
        assert "email" in tokens
        assert "user" not in tokens
        assert "sub" not in tokens
        assert "example" not in tokens

    def test_email_with_hyphen_in_domain(self, agent):
        result = agent._perform_text_substitutions(["user@my-domain.com"])
        tokens = result[0].split()
        assert "email" in tokens
        assert "domain" not in tokens

    def test_email_in_full_copyright_context(self, agent):
        result = agent._perform_text_substitutions(
            ["Copyright (C) 2024 author@example.com All rights reserved"]
        )
        tokens = result[0].split()
        assert "email" in tokens
        assert "copyrightsymbol" in tokens
        assert "date" in tokens
        assert "author" not in tokens
        assert "example" not in tokens

    def test_string_without_email_left_alone(self, agent):
        result = agent._perform_text_substitutions(["just a regular sentence"])
        tokens = result[0].split()
        assert "email" not in tokens

    def test_multiple_emails_in_one_string(self, agent):
        result = agent._perform_text_substitutions(
            ["contact a@b.com or c@d.com for info"]
        )
        assert result[0].count("email") == 2

    def test_at_symbol_alone_does_not_become_email(self, agent):
        # A stray @ is not a valid email; should be stripped as non-alphanumeric
        result = agent._perform_text_substitutions(["foo @ bar"])
        tokens = result[0].split()
        assert "email" not in tokens

    def test_email_with_four_plus_digits_known_limitation(self, agent):
        # Known limitation: the \d{4} year rule runs before the email rule, so
        # any email containing 4+ consecutive digits is corrupted before the
        # email regex can match. Documented here so a future fix can flip the
        # assertion once the substitution order is corrected.
        result = agent._perform_text_substitutions(["author5565@example.com"])
        tokens = result[0].split()
        assert "email" not in tokens, (
            "If this passes, the digits-in-email issue is now fixed and the "
            "test should be updated to assert 'email' IS in tokens."
        )
        assert "date" in tokens
        assert "author" in tokens

    def test_no_email_no_false_positive_tokens(self, agent):
        # An @-less string with copyright-shaped content must not gain EMAIL
        result = agent._perform_text_substitutions(["Copyright 2024 Siemens AG"])
        tokens = result[0].split()
        assert "email" not in tokens
