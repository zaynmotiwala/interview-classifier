"""
Basic classifier tests for the interview project.

These tests demonstrate:
1. Basic classification functionality
2. Rule-based overrides
3. Edge cases and context-awareness
"""
import pytest


class TestBasicClassification:
    """Test basic classification functionality."""

    def test_interested_classification(self, classifier):
        """Test INTERESTED classification for clear positive responses."""
        thread = """
        From: agent@company.com
        Subject: Re: Product Demo

        Hi there,

        Would you be interested in learning more about our product?

        From: prospect@email.com
        Subject: Re: Product Demo

        Yes, I would love to learn more! Please send me details.
        """

        result = classifier(email_thread=thread)
        assert result.classification == "INTERESTED"
        assert "reasoning" in dir(result)

    def test_not_interested_classification(self, classifier):
        """Test NOT INTERESTED classification for clear rejections."""
        thread = """
        From: agent@company.com
        Subject: Partnership Opportunity

        Hello, we have an exciting partnership opportunity.

        From: prospect@email.com
        Subject: Re: Partnership Opportunity

        Thanks but not interested.
        """

        result = classifier(email_thread=thread)
        assert result.classification == "NOT INTERESTED"

    def test_information_request_classification(self, classifier):
        """Test INFORMATION REQUEST for questions."""
        thread = """
        From: agent@company.com
        Subject: Our Solution

        We help companies reduce costs.

        From: prospect@email.com
        Subject: Re: Our Solution

        How much does it cost?
        """

        result = classifier(email_thread=thread)
        assert result.classification == "INFORMATION REQUEST"

    def test_meeting_request_classification(self, classifier):
        """Test MEETING REQUEST for scheduling requests."""
        thread = """
        From: agent@company.com
        Subject: Quick Call?

        Would you have time for a quick 15-minute call?

        From: prospect@email.com
        Subject: Re: Quick Call?

        How about Tuesday at 2pm?
        """

        result = classifier(email_thread=thread)
        assert result.classification == "MEETING REQUEST"


class TestRuleBasedOverrides:
    """Test rule-based override logic."""

    def test_do_not_contact_override(self, classifier):
        """Test DO NOT CONTACT override for unsubscribe requests."""
        thread = """
        From: agent@company.com
        Subject: Follow Up

        Just following up on my previous email.

        From: prospect@email.com
        Subject: Re: Follow Up

        Please unsubscribe me from your emails.
        """

        result = classifier(email_thread=thread)
        assert result.classification == "DO NOT CONTACT"

    def test_thread_complete_after_check_back_later(self, classifier):
        """
        Test THREAD COMPLETE override when prospect acknowledges
        a check-back-later message.
        """
        thread = """
        From: agent@company.com
        Subject: Partnership

        Hi, interested in partnering?

        From: prospect@email.com
        Subject: Re: Partnership

        Check back with me in Q2.

        From: agent@company.com
        Subject: Re: Partnership

        Sounds good, I'll reach out in April. Talk soon!

        From: prospect@email.com
        Subject: Re: Partnership

        Perfect, thanks!
        """

        result = classifier(email_thread=thread)
        assert result.classification == "THREAD COMPLETE"

    def test_data_sourcing_question_override(self, classifier):
        """
        Test that data sourcing questions trigger DO NOT CONTACT.
        Privacy concern = must not contact again.
        """
        thread = """
        From: agent@company.com
        Subject: Introduction

        Hi, I wanted to reach out about our service.

        From: prospect@email.com
        Subject: Re: Introduction

        How did you get my email address?
        """

        result = classifier(email_thread=thread)
        assert result.classification == "DO NOT CONTACT"


class TestContextAwareness:
    """Test context-aware classification (same word, different meanings)."""

    def test_sure_after_engagement_question(self, classifier):
        """
        'Sure' after 'want to learn more?' = INTERESTED
        """
        thread = """
        From: agent@company.com
        Subject: Our Product

        We have a great product that could help you. Want to learn more?

        From: prospect@email.com
        Subject: Re: Our Product

        Sure
        """

        result = classifier(email_thread=thread)
        # Should be INTERESTED because it's responding to an engagement question
        assert result.classification == "INTERESTED"

    def test_sure_after_time_proposal(self, classifier):
        """
        'Sure' after time proposal = MEETING REQUEST
        """
        thread = """
        From: agent@company.com
        Subject: Schedule Call

        How about Tuesday at 3pm?

        From: prospect@email.com
        Subject: Re: Schedule Call

        Sure
        """

        result = classifier(email_thread=thread)
        # Should be MEETING REQUEST because it's confirming a time
        assert result.classification == "MEETING REQUEST"

    def test_maybe_with_condition(self, classifier):
        """
        'Maybe' with condition = INFORMATION REQUEST
        (not INTERESTED, because it depends on future info)
        """
        thread = """
        From: agent@company.com
        Subject: Investment Opportunity

        Are you interested in selling your business?

        From: prospect@email.com
        Subject: Re: Investment Opportunity

        Maybe, depends on the price.
        """

        result = classifier(email_thread=thread)
        assert result.classification == "INFORMATION REQUEST"


class TestTokenUsageTracking:
    """Test token usage tracking functionality."""

    def test_token_usage_captured(self, classifier):
        """Verify that token usage is captured and returned."""
        thread = "Hi, I'm interested in learning more about your product."

        result = classifier(email_thread=thread)

        # Check that usage data exists
        assert hasattr(result, 'get_lm_usage')
        usage = result.get_lm_usage()
        assert 'completion_tokens' in usage
        assert 'prompt_tokens' in usage
        assert 'total_tokens' in usage

        # Check that total_tokens is calculated
        assert hasattr(result, 'total_tokens')
        assert 'input' in result.total_tokens
        assert 'output' in result.total_tokens

    def test_token_usage_with_ooo_toggle(self, classifier):
        """
        Verify that token usage includes delegation detection
        when ooo_toggle is enabled.
        """
        thread = """
        From: agent@company.com
        Subject: Partnership

        Hi, would you be interested?

        From: prospect@email.com
        Subject: Out of Office

        I'm out of office. Please contact my colleague Bob at bob@company.com
        """

        result = classifier(
            email_thread=thread,
            ooo_toggle=True  # Enable delegation detection
        )

        # Should detect delegation and override to DELEGATED
        assert result.classification == "DELEGATED"

        # Token usage should include both classifier + delegation detector
        usage = result.get_lm_usage()
        assert usage['total_tokens'] > 0
