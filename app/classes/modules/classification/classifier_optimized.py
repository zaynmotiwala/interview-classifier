"""
Optimized RevReply Classifier
Simplified DSPy implementation for production use
"""

import dspy
import os
import pickle
import sys
import logging
from datetime import datetime, timezone
from app.dual_logger import log_usage_data_background
# from app.classes.signatures.thread_cleaner import CleanEmailThread  # Not needed for interview project
from app.dual_logger import log_custom_index
from concurrent.futures import ThreadPoolExecutor
from app.classes.helpers.token_calculator import calculate_total_tokens
from app.classes.signatures.delegation_detector import DelegationDetector
from app.classes.signatures.team_alerter import TeamAlertDetection
from app.classes.signatures.phone_number_extractor import PhoneNumberExtractor
from app.utils import clean_phone_number

_executor = ThreadPoolExecutor(max_workers=2)

ENVIRONMENT = os.environ.get("ENVIRONMENT")
try:
    from .... import __version__ as SERVICE_VERSION
except ImportError:
    SERVICE_VERSION = "unknown"


class RevReplyClassifier(dspy.Module):
    def __init__(
        self,
        json_path: str = None,
        lm: dspy.LM = None,
        cache: bool = True,
        enable_delegation_detection: bool = True,
    ):
        super().__init__()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for optimized model first, fallback to latest
        json_path = json_path or self._find_latest_model(current_dir)

        print(f"Resolved optimized classifier path: {json_path}")
        print(f"Working directory: {os.getcwd()}")
        print(f"PythonPath: {sys.path[:3]}...")
        print(f"PYTHONPATH env var: {os.environ.get('PYTHONPATH', 'Not set')}")

        try:
            import pydantic
            print(f"Pydantic version: {pydantic.VERSION}")
        except Exception as e:
            print(f"Pydantic import failed: {e}")

        print(f"DSPy version: {dspy.__version__}")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Path does not exist: {json_path}")

        try:
            self.classifier = dspy.load(json_path, allow_pickle=True)
            print("✅ Optimized classifier loaded successfully")
        except Exception as e:
            print(f"❌ Error loading optimized classifier: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Use GPT-5-mini as default
        self.lm = lm or dspy.LM(
            "azure/gpt-5-mini",
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            max_tokens=16000,
            temperature=1.0,   # Required for GPT-5-mini
            reasoning_effort="medium",  # Reduces thinking tokens to improve latency
            cache=cache
        )

        self._last_usage = None
        # self.cleaner = dspy.Predict(CleanEmailThread)

        # Initialize delegation detector for OOO toggle feature
        self.enable_delegation_detection = enable_delegation_detection
        self.delegation_detector = None
        if enable_delegation_detection:
            try:
                # todo instantiate the delegation detector dspy signature not based on a condition
                self.delegation_detector = dspy.ChainOfThought(DelegationDetector)
                print("✅ Delegation detector loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load delegation detector: {e}")
                self.delegation_detector = None


        # Initialize phone extractor and team alerter for same-day phone request detection
        try:
            self.phone_extractor = dspy.ChainOfThought(PhoneNumberExtractor)
            print("✅ Phone extractor loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load phone extractor: {e}")
            self.phone_extractor = None

        try:
            self.team_alerter = dspy.ChainOfThought(TeamAlertDetection)
            print("✅ Team alerter loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load team alerter: {e}")
            self.team_alerter = None


    def _find_latest_model(self, search_dir):
        """Find the latest optimized model, fallback to regular model"""
        optimized_models = []
        regular_models = []
        
        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path):
                if item.startswith("revclassifier-optimized-"):
                    optimized_models.append(item)
                elif item.startswith("revclassifier-"):
                    regular_models.append(item)
        
        # Prefer optimized models
        if optimized_models:
            latest = sorted(optimized_models)[-1]
            return os.path.join(search_dir, latest)
        elif regular_models:
            latest = sorted(regular_models)[-1]  
            return os.path.join(search_dir, latest)
        else:
            # Fallback to hardcoded path
            return os.path.join(search_dir, "revclassifier-20250926205854")

    def get_lm_usage(self):
        """Return the last captured LM usage data."""
        return self._last_usage or {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }

    def forward(
        self,
        email_thread: str,
        user_id: int = None,
        persona_id: int = None,
        ooo_toggle: bool = False,
        check_same_day_phone_request: bool = False,
        disqualification_criteria: str = "",
    ):
        # Reset usage tracking
        if hasattr(self.lm, 'reset_usage'):
            self.lm.reset_usage()


        # with dspy.context(lm=dspy.LM(
        #             "azure/gpt-4o-mini",  # Best choice for extraction
        #             api_key=os.getenv("AZURE_API_KEY"),
        #             api_base=os.getenv("AZURE_API_BASE"),
        #             api_version=os.getenv("AZURE_API_VERSION"),
        #             max_tokens=4000,
        #             temperature=0.0,  # Zero temp = maximum consistency
        #             cache=True
        #         )):
        #     email_thread = self.cleaner(raw_thread=email_thread)["cleaned_thread"]
        #     print("Cleaned email_thread", email_thread)

            #  we want to log the cleaned email thread in the background
            # _executor.submit(log_custom_index, {
            #     "email_thread": email_thread,
            #     "cleaned_email_thread": cleaned_email_thread,
            #     "date_cleaned": datetime.now().isoformat()
            # }, "cleaned_email_threads")


        # 🚨 RULE-BASED OVERRIDE for user ID -> 1126: Specific question patterns that should always be INFORMATION REQUEST
        specific_question_patterns = [
            "do you guys still take pickup/takeout through your direct line, or is it mostly online?"
        ]


        # Check if thread contains any specific question pattern (case-insensitive)
        rule_override_triggered = False
        email_thread_lower = email_thread.lower()
        for pattern in specific_question_patterns:
            if pattern.lower() in email_thread_lower:
                rule_override_triggered = True
                logging.info(f"✅ Rule-based override triggered: Thread contains pattern '{pattern[:50]}...' -> INFORMATION REQUEST")
                break

        # Run classifier (or create override result)
        if rule_override_triggered:
            # Create a result object that mimics the classifier output
            class OverrideResult:
                def __init__(self):
                    self.classification = "INFORMATION REQUEST"
                    self.reasoning = f"Email thread contains specific question pattern - automatically classified as INFORMATION REQUEST (rule-based override)"

            result = OverrideResult()
            # Set minimal usage since we didn't call the LM
            self._last_usage = {
                'completion_tokens': 0,
                'prompt_tokens': 0,
                'total_tokens': 0
            }
        else:
            # Normal classification path
            with dspy.context(lm=self.lm):
                result = self.classifier(
                    email_thread=email_thread,
                    disqualification_criteria=disqualification_criteria
                )

        # 🚨 OVERRIDE: "I will forward to" = REQUEST COMPLETE (not DELEGATED)
        # Prospect is passing it along themselves — conversation is closed from their end.
        # DELEGATED is only when prospect redirects YOU to contact someone else directly.
        import re as _re2
        forward_patterns = [
            r"\bwill forward\b",
            r"\bwill pass (this |it )?(along|on)\b",
            r"\bforwarding (this |it )?(along|on)\b",
            r"\bpass (this |it )?(along|on) to\b",
            r"\bforward (this |it )?(along|on|to)\b",
        ]
        first_lines = "\n".join(email_thread.strip().split("\n")[:5]).lower()
        if result.classification == "DELEGATED":
            for pat in forward_patterns:
                if _re2.search(pat, first_lines):
                    logging.info(f"Override: 'will forward' pattern detected → DELEGATED overridden to REQUEST COMPLETE")
                    result.classification = "REQUEST COMPLETE"
                    result.reasoning = f"Override: Prospect is forwarding the message themselves (not redirecting you to contact someone else). {result.reasoning}"
                    break

        # 🚨 OVERRIDE: Data-sourcing questions = DO NOT CONTACT
        # Prospect challenges how their email/phone was obtained — privacy concern, must not be contacted again.
        # Fires for any classification except DO NOT CONTACT (already correct).
        data_source_skepticism_patterns = [
            r"\bwhich database\b",
            r"\bwhat database\b",
            r"\bhow did you (get|find|obtain|source) (my |this )?(email|contact|number|phone|info|details|address)\b",
            r"\bwhere did you (get|find|obtain|source) (my |this )?(email|contact|number|phone|info|details|address)\b",
            r"\bnot on any public (business )?records\b",
            r"\bnot (listed|found) (anywhere|publicly|in any)\b",
            r"\bwhere (did you|do you) get (your |these )?(contacts?|emails?|leads?|data|numbers?)\b",
            r"\bhow did you (even )?get (my |this )?(email|contact|number|phone)\b",
            r"\bhow did you find (my |me\b)",
            r"\bwho gave you my (number|email|contact|phone)\b",
            r"\bhow did you get my (number|phone)\b",
            r"\bnever (gave|shared|provided) (my |consent)\b",
            r"\bnever gave consent\b",
        ]
        if result.classification != "DO NOT CONTACT":
            for pat in data_source_skepticism_patterns:
                if _re2.search(pat, first_lines):
                    logging.info(f"Override: data sourcing question detected → {result.classification} overridden to DO NOT CONTACT")
                    result.classification = "DO NOT CONTACT"
                    result.reasoning = f"Override: Prospect is questioning how their contact data was obtained — privacy concern, must not be contacted again. {result.reasoning}"
                    break

        # 🚨 OVERRIDE: Short acknowledgment after Check Back Later = THREAD COMPLETE
        # MUST run BEFORE tentative-interest override — prevents "sounds good" / "will do" /
        # "thank you" from being misclassified as INTERESTED when the prospect is just
        # acknowledging a Check Back Later response the agent already sent.
        import re as _re
        thread_complete_ack_words = {
            "will do", "will do thanks", "will do thank you",
            "sounds good", "sounds good thanks", "sounds good thank you",
            "thank you", "thanks", "got it", "got it thanks", "got it thank you",
            "perfect", "perfect thanks", "great", "great thanks",
            "ok", "ok thanks", "okay", "okay thanks",
            "noted", "understood", "appreciate it", "cheers",
            "all good", "no problem", "np", "cool", "alright",
            "thx", "ty", "tyvm", "👍", "👌",
            # Multi-phrase closures: prospect sends two short ack phrases together
            "perfect thank you", "perfect thanks", "perfect thank you so much",
            "great thank you", "great thanks", "great thank you so much",
            "sounds good thank you", "sounds good thanks",
            "ok thank you", "ok thanks", "okay thank you", "okay thanks",
            "will do thank you", "will do thanks",
            # Meeting confirmation closures: prospect acknowledges a confirmed meeting time
            "likewise", "likewise thanks", "likewise thank you",
            "see you then", "see you there",
            "looking forward to it", "looking forward to chatting", "looking forward to speaking",
        }
        check_back_language = [
            "check back", "follow up", "follow-up", "circle back",
            "touch base", "reach back out", "get back to you", "reconnect",
            "in q1", "in q2", "in q3", "in q4",
            "next quarter", "next month", "next year",
            "in a few months", "in a couple months", "in a few weeks",
            "talk soon", "catch up later", "revisit",
        ]
        # Language indicating the sender has already confirmed a specific meeting time.
        # When the prospect replies with a short ack after this, the thread is closed.
        meeting_confirmation_language = [
            "looking forward to chatting with you on",
            "looking forward to speaking with you on",
            "looking forward to talking with you on",
            "looking forward to meeting with you on",
            "looking forward to our call on",
            "see you on",
            "confirmed for",
            "all set for",
            "scheduled for",
            "booked for",
        ]

        # Extract the most recent prospect message (content before first outbound separator).
        # Handles three real-world formats:
        #   A) "From:" headers with leading newlines (standard)
        #   B) "On [date] ... wrote:" with leading newlines (Gmail/Outlook)
        #   C) Single-line plain-text (HTML-stripped, no newlines) — split without requiring leading \n
        _sep_pattern = r'\n\n?From:|\n\n?On .{5,80} wrote:|(?<!\n)(?:From:|On [A-Z][a-z]{2}.{5,60} wrote:)'
        recent_part = _re.split(_sep_pattern, email_thread, maxsplit=1)[0]
        recent_lines = [l.strip() for l in recent_part.strip().split('\n') if l.strip()]
        recent_text_lines = [
            l for l in recent_lines
            if not l.lower().startswith(('subject:', 'from:', 'to:', 'date:', 'sent:'))
        ]
        # For single-line threads the subject "Re: XYZ <message>" needs inline extraction
        # If all lines were filtered (e.g. only a "Re:" subject line), try stripping the Re: prefix
        if not recent_text_lines and recent_lines:
            for line in recent_lines:
                cleaned = _re.sub(r'^re\s*:\s*[\w\-/]+\s*', '', line.strip(), flags=_re.IGNORECASE).strip()
                if cleaned:
                    recent_text_lines = [cleaned]
                    break
        # Normalize: lowercase, strip punctuation for flexible matching
        recent_raw = ' '.join(recent_text_lines).strip().lower()
        recent_message = _re.sub(r'[.,!?;:\-]', '', recent_raw).strip()
        recent_message = _re.sub(r'\s+', ' ', recent_message)
        # Strip leading Re:/Fwd: + optional thread-ID prefix.
        # Note: recent_message has already had punctuation (colons, hyphens) removed, so
        # "Re: STGRV0-2068/1 Perfect..." normalizes to "re stgrv02068/1 perfect..."
        # Pattern: "re " or "fwd " followed by optional alphanumeric thread-ID token and space.
        recent_msg_stripped = _re.sub(r'^(?:re|fwd?)\s+[\w/]+\s+', '', recent_message, flags=_re.IGNORECASE).strip()
        # Also handle plain "Re: " with no thread ID (after colon stripped: "re  ..." → "re ...")
        recent_msg_stripped = _re.sub(r'^(?:re|fwd?)\s+', '', recent_msg_stripped, flags=_re.IGNORECASE).strip()

        # Normalize smart/curly apostrophes so phrase matching works regardless of email client
        thread_lower = email_thread.lower().replace('\u2019', "'").replace('\u2018', "'")
        has_check_back_context = any(phrase in thread_lower for phrase in check_back_language)
        has_meeting_confirmation_context = any(phrase in thread_lower for phrase in meeting_confirmation_language)

        # Guard: prospect is proposing a meeting — never collapse to THREAD COMPLETE
        meeting_proposal_phrases = [
            "let's meet", "lets meet", "let us meet",
            "i'll meet", "want to meet", "can we meet",
        ]
        has_meeting_proposal = any(phrase in thread_lower for phrase in meeting_proposal_phrases)

        # Match strategy:
        #   1. Exact match on full recent_message (short clean threads)
        #   2. starts-with match on prefix-stripped message (inline/signature-heavy threads)
        #      — only when the remainder after the ack word is NOT scheduling content
        scheduling_continuations = [
            "meet", "call", "schedule", "book",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            " am", " pm", "morning", "afternoon", "evening",
        ]
        def _startswith_ack_only(msg, ack_words):
            """Return matched phrase only if the text after the ack word is not scheduling content."""
            for phrase in ack_words:
                if msg.startswith(phrase):
                    remainder = msg[len(phrase):].strip()
                    if not remainder or not any(kw in remainder for kw in scheduling_continuations):
                        return phrase
            return None

        in_ack = (
            recent_message in thread_complete_ack_words
            or _startswith_ack_only(recent_msg_stripped, thread_complete_ack_words) is not None
        )
        matched_phrase = next(
            (p for p in thread_complete_ack_words if recent_message == p),
            _startswith_ack_only(recent_msg_stripped, thread_complete_ack_words)
        )

        logging.info(f"[thread_complete_override] recent_message={repr(recent_message)} | stripped={repr(recent_msg_stripped[:60])} | has_check_back={has_check_back_context} | has_meeting_confirm={has_meeting_confirmation_context} | has_meeting_proposal={has_meeting_proposal} | in_ack={in_ack}")

        if has_check_back_context and in_ack and not has_meeting_proposal:
            result.classification = "THREAD COMPLETE"
            result.reasoning = f"Override: Short acknowledgment '{matched_phrase}' after Check Back Later context = THREAD COMPLETE. {result.reasoning}"

        # 🚨 OVERRIDE: Short acknowledgment after meeting confirmation = THREAD COMPLETE
        # Fires regardless of LLM classification (MEETING BOOKED or INTERESTED) —
        # when the sender already confirmed a specific meeting time and the prospect
        # replies with a brief ack/pleasantry, there is nothing left to respond to.
        # MUST run before the tentative-interest override below.
        if result.classification != "THREAD COMPLETE" and has_meeting_confirmation_context and in_ack and not has_meeting_proposal:
            result.classification = "THREAD COMPLETE"
            result.reasoning = f"Override: Short acknowledgment '{matched_phrase}' after meeting confirmation = THREAD COMPLETE (meeting confirmed, no further response needed). {result.reasoning}"

        # 🔥 NEW: OVERRIDE: Short acknowledgment after OTHER CLOSURE contexts = THREAD COMPLETE
        # Graceful exit, delegation accepted, information delivered
        graceful_exit_language = [
            "no problem", "no worries", "reach out if you need", "keep us in mind",
            "best of luck", "whenever you're ready", "if anything changes",
            "feel free to reach out", "don't hesitate to contact",
        ]
        delegation_language = [
            "i'll connect with", "i'll reach out to", "i'll contact",
            "i'll loop in", "i'll introduce you to",
        ]
        has_graceful_exit_context = any(phrase in thread_lower for phrase in graceful_exit_language)
        has_delegation_context = any(phrase in thread_lower for phrase in delegation_language)

        if result.classification != "THREAD COMPLETE" and (has_graceful_exit_context or has_delegation_context) and in_ack and not has_meeting_proposal:
            result.classification = "THREAD COMPLETE"
            result.reasoning = f"Override: Short acknowledgment '{matched_phrase}' after closure statement = THREAD COMPLETE. {result.reasoning}"

        # 🔥 NEW: OVERRIDE: Short affirmative after ENGAGEMENT proposal = INTERESTED
        # When agent proposes engagement (not closure), positive ack = INTERESTED
        # MUST run after all THREAD COMPLETE overrides
        engagement_proposal_language = [
            "would you like", "interested in", "want to", "how about",
            "does that work", "make sense", "sound good to you", "thoughts on",
            "can I send", "should I send", "would it be helpful",
            "can I show you", "want me to", "let me know if",
        ]
        positive_ack_words = {
            "yes", "yeah", "yep", "sure", "ok", "okay",
            "sounds good", "perfect", "great", "alright", "cool",
        }
        has_engagement_context = any(phrase in thread_lower for phrase in engagement_proposal_language)

        # Check if recent message is a positive ack (exact match or starts with)
        is_positive_ack = (
            recent_message in positive_ack_words
            or recent_msg_stripped in positive_ack_words
            or any(recent_message.startswith(word) or recent_msg_stripped.startswith(word) for word in positive_ack_words)
        )

        if result.classification != "THREAD COMPLETE" and result.classification != "INTERESTED" and has_engagement_context and is_positive_ack:
            result.classification = "INTERESTED"
            result.reasoning = f"Override: Positive acknowledgment after engagement proposal = INTERESTED (agreeing to proceed). {result.reasoning}"

        # 🚨 OVERRIDE: Prospect self-booking confirmation = THREAD COMPLETE (not MEETING BOOKED)
        # "I booked a time" / "just booked" / etc. means the prospect used a calendar link themselves.
        # There is nothing left to send — the conversation is closed.
        # Fires only when LLM classified as MEETING BOOKED; overrides to THREAD COMPLETE.
        import re as _re3
        self_booking_patterns = [
            r"\bi (already |just |went ahead and )?(booked|scheduled) (a |the )?(time|slot|meeting|call|appointment)\b",
            r"\bjust booked\b",
            r"\balready booked\b",
            r"\bi went ahead and booked\b",
            r"\bbooked a (time|slot|meeting|call|appointment)\b",
            r"\bscheduled a (time|slot|meeting|call|appointment)\b",
        ]
        if result.classification == "MEETING BOOKED":
            for pat in self_booking_patterns:
                if _re3.search(pat, recent_raw):
                    logging.info(f"Override: self-booking confirmation detected → MEETING BOOKED overridden to THREAD COMPLETE")
                    result.classification = "THREAD COMPLETE"
                    result.reasoning = f"Override: Prospect self-booked ('{pat}' matched) — no response needed. {result.reasoning}"
                    break

        # 🔥 CONTEXT-AWARE POSITIVE INTENT OVERRIDE
        # When prospect uses positive words (yes/sure/maybe), determine the RIGHT positive classification
        # based on what the agent said before. Same word can mean different things in different contexts:
        # - "Sure" after time proposal → MEETING REQUEST
        # - "Sure" after "want to learn more?" → INTERESTED
        # - "Sure" after "talk soon!" → THREAD COMPLETE (already handled by closure overrides above)
        #
        # ⚠️ Only fires if not already THREAD COMPLETE AND no closure context exists
        # (closure overrides above take priority)
        has_any_closure_context = (
            has_check_back_context or has_meeting_confirmation_context
            or has_graceful_exit_context or has_delegation_context
        )

        positive_intent_words = {
            "maybe", "may be", "may b", "may e", "mayb", "mabe", "mabye",
            "perhaps", "possibly", "sure",
            "alright", "could be", "might be", "sounds good", "sounds interesting",
            "tell me more", "let me know more", "interested", "yes", "yep", "yeah",
            "ok", "okay",
        }

        if result.classification != "THREAD COMPLETE" and not has_any_closure_context:
            # Check if prospect used positive intent word
            prospect_used_positive_word = False
            matched_word = None
            for line in email_thread.strip().split('\n')[:5]:
                stripped = line.strip().lower().rstrip(".,!?;:")
                if stripped in positive_intent_words or any(stripped.startswith(word + " ") for word in positive_intent_words):
                    prospect_used_positive_word = True
                    matched_word = stripped
                    break

            if prospect_used_positive_word:
                # Extract agent's last message to detect context patterns
                # Look for the last "From:" or "On ... wrote:" separator
                agent_message_match = _re.split(r'(?:From:|On .{5,80} wrote:)', email_thread, maxsplit=2)
                agent_last_message = agent_message_match[1].lower() if len(agent_message_match) > 1 else ""

                # Pattern 1: Time proposal patterns → MEETING REQUEST
                time_proposal_patterns = [
                    r'\d{1,2}:\d{2}\s*(am|pm)',  # "11:00am", "3:30pm"
                    r'\d{1,2}\s*(am|pm)',  # "11am", "3pm"
                    r'@\s*\d',  # "@ 11am"
                    r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                    r'(does|how does|work for you|available for|free for|open for)',
                ]
                has_time_proposal = any(_re.search(pat, agent_last_message) for pat in time_proposal_patterns)

                # Pattern 2: Meeting already confirmed → THREAD COMPLETE (but should be caught by closure overrides)
                meeting_confirmed_patterns = [
                    'meeting is booked', 'meeting is confirmed', 'meeting is scheduled',
                    'confirmed for', 'scheduled for', 'booked for',
                    'calendar invite sent', 'invite sent',
                ]
                has_meeting_confirmed = any(phrase in agent_last_message for phrase in meeting_confirmed_patterns)

                # Pattern 3: Engagement question → INTERESTED
                engagement_patterns = [
                    'want to learn more', 'would you like to', 'interested in',
                    'can i show you', 'would it be helpful', 'want to see',
                    'should i send', 'can i send', 'would you be open to',
                ]
                has_engagement_question = any(phrase in agent_last_message for phrase in engagement_patterns)

                # Choose classification based on detected pattern
                if has_time_proposal:
                    result.classification = "MEETING REQUEST"
                    result.reasoning = f"Override: Positive word '{matched_word}' after time proposal = MEETING REQUEST. {result.reasoning}"
                elif has_meeting_confirmed:
                    result.classification = "THREAD COMPLETE"
                    result.reasoning = f"Override: Positive word '{matched_word}' after meeting confirmation = THREAD COMPLETE. {result.reasoning}"
                elif has_engagement_question:
                    result.classification = "INTERESTED"
                    result.reasoning = f"Override: Positive word '{matched_word}' after engagement question = INTERESTED. {result.reasoning}"
                else:
                    # Default to INFORMATION REQUEST (safest positive)
                    result.classification = "INFORMATION REQUEST"
                    result.reasoning = f"Override: Positive word '{matched_word}' detected, defaulting to INFORMATION REQUEST. {result.reasoning}"


        # DON'T capture usage yet - let both classifier and delegation detector accumulate

        # 🔄 OOO Toggle: Check for delegation in OUT OF OFFICE messages
        if ooo_toggle and result.classification == "OUT OF OFFICE":
            try:
                logging.info(f"OOO Toggle: Running delegation detector for OUT OF OFFICE message")

                # Check if delegation detector is available
                if self.delegation_detector is None:
                    logging.error("Delegation detector not available - failed to load at initialization")
                    logging.warning("Fallback: Keeping original OUT OF OFFICE classification")
                else:
                    # Run delegation detector using the same LM context
                    with dspy.context(lm=self.lm):
                        delegation_result = self.delegation_detector(email_thread=email_thread)

                    # DON'T capture usage here - will capture combined usage at the end

                    logging.debug(f"Delegation detection result: has_delegation={delegation_result.has_delegation}, reasoning={delegation_result.reasoning}")

                    # If delegation found, override classification to DELEGATED
                    if delegation_result.has_delegation:
                        logging.info(f"Delegation detected! Overriding: OUT OF OFFICE → DELEGATED (delegate={delegation_result.delegate_name}, email={delegation_result.delegate_email})")
                        result.classification = "DELEGATED"
                        result.reasoning = f"{result.reasoning} | OVERRIDE (ooo_toggle): Delegation detected - {delegation_result.reasoning}"
                    else:
                        logging.debug(f"No delegation detected. Keeping classification as OUT OF OFFICE")
            except Exception as e:
                logging.error(f"Delegation detection failed: {str(e)}", exc_info=True)
                # Fallback: keep original OUT OF OFFICE classification
                logging.warning(f"Fallback: Keeping original OUT OF OFFICE classification due to delegation detection error")
        elif ooo_toggle:
            logging.debug(f"OOO toggle enabled but classification is {result.classification}, skipping delegation detection")

        # 📞 Same-Day Phone Request Detection: Run TeamAlerter + PhoneNumberExtractor if requested
        phone_number = ""
        same_day_phone_request_detected = False

        # to do : check if its a positive classification then run the team alerter signature
        ALLOWED_CLASSIFICATIONS_FOR_PHONE = [
            "INTERESTED",
            "INFORMATION REQUEST",
            "CHECK BACK LATER",
            "MEETING REQUEST",
            "MEETING BOOKED",
            "SOFT OBJECTION"
        ]

        # Check if both flag is enabled AND classification allows phone detection
        should_run_team_alerter = (
            check_same_day_phone_request and
            result.classification in ALLOWED_CLASSIFICATIONS_FOR_PHONE
        )

        if should_run_team_alerter:
            try:
                logging.info(
                    f"Running TeamAlerter for same-day phone request detection "
                    f"(classification={result.classification})"
                )

                # Check if team alerter is available
                if self.team_alerter is None:
                    logging.error("TeamAlerter not available - failed to load at initialization")
                else:
                    # Calculate today's day name for validation (e.g., "Monday", "Tuesday", ...)
                    today_day_name = datetime.now(timezone.utc).strftime("%A")  # Returns "Monday", "Tuesday", etc.
                    logging.info(f"Today's day name: {today_day_name}")

                    # Run team alerter using the same LM context
                    with dspy.context(lm=self.lm):
                        alerter_result = self.team_alerter(
                            email_thread=email_thread,
                            today_day_name=today_day_name
                        )

                    # Extract detection results (TeamAlertDetection only detects, no phone extraction)
                    same_day_phone_request_detected = alerter_result.same_day_phone_request_detected

                    logging.info(
                        f"TeamAlerter detection: same_day_phone_request_detected={same_day_phone_request_detected}, "
                        f"ai_commentary={alerter_result.ai_commentary}"
                    )

                    # 📞 If same-day phone request detected, use PhoneNumberExtractor to get the phone number
                    if same_day_phone_request_detected and self.phone_extractor is not None:
                        logging.info("Same-day phone request detected - running PhoneNumberExtractor")

                        with dspy.context(lm=self.lm):
                            phone_result = self.phone_extractor(
                                email_thread=email_thread,
                                prospect_name="",  # Will be extracted from thread
                                sender_name="",    # Will be extracted from thread
                                meeting_context="Same-day phone call request detected",
                                is_meeting_scenario=True
                            )

                        raw_phone = phone_result.raw_phone_number if hasattr(phone_result, 'raw_phone_number') else ""
                        phone_number = clean_phone_number(raw_phone) if raw_phone else ""

                        logging.info(
                            f"PhoneNumberExtractor result: raw_phone={raw_phone}, formatted_phone={phone_number}"
                        )
                    elif same_day_phone_request_detected and self.phone_extractor is None:
                        logging.error("PhoneNumberExtractor not available - cannot extract phone number")

            except Exception as e:
                logging.error(f"Same-day phone detection failed: {str(e)}", exc_info=True)
                # Fallback: set defaults on error
                phone_number = ""
                same_day_phone_request_detected = False

        # 📊 Capture combined LM usage (classifier + delegation detector + team alerter if ran)
        # This happens ONCE at the end to avoid double-counting
        try:
            if hasattr(self.lm, 'get_usage_and_reset'):
                combined_usage = self.lm.get_usage_and_reset()
            elif hasattr(self.lm, 'usage'):
                combined_usage = self.lm.usage
            elif hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'get_usage_and_reset'):
                combined_usage = dspy.settings.lm.get_usage_and_reset()
            else:
                combined_usage = {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0
                }

            # Ensure we have a proper dict with required keys
            if combined_usage and isinstance(combined_usage, dict):
                combined_usage = {
                    'completion_tokens': combined_usage.get('completion_tokens', 0),
                    'prompt_tokens': combined_usage.get('prompt_tokens', 0),
                    'total_tokens': combined_usage.get('total_tokens', 0)
                }
            else:
                combined_usage = {
                    'completion_tokens': 0,
                    'prompt_tokens': 0,
                    'total_tokens': 0
                }
        except Exception as e:
            logging.warning(f"Could not capture LM usage: {e}")
            combined_usage = {
                'completion_tokens': 0,
                'prompt_tokens': 0,
                'total_tokens': 0
            }

        # Store combined usage data for later retrieval
        self._last_usage = combined_usage

        # 🚀 Calculate total tokens from combined usage
        # Wrap in a dict structure compatible with calculate_total_tokens
        lm_usage_dict = {
            "user_id": user_id,
            "persona_id": persona_id,
            "classifier": combined_usage
        }
        total_tokens = calculate_total_tokens(lm_usage_dict)
        print(f"🔍 CLASSIFIER DEBUG: Calculated total_tokens: {total_tokens}")

        # Add get_lm_usage, total_tokens, and phone detection results to result object
        result.get_lm_usage = lambda: self._last_usage
        result.total_tokens = total_tokens
        result.phone_number = phone_number
        result.same_day_phone_request_detected = same_day_phone_request_detected

        print(f"🔍 CLASSIFIER DEBUG: Set result.total_tokens = {total_tokens}")
        print(f"🔍 CLASSIFIER DEBUG: Verify hasattr(result, 'total_tokens'): {hasattr(result, 'total_tokens')}")
        print(f"📞 CLASSIFIER DEBUG: phone_number = {phone_number}, same_day_phone_request_detected = {same_day_phone_request_detected}")

        usage_data = {
            "module": "OptimizedRevReplyClassifier",
            "timestamp": datetime.now().isoformat(),
            "lm_usage": combined_usage,  # Combined usage from classifier + delegation detector + team alerter
            "ooo_toggle_enabled": ooo_toggle,
            "delegation_detection_ran": ooo_toggle and result.classification in ["OUT OF OFFICE", "DELEGATED"],
            "check_same_day_phone_request": check_same_day_phone_request,
            "team_alerter_ran": check_same_day_phone_request,
            "same_day_phone_request_detected": same_day_phone_request_detected,
            "phone_number_found": bool(phone_number),
            "user_id": user_id,
            "persona_id": persona_id,
            "email_thread_length": len(email_thread) if email_thread else 0,
            "classification_result": getattr(result, "classification", str(result)),
            "environment": ENVIRONMENT,
            "service_version": SERVICE_VERSION,
        }

        log_usage_data_background(usage_data)
        return result
    
    def __call__(
        self,
        email_thread: str,
        user_id: int = None,
        persona_id: int = None,
        ooo_toggle: bool = False,
        check_same_day_phone_request: bool = False,
        disqualification_criteria: str = "",
    ):
        """Allow calling the module directly without the warning."""
        return self.forward(
            email_thread,
            user_id,
            persona_id,
            ooo_toggle,
            check_same_day_phone_request,
            disqualification_criteria,
        )
        