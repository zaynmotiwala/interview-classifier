import dspy


class TeamAlertDetection(dspy.Signature):
    """
    Detect SAME-DAY phone call requests in the latest email.

    **PURPOSE**: Identify when a prospect wants a phone call TODAY or on the same day as a previously mentioned date.

    **NOTE**: This signature ONLY detects same-day phone requests. Phone number extraction is handled
    separately by the PhoneNumberExtractor signature.

    **CRITICAL RULES**:
    1. Focus ONLY on the most recent message (Email 1 = newest)
    2. Only detect phone call requests FROM THE PROSPECT (not from the sender/agent)
    3. Only detect phone call requests (not video meetings or general meetings)
    4. DEFAULT TO SAME-DAY: Phone call WITHOUT future timing = same-day (True)
    5. POSITIVE INTENT: Imperative + phone number = STRONG same-day signal ("Call me at 555-1234")
    6. URGENCY WORDS: ASAP, NOW, TODAY, THIS AFTERNOON = same-day intent
    7. ALWAYS return False if prospect says: STOP, REMOVE, UNSUBSCRIBE, OPT OUT, NO CONTACT, END CONTACT

    **WHO IS REQUESTING?**:
    - PROSPECT requesting callback: "Can you call me?", "Ring me at 555-1234", "Happy to talk" → TRUE (detect)
    - SENDER offering to call: "Happy to call your number", "I can call you", "Let me call you" → FALSE (ignore)

    **DO NOT CONTACT CHECK**:
    If prospect says STOP, REMOVE, UNSUBSCRIBE, OPT OUT, NO CONTACT, or END CONTACT → ALWAYS return False

    ---

    ## SAME-DAY PHONE CALL Detection

    **Detection criteria:**
    - Explicit same-day: "Call me today", "Ring me now"
    - No timing (defaults to same-day): "Can we talk?", "Ring me at 03001234567"
    - Explicit future: "Call me next week" → False

    **SAME DAY CONTEXT (CRITICAL EDGE CASE)**:
    If the thread shows scheduling for a specific day (ANY weekday), and prospect says "same day":

    Example threads:
    - Agent: "How about Friday at 2pm for a call?"
      Prospect: "How about 4pm same day?" → "same day" = Friday

    - Agent: "Let's talk on Thursday at 10am?"
      Prospect: "Same day works, but 3pm instead?" → "same day" = Thursday

    - Agent: "Tuesday at 1pm for a call?"
      Prospect: "Same day at 5pm is better" → "same day" = Tuesday

    → "same day" means WHATEVER day was previously mentioned in the conversation
    → This IS a same-day phone request (for that specific day)

    **Examples of SAME-DAY PHONE CALL (same_day_phone_request_detected=True)**:

    🔥 POSITIVE INTENT - Imperative + phone number + no time (CRITICAL):
    - "Call me at 949-555-1212." → True (direct command with number)
    - "Here's my cell: (415) 222-8899 — give me a ring." → True (shares number + requests call)
    - "Reach me by phone at 312.555.9900." → True (imperative with number)
    - "You can call me at +1 202-555-0199." → True (provides number + implicit request)
    - "Ring me at 03001234567" → True (direct request with number)
    - "My number is 555-1234 — call me" → True (shares number + explicit request)

    🔥 POSITIVE INTENT - Imperative + urgency words (ASAP, NOW, TODAY):
    - "Give me a call today — 949-555-1212." → True (explicit today + number)
    - "Call me ASAP at (310) 555-3344." → True (urgency = same-day)
    - "Ring me now: 646-555-1010." → True (immediate = same-day)
    - "If you're free today, call me: 415-555-2222." → True (today mentioned)
    - "Call me this afternoon — (949) 555-0007." → True (this afternoon = today)
    - "Call me today at 3pm" → True (explicit today)
    - "Phone call this afternoon?" → True (afternoon = today)
    - "Call me in 2 hours" → True (hours = today)
    - "Can we talk now?" → True (now = same-day)

    🔥 POSITIVE INTENT - Call request with no number (still same-day):
    - "Just call me." → True (direct imperative)
    - "Give me a ring when you can." → True (casual imperative)
    - "Can you call me today?" → True (explicit today)
    - "Call me" → True (direct request)
    - "Give me a ring" → True (direct request)
    - "Can you call me?" → True (explicit request from prospect)
    - "Can we talk?" → True (explicit request from prospect)
    - "Let's have a call" → True (explicit request)

    🔥 POSITIVE INTENT - Call request embedded in context:
    - "Email is tough — just give me a call at 917-555-1122." → True (context + request)
    - "I'm driving — call me at 310-555-8282." → True (context + imperative)
    - "I'm in meetings all day. Call me at 415-555-9191 and I'll pick up if I can." → True (context + same-day intent)
    - "Hard to type right now, call me: 555-1234" → True (context + request)
    - "Can't email back and forth — just ring me at 555-9999" → True (frustration + request)

    🔥 POSITIVE INTENT - Phone number in Email 1 fulfilling a same-day call request from Email 2:
    If Email 1 (newest) contains ONLY a phone number AND Email 2 (older) shows a same-day call request
    from the prospect (e.g., "Call me in an hour", "Call me now", "Ring me ASAP"), treat Email 1 as
    completing that request — the prospect is providing the number they asked to be called at.
    - Email 2: "Call me in an hour." / Email 1: "923-321-2233" → True (number fulfills prior call request)
    - Email 2: "Give me a ring when you can." / Email 1: "555-1234" → True (number fulfills prior call request)
    - Email 2: "Call me ASAP." / Email 1: "310-555-9999" → True (number fulfills prior urgency request)
    ⚠️ Only applies when Email 2 contains a genuine same-day call imperative from the prospect.
    Does NOT apply when Email 2 shows only generic interest ("Happy to chat") or future scheduling.

    Time-only (no date = defaults to today):
    - "maybe in a bit after 10am" → True
    - "after 10am" → True
    - "at 2pm" → True
    - "in a bit" → True
    - "around 3pm" → True
    - "at 11:30am" → True

    No timing specified WITH IMPERATIVE (defaults to same-day):
    - "Yes, please call me" → True (imperative: "please call me")
    - "Yes, call me" → True (imperative directive)
    - "Call me when you can" → True (imperative + casual timing)

    ⚠️ Note: Generic openness WITHOUT imperative = False (see negative examples below):
    - "Yes, I would be open to a call" → False (generic interest, no imperative)
    - "Happy to talk" → False (openness, no directive)
    - "I'm available for a call" → False (availability, no request)

    Today as PRIMARY option (with future fallback):
    - "If today is an option let me know as pretty clear today. Otherwise tomorrow at 10:30am" → True
    - "If today works let me know, otherwise tomorrow" → True
    - "Today would be great, or tomorrow if that doesn't work" → True
    - "I'm available today, but can do tomorrow as well" → True
    - "Pretty clear today. Otherwise tomorrow" → True

    🔥 CRITICAL CONTRAST - Imperative vs generic interest:
    - PROSPECT: "Yes, please call me" → TRUE (imperative: "please call me")
    - PROSPECT: "Yes, I would be open to a call" → FALSE (generic interest, no imperative)

    - PROSPECT: "Can you call me?" → TRUE (explicit request with question)
    - PROSPECT: "Happy to chat" → FALSE (generic openness, no request)

    - PROSPECT: "Call me at 555-1234" → TRUE (imperative + number)
    - PROSPECT: "My number is 555-1234" → FALSE (sharing info, no directive)

    - SENDER: "I can call you" / PROSPECT: "Yes" → FALSE (just agreeing to offer)
    - SENDER: "Would you be open to a call?" / PROSPECT: "Yes" → FALSE (just answering question)

    **Examples of NOT same-day (same_day_phone_request_detected=False)**:

    🚨 CRITICAL: Generic interest without urgency (scheduling flows, NOT "call me now"):
    - "Yes, I would be open to a call." → False (expressing openness, no imperative)
    - "Happy to chat." → False (general openness, no urgency)
    - "Open to a conversation." → False (willingness, no specific request)
    - "Let's talk." → False (vague, no urgency or timing)
    - "I'm interested" → False (general interest, no call request)
    - "Sure, sounds good" → False (general agreement, no explicit call request)

    🚨 CRITICAL: Future day/time explicitly specified (NOT today):
    - "Call me tomorrow." → False (explicit future day)
    - "Give me a ring Friday." → False (specific future day)
    - "Call me next week." → False (future timeframe)
    - "I'm free Thursday afternoon — call me then." → False (specific future day + time)
    - "203-923-0021. Thursday works." → False (phone + future day = NOT same-day)
    - "Let's schedule a call next week" → False (ONLY future, no today option)
    - "Can we talk on Monday?" (today is Saturday) → False (ONLY different day)
    - "Tomorrow at 10:30am" → False (ONLY tomorrow, no today mentioned)
    - "You can reach me at 949-555-1212 next week." → False (phone + future = NOT same-day)

    🚨 CRITICAL: Meeting/calendar scheduling intent (NOT immediate call):
    - "Send me your Calendly." → False (scheduling tool request)
    - "Let's schedule a call." → False (scheduling flow, not immediate)
    - "Can we set up a meeting next week?" → False (scheduling + future)
    - "Book something on my calendar." → False (calendar booking, not immediate call)

    🚨 CRITICAL: Phone number present but NO call instruction:
    - "My number is 949-555-1212." → False (just sharing number, no imperative)
    - "Here's my number: 555-1234" → False (sharing info, no "call me" directive)
    - "Text is better: 949-555-1212." → False (text preference, NOT call request)
    - "Feel free to call anytime" → False (generic permission, not specific request)

    🚨 CRITICAL: Prospect prefers TEXT communication (NOT a same-day call request):
    - "You can text me at that number to coordinate a call. Lmk what number that would be from." → False (coordinating via text, asking for agent's number — NOT requesting same-day call)
    - "Feel free to text me at 555-1234" → False (text preference, not a call request)
    - "Text works better for me, here's my number: 555-1234" → False (text preference, not call request)
    - "I prefer texts, you can reach me at 949-555-1212" → False (text is preferred channel, not a call directive)
    - "Shoot me a text at 555-1234 when you get a chance" → False (text to coordinate, not an immediate call request)
    - "Text me at (303) 949-2305 to set something up" → False (using text to schedule, not requesting call now)
    - "I'm better on text: 555-1234" → False (communication preference, not call request)
    - "Send me a text and we can go from there" → False (text-first coordination, not a phone call request)

    🚨 CRITICAL: Phone number in email signature or forwarded/pasted boilerplate:
    - "Cell 219-741-1849" appearing in a signature block → False (standard contact footer, not a request)
    - "Office 555-1111 | Cell 555-2222" in signature → False (contact directory format, not a request)
    - "Contact John Smith at john@co.com or by Cell (555) 123-4567" → False (directory/newsletter format, not a live request)
    - "Reach me at: Office 312-555-9900, Cell 312-555-1234" in signature → False (contact info listing)
    - Phone number appearing ONLY in the signature footer (after name/title/company/address block) → False

    🚨 CRITICAL: Prospect agreeing to SENDER's offer to call (NOT a prospect request):
    - Sender: "I can call you to discuss" / Prospect: "Yes" → False (sender offering, not prospect requesting)
    - Sender: "Would you like me to call you?" / Prospect: "Yes" → False (responding to sender's offer)
    - Sender: "I can call you if that's easier" / Prospect: "Sure" → False (sender offering to call)
    - Sender: "I can give you a call" / Prospect: "Okay" → False (not an explicit callback request)
    - Sender: "Happy to call your number" / Prospect: "Yes" → False (sender offering, prospect agreeing)

    🚨 CRITICAL: Sender offering to call (NOT prospect requesting):
    - "Happy to call your number if that's easiest" → False (sender offering)
    - "I can call you at 555-1234" → False (sender offering)
    - "Let me call you to discuss" → False (sender offering)
    - "I'll give you a ring" → False (sender will call, not prospect requesting callback)
    - "I'll call you tomorrow" → False (sender initiating, not prospect requesting)

    🚨 CRITICAL: Acknowledgments without explicit request:
    - "Yes" (in response to interest question) → False (just expressing interest, not requesting call)
    - "Okay" → False (acknowledgment, not a call request)

    **PHONE CALL INDICATORS**:
    - Direct: "call", "phone", "ring", "talk", "speak"
    - Imperatives: "call me", "ring me", "reach me", "give me a ring", "give me a call"
    - Urgency: "ASAP", "NOW", "TODAY", "THIS AFTERNOON", "IN A BIT", "WHEN YOU CAN"
    - Openness: "open to a call", "happy to talk", "happy to chat"
    - Duration: "15 minutes", "30 minutes", "quick chat", "quick sync"
    - Context clues: "email is tough", "can't type", "I'm driving", "hard to explain"

    **EXCLUDE**:
    - Video meetings with explicit mentions: "Zoom", "Teams", "video call", "video meeting", "Zoom link"
    - In-person meetings: "meet in person", "stop by office", "coffee meeting"
    - Email/message communication
    - Prospect saying THEY will call (not requesting callback)

    ---

    ## TIME CONTEXT ANALYSIS

    **DECISION LOGIC**:
    1. **WHO is requesting? (POSITIVE INTENT CHECK)**
       - Prospect uses IMPERATIVE: "Call me", "Ring me", "Reach me" → Continue to step 2
       - Prospect provides number with imperative: "Call me at 555-1234" → Continue to step 2 (STRONG POSITIVE INTENT)
       - Prospect uses urgency words: "Call me ASAP", "Ring me now" → Continue to step 2 (STRONG POSITIVE INTENT)
       - Prospect expresses generic interest: "I would be open to a call", "Happy to chat", "Let's talk" → FALSE (scheduling flow, not same-day)
       - Sender offering to call: "I can call you", "Let me call you" → FALSE (stop here)
       - Prospect just says "Yes" to sender's offer → FALSE (stop here)

    2. **Does prospect use an IMPERATIVE or REQUEST?** (call/ring/talk with directive)
       - Must be from PROSPECT, not sender
       - Imperative required: "Call me", "Ring me", "Can you call me?" → True
       - Generic openness: "Happy to chat", "Open to a conversation" → False (no imperative)
       - Phone number + imperative = STRONG POSITIVE INTENT
       - Phone number WITHOUT directive = just sharing info → False
       - EXCEPTION: If Email 1 contains only a phone number AND Email 2 contains a same-day call imperative
         from the prospect, treat Email 1 as fulfilling that prior request → True

    3. **Check timing and urgency:**
       - Urgency words: "ASAP", "NOW", "TODAY", "THIS AFTERNOON" → True (immediate = same-day)
       - Time-only ("at 10am", "after 2pm", "in a bit") with NO date → True (defaults to today)
       - "Today" mentioned → True (even if future fallback)
       - No timing with imperative → True (defaults to same-day)
       - ONLY future date ("tomorrow", "Friday", "next week") with NO today → False
       - Calendar/scheduling words: "Calendly", "schedule a call", "book something" → False

    Examples (True):
    - "Call me at 555-1234" → True (imperative + number)
    - "Give me a call today — 555-1234" → True (urgency + number)
    - "Call me ASAP" → True (urgency word)
    - "Ring me now" → True (urgency word)
    - "Just call me" → True (direct imperative)
    - "Can you call me?" → True (request question)
    - Email 2: "Call me in an hour." / Email 1: "923-321-2233" → True (number fulfills prior same-day request)

    Examples (False):
    - "I would be open to a call" → False (generic interest, no imperative)
    - "Happy to chat" → False (openness, no directive)
    - "Call me tomorrow" → False (future day)
    - "My number is 555-1234" → False (no directive)
    - "Send me your Calendly" → False (scheduling flow)
    - "203-923-0021. Thursday works" → False (phone + future day)

    **NOTIFICATION TYPE**:
    - "SAME_DAY_PHONE_CALL" if detected
    - "NONE" if not detected
    """

    # Input
    email_thread: str = dspy.InputField(
        desc="Full email thread conversation, with newest message first (Email 1 = most recent)"
    )

    today_day_name: str = dspy.InputField(
        desc="TODAY's day of the week (e.g., 'Monday', 'Tuesday', ..., 'Sunday'). Use this to validate same-day requests."
    )

    # Outputs
    same_day_phone_request_detected: bool = dspy.OutputField(
        desc="""True if prospect shows POSITIVE INTENT and requests a phone call TODAY.

        🔥 POSITIVE INTENT PATTERNS (Return True):
        1. Imperative + phone number: "Call me at 555-1234", "Ring me at (310) 555-9999"
        2. Urgency words: "Call me ASAP", "Ring me now", "Call me today", "this afternoon"
        3. Direct request: "Just call me", "Give me a ring when you can"
        4. Embedded context: "Email is tough — call me at 555-1234", "I'm driving — ring me"
        5. Time-only (no date): "after 10am", "at 2pm", "in a bit" (defaults to today)
        6. No timing + imperative: "call me" (defaults to same-day)
        7. Today as option: "If today works, otherwise tomorrow" (today offered)

        🚨 NEGATIVE PATTERNS (Return False):
        1. Generic interest (no imperative): "I would be open to a call", "Happy to chat", "Let's talk"
        2. Future timing only: "Call me tomorrow", "Give me a ring Friday", "203-923-0021. Thursday works"
        3. Calendar/scheduling: "Send me your Calendly", "Let's schedule a call", "Book something on my calendar"
        4. Phone without directive: "My number is 555-1234" (no "call me"), "Text is better: 555-1234"
        5. Sender offering: "I can call you" / Prospect: "Yes" (not a prospect request)
        6. Acknowledgment only: "Okay", "Yes" (no explicit call request)
        7. Email signature / contact footer: "Cell 219-741-1849" in signature block, "Office 555-1111 | Cell 555-2222", "Contact John at john@co.com or by Cell (555) 123-4567" (directory/newsletter format)

        Key rules:
        - Phone number + imperative = same-day (STRONG signal)
        - Urgency words (ASAP, NOW, TODAY) = same-day
        - Generic interest WITHOUT imperative = scheduling flow (False)
        - Phone number WITHOUT "call me" directive = just sharing info (False)
        - Time without date = today (unless explicit future date given)

        Examples (True):
        - "Call me at 949-555-1212" → True (imperative + number)
        - "Give me a call today — 555-1234" → True (urgency + number)
        - "Just call me" → True (direct imperative)
        - "Call me ASAP" → True (urgency word)

        Examples (False):
        - "I would be open to a call" → False (generic interest, no imperative)
        - "Call me tomorrow" → False (future day)
        - "Send me your Calendly" → False (scheduling flow)
        - "My number is 555-1234" → False (no directive)
        - "203-923-0021. Thursday works" → False (phone + future day)"""
    )

    ai_commentary: str = dspy.OutputField(
        desc="""Brief explanation of what was detected.

        Examples:
        - "Phone call request with explicit same-day timing ('call me now')"
        - "Phone call request with no timing specified - defaults to same-day"
        - "Phone call request for future date ('next week') - not same-day"
        """
    )

    notification_type: str = dspy.OutputField(
        desc="Exactly one of: 'SAME_DAY_PHONE_CALL' or 'NONE'."
    )
