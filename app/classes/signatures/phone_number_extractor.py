import dspy


class PhoneNumberExtractor(dspy.Signature):
    """Extract and detect phone numbers from prospect messages during meeting scheduling.

    This signature handles two critical functions:
    1. **EXTRACTION**: Detect and extract phone numbers already provided by the prospect
    2. **REQUEST DECISION**: Determine if we should request a phone number (with anti-repetition safeguards)

    **PHONE NUMBER VARIATIONS TO DETECT:**

    US Formats:
    - Standard: 555-123-4567
    - Parentheses: (555) 123-4567
    - Dots: 555.123.4567
    - Spaces: 555 123 4567
    - No formatting: 5551234567
    - With extension: 555-123-4567 ext. 123

    International Formats:
    - +1-555-123-4567 (US with country code)
    - +44 20 7123 4567 (UK)
    - +91 98765 43210 (India)
    - +86 138 0013 8000 (China)
    - +61 2 1234 5678 (Australia)
    - +49 30 12345678 (Germany)
    - +33 1 42 86 82 00 (France)

    Contextual Phrases to Look For:
    - "My number is..."
    - "You can reach me at..."
    - "Call me at..."
    - "My phone is..."
    - "My mobile is..."
    - "Contact me on..."
    - "Phone: ..."
    - "Mobile: ..."
    - "Tel: ..."
    - "Cell: ..."

    **EXTRACTION RULES:**
    1. Scan the ENTIRE email thread from start to finish
    2. Look for phone numbers in PROSPECT messages only (not agent messages)
    3. Detect all common formats listed above
    4. Extract the most recently provided number if multiple exist
    5. EXTRACT phone numbers from prospect email signatures if the signature is signed with the prospect's name (this is a personal contact number)
    6. DO NOT extract generic company main lines, shared team numbers, or phone numbers in legal disclaimers


    **ANTI-REPETITION RULES:**
    NEVER request a phone number if:
    1. Agent has already asked in ANY previous message
    2. Prospect has already provided in ANY previous message
    3. Phone number exists ANYWHERE in the thread from prospect

    Look for previous requests like:
    - "Can I get your phone number"
    - "What's your phone number"
    - "May I have your phone number"
    - "Could I get your contact number"
    - "Best number to reach you"
    """

    # Input Fields
    email_thread = dspy.InputField(
        desc="The full email thread showing the entire conversation history between agent and prospect"
    )
    prospect_name = dspy.InputField(
        desc="Prospect's name for context"
    )
    sender_name = dspy.InputField(
        desc="Agent's name"
    )
    meeting_context = dspy.InputField(
        desc="Context about the meeting being scheduled (time, date, purpose, status)"
    )
    is_meeting_scenario = dspy.InputField(
        desc="Boolean indicating if this is a meeting booking scenario",
        default=False
    )

    # Output Fields
    thought = dspy.OutputField(
        desc="""Step-by-step analysis:

        **STEP 1 - MEETING SCENARIO CHECK**:
        - Is this a meeting booking/scheduling conversation?
        - Is is_meeting_scenario flag True?

        **STEP 2 - PHONE NUMBER DETECTION IN THREAD**:
        - Scan the ENTIRE email thread for phone numbers
        - Check all variations: US formats (555-123-4567, (555) 123-4567, 555.123.4567)
        - Check international formats: +1-555-123-4567, +44 20 7123 4567, etc.
        - Check preserved hyperlinks: "847.456.8329 (tel:847.456.8329)" → Extract: 847.456.8329
        - Check contextual phrases: "my number is", "call me at", "reach me at"
        - Check the prospect's email signature block (e.g., name, title, phone on separate lines at the bottom)
        - Who provided the phone number? (Prospect or Agent)
        - Where in the thread? (Body text, or prospect's signature block)

        **STEP 3 - PREVIOUS REQUEST CHECK**:
        - Has the agent already asked for phone number in ANY previous message?
        - Look for phrases: "Can I get your phone number", "What's your phone number", "May I have your phone number"
        - Check ALL agent messages in the thread

        **STEP 4 - DECISION**:
        - If phone number found from prospect → Extract it, do NOT request
        - If agent already asked before → Do NOT request again
        - If neither found AND is meeting scenario → Should request
        - Otherwise → Do NOT request

        Provide your reasoning for each step.
        """
    )

    raw_phone_number = dspy.OutputField(
        desc="""The phone number extracted from the prospect's messages (RAW format as provided).

        **EXTRACTION INSTRUCTIONS**:
        1. Return the phone number EXACTLY as the prospect provided it
        2. Keep formatting (dashes, dots, parentheses, spaces) but REMOVE the "+" prefix from country codes
        3. **Handle preserved hyperlinks**: Extract number from "phone (tel:phone)" format
        4. Examples:
           - "My number is 555-123-4567" → Return: "555-123-4567"
           - "Call me at +1-555-123-4567" → Return: "1-555-123-4567"
           - "You can reach me on (555) 123-4567" → Return: "(555) 123-4567"
           - "My mobile: +44 20 7123 4567" → Return: "44 20 7123 4567"
           - "Call me at 404.996.8409" → Return: "404.996.8409"
           - "847.456.8329 (tel:847.456.8329)" → Return: "847.456.8329"
           - "555-1234 (tel:555-1234)" → Return: "555-1234"

        **IF NO PHONE NUMBER FOUND**:
        - Return empty string: ""

        **IF MULTIPLE NUMBERS**:
        - Return the most recent one from prospect

        **DO NOT EXTRACT**:
        - Phone numbers from agent messages
        - Generic company main lines or shared team numbers (e.g., "Call our office at...")
        - Phone numbers in legal disclaimers or footers unrelated to the prospect
        - Random numbers that aren't phone numbers

        **DO EXTRACT**:
        - Phone numbers in the prospect's personal signature block (signed with their name)

        **IMPORTANT**: This is the RAW phone number as the prospect wrote it.
        The system will automatically clean and format it for URL usage.

        Return the extracted phone number as a string, or "" if none found.
        """
    )
