import dspy


class DelegationDetector(dspy.Signature):
    """
    Detect if an email thread contains a delegation or referral to another person WITH explicit contact information.

    **CRITICAL:** Only detect delegation when EXPLICIT contact information is provided (email address, phone number, or specific person to contact).

    A delegation/referral occurs when the prospect:
    - Provides another person's EMAIL or PHONE number
    - Explicitly says to contact someone else WITH their contact details
    - Refers the agent to a specific decision-maker WITH contact info

    Examples of DELEGATION (has_delegation=True):
    - "Please contact Sarah at sarah@company.com"  ✅ Has email
    - "John handles this, reach out to him: john@company.com"  ✅ Has email
    - "Contact my manager at zohaibnadeem0566@gmail.com"  ✅ Has email
    - "Reach out to Lindsey Graham at hr.support@occ-ortho.com"  ✅ Has email

    Examples of NOT DELEGATION (has_delegation=False):
    - "I'm out of office until Monday"  ❌ No referral
    - "Jessica will handle my emails while I'm out"  ❌ No contact info (OOO auto-reply)
    - "I'm forwarding this to our CTO"  ❌ No specific contact
    - "Let me check with my team"  ❌ No specific contact
    - "Mike is the right person" ❌ No contact info provided
    - "Not interested"  ❌ Rejection, no referral
    """

    email_thread = dspy.InputField(
        desc="The complete email thread to analyze for delegation/referral"
    )

    has_delegation: bool = dspy.OutputField(
        desc="True if email contains delegation/referral with contact info, False otherwise."
    )

    delegate_name = dspy.OutputField(
        desc="The name of the person being referred to, if delegation detected. Empty string if no delegation."
    )

    delegate_email = dspy.OutputField(
        desc="The email of the person being referred to, if provided. Empty string if not provided."
    )

    reasoning = dspy.OutputField(
        desc="Brief explanation of why this is or isn't a delegation (1-2 sentences)"
    )
