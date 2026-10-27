"""
FastAPI application for the Email Classifier Interview Project.

This is a simplified, standalone version of the RevReply Email Classifier.
It demonstrates a production-ready API endpoint backed by an AI classifier.
"""
import os
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import dspy

from app.version import __version__
from app.classes.models.revreplyclassifier import RevReplyClassifierRequest
from app.classes.modules.classification.classifier_optimized import RevReplyClassifier

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global classifier instance (loaded at startup)
GLOBAL_CLASSIFIER: RevReplyClassifier = None

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key: str = Depends(api_key_header)):
    """Validate API key from request header."""
    expected_key = os.getenv("API_KEY", "test-api-key-12345")
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads the classifier model at startup.
    """
    global GLOBAL_CLASSIFIER

    logger.info("🚀 Starting Email Classifier API...")
    logger.info(f"Version: {__version__}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'unknown')}")

    # Initialize Azure OpenAI LM
    try:
        azure_lm = dspy.LM(
            "azure/gpt-5-mini",
            api_key=os.getenv("AZURE_API_KEY"),
            api_base=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            max_tokens=16000,
            temperature=1.0,
            reasoning_effort="medium",
            cache=True
        )
        logger.info("✅ Azure OpenAI LM initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Azure OpenAI: {e}")
        azure_lm = None

    # Load classifier
    try:
        GLOBAL_CLASSIFIER = RevReplyClassifier(lm=azure_lm, cache=True)
        logger.info("✅ Classifier loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load classifier: {e}")
        GLOBAL_CLASSIFIER = None

    yield  # Application runs here

    # Cleanup on shutdown
    logger.info("🛑 Shutting down Email Classifier API...")


# Initialize FastAPI app
app = FastAPI(
    title="Email Classifier API",
    description="AI-powered email classification system for prospect intent detection",
    version=__version__,
    lifespan=lifespan
)


@app.get("/")
def root():
    """Root endpoint - API information."""
    return {
        "service": "Email Classifier API",
        "version": __version__,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "classify": "/api/classify",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    classifier_status = "ok" if GLOBAL_CLASSIFIER is not None else "failed"
    return {
        "status": "healthy" if classifier_status == "ok" else "degraded",
        "version": __version__,
        "classifier": classifier_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/classify")
def classify(
    request: RevReplyClassifierRequest,
    req: Request,
    api_key: str = Depends(get_api_key),
):
    """
    Classify an email thread to determine prospect intent.

    **Classification Categories:**
    - `INTERESTED` - Prospect shows interest
    - `INFORMATION REQUEST` - Asking questions
    - `NOT INTERESTED` - Clear rejection
    - `CHECK BACK LATER` - Not ready now, check later
    - `SOFT OBJECTION` - Hesitation or concern
    - `MEETING REQUEST` - Requesting to schedule
    - `MEETING BOOKED` - Meeting confirmed
    - `DELEGATED` - Forwarded to someone else
    - `DO NOT CONTACT` - Explicit opt-out
    - `OUT OF OFFICE` - Automated OOO response
    - `WRONG PERSON` - Misdirected email
    - `REQUEST COMPLETE` - Task completed
    - `AUTOMATIC REPLY` - System-generated
    - `THREAD COMPLETE` - Conversation concluded

    **Request Body:**
    ```json
    {
        "email_thread": "Email conversation text...",
        "thread_id": "optional-thread-id",
        "user_id": 123,
        "persona_id": 456,
        "ooo_toggle": false,
        "check_same_day_phone_request": false,
        "disqualification_criteria": ""
    }
    ```

    **Response:**
    ```json
    {
        "prospect_classification": "INTERESTED",
        "reasoning": "Explanation of classification...",
        "lm_usage": {
            "user_id": 123,
            "persona_id": 456,
            "classifier": {
                "completion_tokens": 50,
                "prompt_tokens": 200,
                "total_tokens": 250
            }
        },
        "total_tokens": {
            "input": 200,
            "output": 50
        },
        "phone_number": "",
        "same_day_phone_request_detected": false
    }
    ```
    """
    try:
        request_start_time = datetime.now(timezone.utc)

        # Check if classifier is available
        if GLOBAL_CLASSIFIER is None:
            logger.error("Classifier not available - failed to load at startup")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Classifier service unavailable"
            )

        # Log request
        logger.info(
            f"Classification request: thread_id={request.thread_id}, "
            f"user_id={request.user_id}, persona_id={request.persona_id}"
        )

        # Run classification
        response = GLOBAL_CLASSIFIER(
            email_thread=request.email_thread,
            user_id=request.user_id,
            persona_id=request.persona_id,
            ooo_toggle=request.ooo_toggle,
            check_same_day_phone_request=request.check_same_day_phone_request,
            disqualification_criteria=request.disqualification_criteria,
        )

        # Extract results
        classification = response.classification
        reasoning = response.reasoning
        phone_number = getattr(response, 'phone_number', "")
        same_day_phone_request_detected = getattr(
            response, 'same_day_phone_request_detected', False
        )

        # Capture LM usage
        classifier_lm_usage = (
            response.get_lm_usage() if hasattr(response, 'get_lm_usage') else {}
        )

        lm_usage = {
            "user_id": request.user_id,
            "persona_id": request.persona_id,
            "classifier": classifier_lm_usage,
        }

        # Get total tokens
        total_tokens = (
            response.total_tokens if hasattr(response, 'total_tokens')
            else {"input": 0, "output": 0}
        )

        # Log response
        logger.info(
            f"Classification complete: {classification} "
            f"(tokens: {total_tokens})"
        )

        return {
            "prospect_classification": classification,
            "reasoning": reasoning,
            "lm_usage": lm_usage,
            "total_tokens": total_tokens,
            "phone_number": phone_number,
            "same_day_phone_request_detected": same_day_phone_request_detected,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc) if os.getenv("ENVIRONMENT") == "local" else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
