import os
import uvicorn
import uuid
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from masumi.config import Config
from masumi.payment import Payment
from logging_config import setup_logging
import cuid2
import base64
from datetime import datetime
import hashlib

from image_service import ImageGenerationService, ModelType
from blockfrost_service import BlockfrostService

# region congif
# Configure logging
logger = setup_logging()

# Load environment variables
load_dotenv(override=True)

# Retrieve API Keys and URLs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL", "")
PAYMENT_API_KEY = os.getenv("PAYMENT_API_KEY", "")
NETWORK = os.getenv("NETWORK", "preview")

logger.info("Starting application with configuration:")
logger.info(f"PAYMENT_SERVICE_URL: {PAYMENT_SERVICE_URL}")


# validate critical environment variables at startup
def validate_url(url: str, name: str) -> str:
    """Validate that a URL is properly formatted"""
    if not url:
        return f"{name} is not set"

    # check if it starts with http:// or https://
    if not url.startswith(("http://", "https://")):
        return f"{name} must start with 'https://' or 'http://' (got: '{url}')"

    # parse URL to check if it's valid
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return f"{name} is not a valid URL format (got: '{url}')"
    except Exception:
        return f"{name} is not a valid URL format (got: '{url}')"

    return ""  # no error


def validate_environment():
    """Validate that all required environment variables are set"""
    errors = []

    agent_id = os.getenv("AGENT_IDENTIFIER", "").strip()
    if not agent_id:
        errors.append("AGENT_IDENTIFIER is not set")
    elif agent_id == "REPLACE":
        errors.append(
            "AGENT_IDENTIFIER is set to placeholder 'REPLACE' - please set a real agent identifier"
        )

    # validate payment service URL format
    url_error = validate_url(PAYMENT_SERVICE_URL, "PAYMENT_SERVICE_URL")
    if url_error:
        errors.append(url_error)

    if not PAYMENT_API_KEY:
        errors.append("PAYMENT_API_KEY is not set")

    if not NETWORK:
        errors.append("NETWORK is not set")

    if errors:
        logger.error("Critical environment variable validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        logger.error("Please fix these configuration issues before starting the server")
        return False

    logger.info("Environment validation passed")
    return True


# run validation but don't fail startup (for debugging)
validation_passed = validate_environment()
if not validation_passed:
    logger.warning(
        "Starting server despite configuration errors - some endpoints may not work properly"
    )

# Initialize FastAPI
app = FastAPI(
    title="API following the Masumi API Standard",
    description="Masumi-compliant agent service with payment integration",
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────────────────────
# region Temporary in-memory job store
# DO NOT USE IN PRODUCTION)
# ─────────────────────────────────────────────────────────────────────────────
jobs = {}
payment_instances = {}

# track server start time for uptime calculation
server_start_time = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# region Initialize Masumi Payment Config
# ─────────────────────────────────────────────────────────────────────────────
# config will be created in start_job to allow proper validation


# ─────────────────────────────────────────────────────────────────────────────
# region Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────
class InputDataItem(BaseModel):
    key: str
    value: str


class StartJobRequest(BaseModel):
    input_data: list[InputDataItem]

    class Config:
        json_schema_extra = {
            "example": {
                "input_data": [
                    {
                        "key": "prompt",
                        "value": "Generate an image of a gray tabby cat hugging an otter",
                    },
                    {"key": "model_type", "value": "DALLE"},
                ]
            }
        }


class ProvideInputRequest(BaseModel):
    job_id: str


class ImageRequest(BaseModel):
    prompt: str
    model_type: ModelType


# ─────────────────────────────────────────────────────────────────────────────
# region Task Execution THIS IS THE MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
async def execute_agentic_task(input_data: dict) -> object:
    """Execute task using the LangGraph image-generation agent."""
    from langgraph_service import LangGraphService

    logger.info(f"starting task with input: {input_data}")
    service = LangGraphService(logger=logger)
    result = await service.execute_task(input_data)
    logger.info("task completed successfully")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# region 1) Start Job (MIP-003: /start_job)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/start_job")
async def start_job(data: StartJobRequest):
    """Initiates a job and creates a payment request"""
    print(f"Received data: {data}")
    print(f"Received data.input_data: {data.input_data}")
    try:
        job_id = str(uuid.uuid4())

        # validate required environment variables
        agent_identifier = os.getenv("AGENT_IDENTIFIER", "").strip()
        if not agent_identifier or agent_identifier == "REPLACE":
            logger.error(
                "AGENT_IDENTIFIER environment variable is missing or set to placeholder 'REPLACE'"
            )
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: AGENT_IDENTIFIER not properly configured. Please contact administrator.",
            )

        # validate payment service URL format
        url_error = validate_url(PAYMENT_SERVICE_URL, "PAYMENT_SERVICE_URL")
        if url_error:
            logger.error(f"PAYMENT_SERVICE_URL validation failed: {url_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Server configuration error: {url_error}. Please contact administrator.",
            )

        if not PAYMENT_API_KEY:
            logger.error("PAYMENT_API_KEY environment variable is missing")
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: PAYMENT_API_KEY not configured. Please contact administrator.",
            )

        # generate identifier_from_purchaser internally using cuid2
        identifier_from_purchaser = cuid2.Cuid().generate()
        logger.info(f"Generated identifier_from_purchaser: {identifier_from_purchaser}")

        # convert input_data array to dict for internal processing
        input_data_dict = {item.key: item.value for item in data.input_data}

        # validate required image-generation input
        if "prompt" not in input_data_dict:
            logger.error("Required field 'prompt' missing from input_data")
            raise HTTPException(
                status_code=400,
                detail="Bad Request: 'prompt' is required in input_data array",
            )

        # Optional model_type; default handled by AgenticService but we can log it
        model_type = (input_data_dict.get("model_type") or "DALLE").upper()
        if model_type not in ("OPENAI", "DALLE"):
            logger.error(f"Invalid model_type provided: {model_type}")
            raise HTTPException(
                status_code=400,
                detail="Bad Request: 'model_type' must be either 'OPENAI' or 'DALLE' if provided",
            )

        # Log the prompt (truncate if too long)
        input_text = input_data_dict.get("prompt", "")
        truncated_input = (
            input_text[:100] + "..." if len(input_text) > 100 else input_text
        )
        logger.info(f"Received job request with input: '{truncated_input}'")
        logger.info(f"Starting job {job_id} with agent {agent_identifier}")

        # Define payment amounts
        payment_amount = int(os.getenv("PAYMENT_AMOUNT", "1000000"))  # 1 ADA
        payment_unit = os.getenv("PAYMENT_UNIT", "lovelace")  # Default lovelace

        logger.info(f"Using payment amount: {payment_amount} {payment_unit}")

        # create config after validation
        config = Config(
            payment_service_url=PAYMENT_SERVICE_URL, payment_api_key=PAYMENT_API_KEY
        )

        # Create a payment request using Masumi
        payment = Payment(
            agent_identifier=agent_identifier,
            # amounts=amounts,
            config=config,
            identifier_from_purchaser=identifier_from_purchaser,
            input_data=input_data_dict,
            network=NETWORK,
        )

        logger.info("Creating payment request...")
        payment_request = await payment.create_payment_request()
        payment_id = payment_request["data"]["blockchainIdentifier"]
        payment.payment_ids.add(payment_id)
        logger.info(f"Created payment request with ID: {payment_id}")

        # Store job info (Awaiting payment)
        jobs[job_id] = {
            "status": "awaiting_payment",
            "payment_status": "pending",
            "payment_id": payment_id,
            "input_data": input_data_dict,
            "result": None,
            "identifier_from_purchaser": identifier_from_purchaser,
        }

        async def payment_callback(payment_id: str):
            await handle_payment_status(job_id, payment_id)

        # Start monitoring the payment status
        payment_instances[job_id] = payment
        logger.info(f"Starting payment status monitoring for job {job_id}")
        await payment.start_status_monitoring(payment_callback)

        # Get SELLER_VKEY from environment
        seller_vkey = os.getenv("SELLER_VKEY", "")
        if not seller_vkey:
            logger.error("SELLER_VKEY environment variable is missing")
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: SELLER_VKEY not configured. Please contact administrator.",
            )

        # Return the response in the format expected by the /purchase endpoint
        # Include both the original fields and the extended fields
        return {
            # Original fields for backward compatibility
            "job_id": job_id,
            "payment_id": payment_id,
            # Extended fields for /purchase endpoint
            "identifierFromPurchaser": identifier_from_purchaser,
            "network": NETWORK,
            "sellerVkey": seller_vkey,
            "paymentType": "Web3CardanoV1",
            "blockchainIdentifier": payment_id,
            "submitResultTime": str(payment_request["data"]["submitResultTime"]),
            "unlockTime": str(payment_request["data"]["unlockTime"]),
            "externalDisputeUnlockTime": str(
                payment_request["data"]["externalDisputeUnlockTime"]
            ),
            "agentIdentifier": agent_identifier,
            "inputHash": payment_request["data"]["inputHash"],
        }
    except HTTPException:
        # re-raise HTTP exceptions (our custom errors)
        raise
    except ValueError as e:
        logger.error(f"Value error in start_job: {str(e)}", exc_info=True)
        if "PAYMENT_AMOUNT" in str(e):
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: Invalid PAYMENT_AMOUNT value. Please contact administrator.",
            )
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except KeyError as e:
        logger.error(f"Missing required field in request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in start_job: {str(e)}", exc_info=True)
        # check if it's a masumi payment service error
        if "Network error" in str(e) or "payment" in str(e).lower():
            raise HTTPException(
                status_code=502,
                detail="Payment service unavailable. Please try again later or contact administrator.",
            )
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please contact administrator.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# region 2) Process Payment and Execute AI Task
# ─────────────────────────────────────────────────────────────────────────────
async def handle_payment_status(job_id: str, payment_id: str) -> None:
    """Executes task after payment confirmation"""
    try:
        logger.info(
            f"Payment {payment_id} completed for job {job_id}, executing task..."
        )

        # Update job status to running
        jobs[job_id]["status"] = "running"
        input_data = jobs[job_id]["input_data"]
        logger.info(f"Input data: {input_data}")

        # Execute the AI task
        result = await execute_agentic_task(input_data)
        result_dict = result.json_dict  # type: ignore
        logger.info(f"task completed for job {job_id}")

        # Mark payment as completed on Masumi
        # Use a shorter string for the result hash
        await payment_instances[job_id].complete_payment(payment_id, result_dict)
        logger.info(f"Payment completed for job {job_id}")

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["payment_status"] = "completed"
        jobs[job_id]["result"] = result

        # Stop monitoring payment status
        if job_id in payment_instances:
            payment_instances[job_id].stop_status_monitoring()
            del payment_instances[job_id]
    except Exception as e:
        logger.error(
            f"Error processing payment {payment_id} for job {job_id}: {str(e)}",
            exc_info=True,
        )
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

        # Still stop monitoring to prevent repeated failures
        if job_id in payment_instances:
            payment_instances[job_id].stop_status_monitoring()
            del payment_instances[job_id]


# ─────────────────────────────────────────────────────────────────────────────
# region 3) Check Job and Payment Status (MIP-003: /status)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status")
async def get_status(job_id: str):
    """Retrieves the current status of a specific job"""
    logger.info(f"Checking status for job {job_id}")
    if job_id not in jobs:
        logger.warning(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Check latest payment status if payment instance exists
    if job_id in payment_instances:
        try:
            status = await payment_instances[job_id].check_payment_status()
            job["payment_status"] = status.get("data", {}).get("status")
            logger.info(
                f"Updated payment status for job {job_id}: {job['payment_status']}"
            )
        except ValueError as e:
            logger.warning(f"Error checking payment status: {str(e)}")
            job["payment_status"] = "unknown"
        except Exception as e:
            logger.error(f"Error checking payment status: {str(e)}", exc_info=True)
            job["payment_status"] = "error"

    result_data = job.get("result")
    result = result_data.raw if result_data and hasattr(result_data, "raw") else None

    return {
        "job_id": job_id,
        "status": job["status"],
        "payment_status": job["payment_status"],
        "result": result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# region 4) Check Server Availability (MIP-003: /availability)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/availability")
async def check_availability():
    """Checks if the server is operational"""
    current_time = time.time()
    uptime_seconds = int(current_time - server_start_time)

    return {
        "status": "available",
        "uptime": uptime_seconds,
        "message": "Server operational.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# region 5) Retrieve Input Schema (MIP-003: /input_schema)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/input_schema")
async def input_schema():
    """
    Returns the expected input schema for the /start_job endpoint.
    Fulfills MIP-003 /input_schema endpoint.
    """
    return {
        "input_data": [
            {
                "id": "prompt",
                "type": "string",
                "name": "Image Prompt",
                "data": {
                    "description": "Text description of the image you want to generate",
                    "placeholder": "Generate an image of a gray tabby cat hugging an otter",
                },
            },
            {
                "id": "model_type",
                "type": "string",
                "name": "Model Type",
                "data": {
                    "description": "Which image model to use: 'OPENAI' (Responses API with image_generation tool) or 'DALLE' (dall-e-3 images API). Defaults to 'DALLE' if omitted.",
                    "placeholder": "DALLE",
                },
            },
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# region 6) Health Check
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Returns the health of the server.
    """
    return {"status": "healthy"}


@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    """
    Simple image generation endpoint.

    - model_type="OPENAI": uses the Responses API with image_generation tool
    - model_type="DALLE": uses the Images API (dall-e-3)
    """
    try:
        # Check if API key is set
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500, detail="OPENAI_API_KEY environment variable is not set"
            )

        # ImageGenerationService uses OpenAI env configuration (OPENAI_API_KEY etc.)
        print(f"[API] Starting image generation request...")
        print(
            f"   Prompt: {request.prompt[:100]}{'...' if len(request.prompt) > 100 else ''}"
        )
        print(f"   Model Type: {request.model_type}")

        service = ImageGenerationService()
        image_bytes = service.generate_image(request.prompt, request.model_type)

        # Save image to assets folder
        print(f"[SAVE] Saving image to assets folder...")
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Generate unique filename: timestamp + hash of prompt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()[:8]
        filename = f"image_{timestamp}_{prompt_hash}.png"
        filepath = os.path.join(assets_dir, filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        print(f"[SAVE] Image saved to: {filepath}")
        logger.info(f"Saved generated image to: {filepath}")

        # Upload to IPFS via Blockfrost (if configured)
        ipfs_hash = None
        print(f"[IPFS] Starting IPFS upload via Blockfrost...")
        try:
            blockfrost_service = BlockfrostService(logger=logger)
            ipfs_hash = blockfrost_service.upload_and_pin(filepath)
            print(f"[IPFS] Image uploaded to IPFS successfully!")
            print(f"   IPFS Hash: {ipfs_hash}")
            logger.info(f"Uploaded image to IPFS: {ipfs_hash}")

            # Delete local file after successful IPFS upload
            print(f"[CLEANUP] Deleting local file...")
            try:
                os.remove(filepath)
                print(f"[CLEANUP] Local file deleted: {filepath}")
                logger.info(f"Deleted local file after IPFS upload: {filepath}")
            except Exception as delete_error:
                print(f"[CLEANUP] WARNING: Failed to delete local file: {delete_error}")
                logger.warning(
                    f"Failed to delete local file {filepath}: {delete_error}"
                )
        except (ValueError, Exception) as e:
            print(f"[IPFS] ERROR: Blockfrost IPFS upload not available: {e}")
            logger.warning(f"Blockfrost IPFS upload not available: {e}")

        # Return only IPFS hash if available, otherwise return base64 as fallback
        if ipfs_hash:
            print(f"[API] Request completed successfully! Returning IPFS hash.")
            return {
                "ipfs_hash": ipfs_hash,
                "format": "png",
                "model_type": request.model_type,
                "saved_path": f"assets/{filename}",
            }
        else:
            # Fallback to base64 if IPFS upload failed
            print(f"[API] WARNING: IPFS upload failed, returning base64 as fallback")
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            return {
                "image_base64": image_b64,
                "format": "png",
                "model_type": request.model_type,
                "saved_path": f"assets/{filename}",
            }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        # configuration / API key errors
        logger.error(f"ValueError in generate_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating image: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image: {type(e).__name__}: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# region Main Logic if Called as a Script
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Start the FastAPI server with Masumi integration."""
    print("Starting FastAPI server with Masumi integration...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "api":
        print("Starting FastAPI server with Masumi integration...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        main()
