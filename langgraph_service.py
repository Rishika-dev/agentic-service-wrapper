"""
LangGraph-style Service Implementation for Image Generation WITHOUT ReAct.

This service uses a LangChain tool `generate_image_tool` that wraps
`ImageGenerationService`, but calls it directly without any ReAct agent
orchestration. It is designed to be plugged into the Masumi flow:

- Input:  dict with 'prompt' and optional 'model_type' ('OPENAI' or 'DALLE')
- Output: object with `.raw` and `.json_dict` including `image_base64`
"""

from typing import Dict, Any, Optional
import os
import base64
from datetime import datetime
import hashlib

from langchain_core.tools import tool

from image_service import ImageGenerationService, ModelType
from blockfrost_service import BlockfrostService
from blockfrost_service import BlockfrostService


class LangGraphResult:
    """Simple result object compatible with Masumi completion flow."""

    def __init__(
        self,
        prompt: str,
        model_type: str,
        image_base64: str,
        ipfs_hash: Optional[str] = None,
    ):
        self.prompt = prompt
        self.model_type = model_type
        self.image_base64 = image_base64
        self.ipfs_hash = ipfs_hash

        # What Masumi /status will return - only IPFS hash if available, otherwise base64
        self.raw = ipfs_hash if ipfs_hash else image_base64

        # What Masumi payment completion will hash/store - only IPFS hash
        self.json_dict: Dict[str, Any] = {
            "task": "image_generation_tool_only",
            "prompt": prompt,
            "model_type": model_type,
        }
        if ipfs_hash:
            self.json_dict["ipfs_hash"] = ipfs_hash
        else:
            # Fallback to base64 if IPFS upload failed
            self.json_dict["image_base64"] = image_base64


class LangGraphService:
    """
    Service that uses a LangChain tool `generate_image_tool` for image generation.

    No ReAct agent orchestration - calls the tool directly.
    """

    def __init__(self, logger=None):
        self.logger = logger
        # ImageGenerationService uses OpenAI env configuration
        self.image_service = ImageGenerationService()
        # Initialize Blockfrost service (optional - will skip if not configured)
        try:
            self.blockfrost_service = BlockfrostService(logger=self.logger)
        except (ValueError, Exception) as e:
            if self.logger:
                self.logger.warning(f"Blockfrost service not available: {e}")
            self.blockfrost_service = None
        # Store last IPFS hash from tool execution
        self.last_ipfs_hash: Optional[str] = None
        # Create the tool (kept for consistency with LangChain tool pattern)
        self.generate_image_tool = self._create_tool()

    def _create_tool(self):
        """Create the image-generation tool (no ReAct agent)."""

        image_service = self.image_service
        logger = self.logger  # Capture logger for use in closure
        blockfrost_service = self.blockfrost_service  # Capture for use in closure
        service_instance = self  # Capture service instance to store IPFS hash

        @tool
        def generate_image_tool(prompt: str, model_type: str = "DALLE") -> str:
            """
            Generate an image based on the prompt and model_type.

            Args:
                prompt: Description of the image to generate.
                model_type: 'OPENAI' (Responses API with image_generation tool)
                            or 'DALLE' (dall-e-3 images API). Defaults to 'DALLE'.

            Returns:
                Base64-encoded PNG image as a string.
            """
            model_type_upper = (model_type or "DALLE").upper()
            if model_type_upper not in ("OPENAI", "DALLE"):
                raise ValueError("model_type must be either 'OPENAI' or 'DALLE'")

            mt: ModelType = model_type_upper  # type: ignore[assignment]

            print(f"[TOOL] Calling image generation service...")
            image_bytes = image_service.generate_image(prompt, mt)
            print(f"[TOOL] Image generation completed")

            # Save image to assets folder
            print(f"[SAVE] Saving image to assets folder...")
            assets_dir = os.path.join(os.path.dirname(__file__), "assets")
            os.makedirs(assets_dir, exist_ok=True)

            # Generate unique filename: timestamp + hash of prompt
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            filename = f"image_{timestamp}_{prompt_hash}.png"
            filepath = os.path.join(assets_dir, filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            print(f"[SAVE] Image saved to: {filepath}")
            if logger:
                logger.info(f"Saved generated image to: {filepath}")

            # Upload to IPFS via Blockfrost (if configured)
            ipfs_hash = None
            if blockfrost_service:
                print(f"[IPFS] Starting IPFS upload via Blockfrost...")
                try:
                    ipfs_hash = blockfrost_service.upload_and_pin(filepath)
                    service_instance.last_ipfs_hash = (
                        ipfs_hash  # Store for execute_task
                    )
                    print(f"[IPFS] Image uploaded to IPFS successfully!")
                    print(f"   IPFS Hash: {ipfs_hash}")
                    if logger:
                        logger.info(f"Uploaded image to IPFS: {ipfs_hash}")

                    # Delete local file after successful IPFS upload
                    print(f"[CLEANUP] Deleting local file...")
                    try:
                        os.remove(filepath)
                        print(f"[CLEANUP] Local file deleted: {filepath}")
                        if logger:
                            logger.info(
                                f"Deleted local file after IPFS upload: {filepath}"
                            )
                    except Exception as delete_error:
                        print(
                            f"[CLEANUP] WARNING: Failed to delete local file: {delete_error}"
                        )
                        if logger:
                            logger.warning(
                                f"Failed to delete local file {filepath}: {delete_error}"
                            )
                except Exception as e:
                    print(f"[IPFS] ERROR: Failed to upload to IPFS: {e}")
                    if logger:
                        logger.warning(f"Failed to upload to IPFS: {e}")
                    service_instance.last_ipfs_hash = None
            else:
                print(
                    f"[IPFS] WARNING: Blockfrost service not configured, skipping IPFS upload"
                )

            result = base64.b64encode(image_bytes).decode("utf-8")
            print(f"[TOOL] Image generation and IPFS upload flow completed!")
            return result

        return generate_image_tool

    async def execute_task(self, input_data: dict) -> LangGraphResult:
        """
        Execute image-generation task by calling generate_image_tool directly.

        Args:
            input_data: Dictionary containing:
                - 'prompt': Text description of the image (required)
                - 'model_type': Optional 'OPENAI' or 'DALLE' (default: 'DALLE')

        Returns:
            LangGraphResult with base64-encoded PNG image.
        """
        prompt = input_data.get("prompt", "")
        model_type_str = (input_data.get("model_type") or "DALLE").upper()

        print(f"[MASUMI] Starting image generation task...")
        print(f"   Model Type: {model_type_str}")
        print(f"   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        if self.logger:
            self.logger.info(
                f"Image generation task: model_type={model_type_str}, "
                f"prompt='{prompt[:80]}{'...' if len(prompt) > 80 else ''}'"
            )

        # Call the tool directly (no ReAct agent needed)
        # The @tool decorator wraps it, so we need to use .invoke() method
        # Reset IPFS hash before calling tool
        self.last_ipfs_hash = None
        image_base64 = self.generate_image_tool.invoke({
            "prompt": prompt,
            "model_type": model_type_str
        })

        # Get IPFS hash from tool execution (stored by tool)
        ipfs_hash = self.last_ipfs_hash

        if ipfs_hash:
            print(f"[MASUMI] Task completed successfully! IPFS Hash: {ipfs_hash}")
        else:
            print(f"[MASUMI] WARNING: Task completed but IPFS upload was not available")

        if self.logger:
            self.logger.info("Image generation completed")

        return LangGraphResult(
            prompt=prompt,
            model_type=model_type_str,
            image_base64=image_base64,
            ipfs_hash=ipfs_hash,
        )


# Optional: simple manual test
async def test_langgraph_service():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    service = LangGraphService()
    input_data = {
        "prompt": "Generate an image of a gray tabby cat hugging an otter with an orange scarf",
        "model_type": "DALLE",
    }
    result = await service.execute_task(input_data)
    print("Got image_base64 length:", len(result.image_base64))


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_langgraph_service())
