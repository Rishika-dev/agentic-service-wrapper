import base64
from typing import Optional

from image_service import ImageGenerationService, ModelType


class ServiceResult:
    """Result object compatible with Masumi completion payloads"""

    def __init__(
        self,
        prompt: str,
        model_type: str,
        image_base64: str,
    ):
        self.prompt = prompt
        self.model_type = model_type
        self.image_base64 = image_base64

        # raw payload returned to /status
        self.raw = {
            "task": "image_generation",
            "prompt": prompt,
            "model_type": model_type,
            "image_base64": image_base64,
        }

        # json_dict used when completing Masumi payments
        self.json_dict = self.raw


class AgenticService:
    """Agentic service that performs image generation via OpenAI / DALL·E."""

    def __init__(self, logger=None, api_key: Optional[str] = None):
        self.logger = logger
        # ImageGenerationService now uses OpenAI env configuration (OPENAI_API_KEY etc.)
        # The api_key parameter is kept for backward compatibility but not used
        self.image_service = ImageGenerationService()

    async def execute_task(self, input_data: dict) -> ServiceResult:
        """
        Execute image generation task.

        Expected input_data:
          - 'prompt' (str): required
          - 'model_type' (str): 'OPENAI' or 'DALLE' (default: 'DALLE')
        """
        prompt = input_data.get("prompt", "")
        model_type_str = (input_data.get("model_type") or "DALLE").upper()

        if model_type_str not in ("OPENAI", "DALLE"):
            raise ValueError("model_type must be either 'OPENAI' or 'DALLE'")

        model_type: ModelType = model_type_str  # type: ignore[assignment]

        if self.logger:
            self.logger.info(
                f"Generating image with model_type={model_type} for prompt: "
                f"'{prompt[:80]}{'...' if len(prompt) > 80 else ''}'"
            )

        image_bytes = self.image_service.generate_image(prompt, model_type)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        if self.logger:
            self.logger.info("Image generation completed successfully")

        return ServiceResult(
            prompt=prompt, model_type=model_type, image_base64=image_b64
        )


def get_agentic_service(logger=None):
    """
    Factory function to get the appropriate service for this branch.

    This branch implements an image-generation agent used by the Masumi flow.
    """
    return AgenticService(logger)
