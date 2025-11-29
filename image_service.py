import base64
from typing import Literal

from openai import OpenAI


ModelType = Literal["OPENAI", "DALLE"]


class ImageGenerationService:
    """
    Simple image generation wrapper for OpenAI image models.

    - model_type="OPENAI": uses the Responses API with the image_generation tool
    - model_type="DALLE": uses the Images API (e.g. dall-e-3)
    """

    def __init__(self):
        """
        Initialize the OpenAI client.

        Relies on standard OpenAI / LangChain-style environment configuration:
        - OPENAI_API_KEY (and other OpenAI_* vars) are read from the environment.
        """
        self.client = OpenAI()

    def generate_image(self, prompt: str, model_type: ModelType = "DALLE") -> bytes:
        """
        Generate an image and return raw PNG bytes.

        Args:
            prompt: Text description of the image to generate.
            model_type: "OPENAI" or "DALLE".
        """
        print(f"[IMAGE GENERATION] Starting image generation...")
        print(f"   Model: {model_type}")
        print(f"   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        if model_type == "OPENAI":
            # Uses the Responses API with the image_generation tool
            # Note: This API may require specific model versions
            try:
                response = self.client.responses.create(
                    model="gpt-4o-mini",  # Updated to standard model name
                    input=prompt,
                    tools=[{"type": "image_generation"}],
                )
            except Exception as e:
                raise RuntimeError(
                    f"OpenAI Responses API error: {type(e).__name__}: {str(e)}. "
                    f"Note: The responses.create API may not be available for all models or API keys."
                ) from e

            image_data = [
                output.result
                for output in response.output  # type: ignore[attr-defined]
                if getattr(output, "type", None) == "image_generation_call"
            ]

            if not image_data:
                raise RuntimeError("No image data returned from OpenAI Responses API")

            image_base64 = image_data[0]

        elif model_type == "DALLE":
            # Uses the Images API (e.g. dall-e-3)
            result = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                response_format="b64_json",
            )

            image_base64 = result.data[0].b64_json  # type: ignore[assignment]

        else:
            raise ValueError("model_type must be either 'OPENAI' or 'DALLE'")

        image_bytes = base64.b64decode(image_base64)
        print(
            f"[IMAGE GENERATION] Image generated successfully ({len(image_bytes)} bytes)"
        )
        return image_bytes
