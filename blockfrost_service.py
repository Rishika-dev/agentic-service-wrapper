"""
Blockfrost IPFS Service for uploading images to IPFS.

Uses Blockfrost IPFS API to upload and pin images via direct HTTP calls.
Documentation: https://docs.blockfrost.io/#tag/ipfs--add
"""

import os
from typing import Optional
import requests


class BlockfrostService:
    """
    Service for uploading files to IPFS using Blockfrost REST API.

    Requires BLOCKFROST_PROJECT_ID environment variable.
    """

    def __init__(self, project_id: Optional[str] = None, logger=None):
        """
        Initialize Blockfrost IPFS service.

        Args:
            project_id: Blockfrost project ID. If not provided, reads from
                       BLOCKFROST_PROJECT_ID environment variable.
            logger: Optional logger instance for logging operations.
        """
        self.logger = logger
        self.project_id = project_id or os.getenv("BLOCKFROST_PROJECT_ID")
        if not self.project_id:
            raise ValueError(
                "BLOCKFROST_PROJECT_ID environment variable is not set. "
                "Get your project ID from https://blockfrost.io/"
            )

        self.base_url = "https://ipfs.blockfrost.io/api/v0"
        self.headers = {"project_id": self.project_id}

        if self.logger:
            self.logger.info("Blockfrost IPFS service initialized")

    def upload_file(self, file_path: str) -> Optional[str]:
        """
        Upload a file to IPFS using Blockfrost REST API.

        Args:
            file_path: Path to the file to upload.

        Returns:
            IPFS hash (CID) if successful, None otherwise.
        """
        if not os.path.exists(file_path):
            if self.logger:
                self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if self.logger:
            file_size = os.path.getsize(file_path)
            self.logger.info(
                f"Uploading file to IPFS: {file_path} (size: {file_size} bytes)"
            )

        try:
            url = f"{self.base_url}/ipfs/add"

            with open(file_path, "rb") as f:
                files = {
                    "file": (os.path.basename(file_path), f, "application/octet-stream")
                }
                if self.logger:
                    self.logger.debug(
                        f"POST {url} with file: {os.path.basename(file_path)}"
                    )

                response = requests.post(
                    url, headers=self.headers, files=files, timeout=60
                )
                response.raise_for_status()

                result = response.json()
                ipfs_hash = result.get("ipfs_hash")

                if not ipfs_hash:
                    if self.logger:
                        self.logger.error(f"No IPFS hash in response: {result}")
                    raise RuntimeError(f"No IPFS hash in response: {result}")

                if self.logger:
                    self.logger.info(f"Successfully uploaded to IPFS: {ipfs_hash}")

                return ipfs_hash
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if e.response else str(e)
            if self.logger:
                self.logger.error(
                    f"Blockfrost IPFS upload HTTP error: {e} - {error_detail}"
                )
            raise RuntimeError(
                f"Blockfrost IPFS upload HTTP error: {e} - {error_detail}"
            ) from e
        except requests.exceptions.RequestException as e:
            if self.logger:
                self.logger.error(f"Blockfrost IPFS upload request error: {e}")
            raise RuntimeError(f"Blockfrost IPFS upload request error: {e}") from e
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Unexpected error uploading to IPFS: {e}", exc_info=True
                )
            raise RuntimeError(f"Unexpected error uploading to IPFS: {e}") from e

    def pin_file(self, ipfs_hash: str) -> bool:
        """
        Pin a file on IPFS to prevent garbage collection.

        Args:
            ipfs_hash: IPFS hash (CID) of the file to pin.

        Returns:
            True if successful, False otherwise.
        """
        if self.logger:
            self.logger.info(f"Pinning IPFS hash: {ipfs_hash}")

        try:
            url = f"{self.base_url}/ipfs/pin/add/{ipfs_hash}"
            if self.logger:
                self.logger.debug(f"POST {url}")

            response = requests.post(url, headers=self.headers, timeout=30)

            # 200 or 201 means success
            if response.status_code in (200, 201):
                if self.logger:
                    self.logger.info(f"Successfully pinned IPFS hash: {ipfs_hash}")
                return True

            # File might already be pinned (409 Conflict)
            if response.status_code == 409:
                if self.logger:
                    self.logger.info(f"IPFS hash already pinned: {ipfs_hash}")
                return True

            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as e:
            # File might already be pinned
            if e.response and e.response.status_code == 409:
                if self.logger:
                    self.logger.info(f"IPFS hash already pinned (409): {ipfs_hash}")
                return True
            error_detail = e.response.text if e.response else str(e)
            if self.logger:
                self.logger.error(
                    f"Blockfrost IPFS pin HTTP error: {e} - {error_detail}"
                )
            raise RuntimeError(
                f"Blockfrost IPFS pin HTTP error: {e} - {error_detail}"
            ) from e
        except requests.exceptions.RequestException as e:
            if self.logger:
                self.logger.error(f"Blockfrost IPFS pin request error: {e}")
            raise RuntimeError(f"Blockfrost IPFS pin request error: {e}") from e
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Unexpected error pinning on IPFS: {e}", exc_info=True
                )
            raise RuntimeError(f"Unexpected error pinning on IPFS: {e}") from e

    def upload_and_pin(self, file_path: str) -> Optional[str]:
        """
        Upload a file to IPFS and pin it.

        Args:
            file_path: Path to the file to upload.

        Returns:
            IPFS hash (CID) if successful, None otherwise.
        """
        if self.logger:
            self.logger.info(f"Starting upload and pin operation for: {file_path}")

        ipfs_hash = self.upload_file(file_path)
        if ipfs_hash:
            self.pin_file(ipfs_hash)
            if self.logger:
                self.logger.info(f"Completed upload and pin: {ipfs_hash}")
        else:
            if self.logger:
                self.logger.warning(f"Upload completed but no IPFS hash returned")

        return ipfs_hash
