"""
Mock data generators for different model types.
"""

import base64
import mimetypes
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .config import MockDataConfig, ModelConfig


def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """
    Load an image from local file and convert to base64.
    
    Args:
        image_path: Path to the local image file
        
    Returns:
        Tuple of (base64_data, mime_type)
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        # Default to jpeg if unknown
        mime_type = "image/jpeg"
    
    # Read and encode
    with open(path, "rb") as f:
        image_data = f.read()
    
    base64_data = base64.b64encode(image_data).decode("utf-8")
    return base64_data, mime_type


@dataclass
class MockRequest:
    """A mock request for benchmarking."""
    payload: Dict[str, Any]
    endpoint: str
    description: str = ""


class BaseMockGenerator:
    """Base class for mock data generators."""
    
    def __init__(self, model_config: ModelConfig, mock_data_config: MockDataConfig):
        self.model_config = model_config
        self.mock_data = mock_data_config
    
    def generate(self, count: int = 1) -> List[MockRequest]:
        """Generate mock requests."""
        raise NotImplementedError
    
    def get_endpoint(self) -> str:
        """Get the API endpoint for this model type."""
        raise NotImplementedError


class ChatMockGenerator(BaseMockGenerator):
    """Mock data generator for chat models."""
    
    def get_endpoint(self) -> str:
        return "/chat/completions"
    
    def generate(self, count: int = 1) -> List[MockRequest]:
        """Generate chat completion requests."""
        requests = []
        prompts = self.mock_data.chat_prompts
        
        for i in range(count):
            prompt = prompts[i % len(prompts)]
            payload = {
                "model": self.model_config.name,
                "max_tokens": self.model_config.max_tokens,
                "temperature": self.model_config.temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            requests.append(MockRequest(
                payload=payload,
                endpoint=self.get_endpoint(),
                description=f"Chat completion: {prompt[:50]}..."
            ))
        
        return requests


class EmbedMockGenerator(BaseMockGenerator):
    """Mock data generator for embedding models."""
    
    def get_endpoint(self) -> str:
        return "/embeddings"
    
    def generate(self, count: int = 1) -> List[MockRequest]:
        """Generate embedding requests."""
        requests = []
        texts = self.mock_data.embed_texts
        
        for i in range(count):
            text = texts[i % len(texts)]
            payload = {
                "model": self.model_config.name,
                "input": text,
                "encoding_format": "float"
            }
            requests.append(MockRequest(
                payload=payload,
                endpoint=self.get_endpoint(),
                description=f"Embedding: {text[:50]}..."
            ))
        
        return requests


class RerankerMockGenerator(BaseMockGenerator):
    """Mock data generator for reranker models."""
    
    def get_endpoint(self) -> str:
        return "/rerank"
    
    def generate(self, count: int = 1) -> List[MockRequest]:
        """Generate reranker requests."""
        requests = []
        query = self.mock_data.reranker_query
        documents = self.mock_data.reranker_documents
        
        for _ in range(count):
            # Shuffle documents for variety
            shuffled_docs = documents.copy()
            random.shuffle(shuffled_docs)
            
            payload = {
                "model": self.model_config.name,
                "query": query,
                "documents": shuffled_docs,
                "top_n": min(3, len(shuffled_docs))
            }
            requests.append(MockRequest(
                payload=payload,
                endpoint=self.get_endpoint(),
                description=f"Rerank: {query[:50]}..."
            ))
        
        return requests


class VisionMockGenerator(BaseMockGenerator):
    """Mock data generator for vision models."""
    
    def __init__(self, model_config: ModelConfig, mock_data_config: MockDataConfig):
        super().__init__(model_config, mock_data_config)
        self._cached_images: Dict[str, tuple[str, str]] = {}  # path -> (base64, mime_type)
    
    def get_endpoint(self) -> str:
        return "/chat/completions"
    
    def _get_image_content(self, image_index: int = 0) -> Dict[str, Any]:
        """
        Get image content for the request.
        Supports: base64 string, local file paths, URLs, and multiple images.
        """
        # Check for multiple local images
        if self.mock_data.vision_image_paths:
            paths = self.mock_data.vision_image_paths
            image_path = paths[image_index % len(paths)]
            
            # Cache the base64 encoding
            if image_path not in self._cached_images:
                self._cached_images[image_path] = load_image_as_base64(image_path)
            
            base64_data, mime_type = self._cached_images[image_path]
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_data}"
                }
            }
        
        # Check for single local image path
        if self.mock_data.vision_image_path:
            image_path = self.mock_data.vision_image_path
            
            # Cache the base64 encoding
            if image_path not in self._cached_images:
                self._cached_images[image_path] = load_image_as_base64(image_path)
            
            base64_data, mime_type = self._cached_images[image_path]
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_data}"
                }
            }
        
        # Check for pre-encoded base64
        if self.mock_data.vision_image_base64:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.mock_data.vision_image_base64}"
                }
            }
        
        # Default to URL
        return {
            "type": "image_url",
            "image_url": {
                "url": self.mock_data.vision_image_url
            }
        }
    
    def generate(self, count: int = 1) -> List[MockRequest]:
        """Generate vision model requests."""
        requests = []
        prompts = self.mock_data.vision_prompts
        
        for i in range(count):
            prompt = prompts[i % len(prompts)]
            
            # Get image content (cycles through images if multiple)
            image_content = self._get_image_content(i)
            
            payload = {
                "model": self.model_config.name,
                "max_tokens": self.model_config.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            image_content
                        ]
                    }
                ],
                "stream": False
            }
            requests.append(MockRequest(
                payload=payload,
                endpoint=self.get_endpoint(),
                description=f"Vision: {prompt[:50]}..."
            ))
        
        return requests


def get_mock_generator(model_config: ModelConfig, mock_data_config: MockDataConfig) -> BaseMockGenerator:
    """Factory function to get the appropriate mock generator."""
    generators = {
        "chat": ChatMockGenerator,
        "embed": EmbedMockGenerator,
        "embedding": EmbedMockGenerator,
        "reranker": RerankerMockGenerator,
        "rerank": RerankerMockGenerator,
        "vision": VisionMockGenerator
    }
    
    model_type = model_config.type.lower()
    generator_class = generators.get(model_type)
    
    if generator_class is None:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: {list(generators.keys())}")
    
    return generator_class(model_config, mock_data_config)
