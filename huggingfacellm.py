import traceback

import requests


class HuggingFaceLLM:
    """Custom LLM class to replace ChatOpenAI with Hugging Face API"""

    def __init__(self, model_name="google/flan-t5-base", hf_token=None):
        self.hf_token = hf_token
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.model_name = model_name
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

        # Alternative free models you can try:
        # "microsoft/DialoGPT-medium" - Conversational
        # "google/flan-t5-base" - Q&A focused
        # "bigscience/bloom-560m" - General purpose
        # "microsoft/DialoGPT-small" - Faster, smaller
        # Models that work without tokens (public):

        self.free_models = [
            "google/flan-t5-base",  # Best for Q&A - works without token
            "google/flan-t5-small",  # Faster version
            "bigscience/bloom-560m",  # General purpose
            "facebook/bart-large-cnn",  # Good for summarization
        ]

    def invoke(self, prompt):
        """Method to match LangChain's ChatOpenAI interface"""

        # Special handling for different model types
        if "flan-t5" in self.model_name.lower():
            # FLAN-T5 works better with direct questions
            payload = {"inputs": prompt}
        else:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 200,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False
                }
            }

        try:
            # Try without token first
            response = requests.post(self.api_url, json=payload, timeout=30)

            # If 403 with token, try without token
            if response.status_code == 403 and self.headers:
                print("🔄 Token failed, trying without authentication...")
                response = requests.post(self.api_url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()

                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        generated_text = result[0].get('generated_text', result[0].get('summary_text', ''))
                    else:
                        generated_text = str(result[0])
                elif isinstance(result, dict):
                    generated_text = result.get('generated_text', result.get('summary_text', str(result)))
                else:
                    generated_text = str(result)

                # Clean up the response
                if generated_text:
                    # Remove the original prompt if it's repeated
                    if prompt in generated_text:
                        generated_text = generated_text.replace(prompt, "").strip()

                    return type('Response', (), {'content': generated_text})()
                else:
                    return type('Response', (), {
                        'content': "I understand your question, but I need more context to provide a good answer."})()

            elif response.status_code == 503:
                return type('Response', (), {'content': "🔄 Model is loading, please try again in a moment."})()
            elif response.status_code == 429:
                return type('Response', (), {'content': "⏰ Too many requests, please wait a moment and try again."})()
            else:
                # Try alternative model
                return self._try_alternative_model(prompt)

        except Exception as e:
            return type('Response', (), {'content': f"Let me try a different approach... Error: {str(e)}"})()

    def _try_alternative_model(self, prompt):
        """Try alternative free models if main one fails"""
        print(1)
        for model in self.free_models:
            print("model",model, "self.model_name", self.model_name)
            if model != self.model_name or 'flan-t5-base' in self.model_name:   # Don't try the same model
                print(3)
                try:
                    alt_url = f"https://api-inference.huggingface.co/models/{model}"
                    payload = {"inputs": prompt}
                    response = requests.post(alt_url, json=payload, timeout=15)
                    print(response)

                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            text = result[0].get('generated_text', result[0].get('summary_text', ''))
                            if text and text != prompt:
                                return type('Response', (), {'content': f"[Using {model}] {text}"})()

                except Exception as e:
                    print(traceback.format_exc())
                    continue

        # If all models fail, return a helpful message
        return type('Response', (), {
            'content': "I'm having trouble accessing the AI models right now. Please try again later or check your internet connection."})()

