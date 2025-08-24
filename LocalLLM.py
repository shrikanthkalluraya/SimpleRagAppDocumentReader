from transformers import pipeline
import torch


class LocalLLM:
    """Local LLM that runs on your computer - NO API needed!"""

    def __init__(self, model_name="google/flan-t5-small"):
        print(f"🤖 Loading local model: {model_name}")
        print("📦 This might take a moment the first time...")

        try:
            # Use transformers pipeline for easy local inference
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,  # GPU if available, else CPU
                max_length=200,
                do_sample=True,
                temperature=0.7
            )
            print("✅ Local model loaded successfully!")

        except Exception as e:
            print(f"⚠️ Error loading {model_name}, trying backup model...")
            # Fallback to even smaller model
            self.generator = pipeline(
                "text2text-generation",
                model="t5-small",
                device=-1,  # Force CPU for compatibility
                max_length=150
            )
            print("✅ Backup model loaded!")

    def invoke(self, prompt):
        """Generate response locally"""
        try:
            # Generate response
            result = self.generator(prompt, max_length=200, num_return_sequences=1)

            if result and len(result) > 0:
                generated_text = result[0]['generated_text']
                return type('Response', (), {'content': generated_text})()
            else:
                return type('Response', (), {'content': "I'm not sure how to answer that question."})()

        except Exception as e:
            return type('Response', (), {'content': f"Error generating response: {str(e)}"})()


class SimpleLocalLLM:
    """Ultra-simple fallback that works without any special models"""

    def __init__(self):
        print("🤖 Using simple pattern-based responses (no model needed)")

    def invoke(self, prompt):
        """Simple pattern matching for basic Q&A"""
        prompt_lower = prompt.lower()

        # Extract question from prompt
        if "question:" in prompt_lower:
            question = prompt_lower.split("question:")[-1].strip()
        else:
            question = prompt_lower

        # Simple pattern matching
        if any(word in question for word in ["who", "character", "person"]):
            if "context" in prompt_lower:
                context = prompt.split("Context")[1].split("Question")[0] if "Context" in prompt else ""
                # Extract names from context (very basic)
                words = context.split()
                names = [word for word in words if word.istitle() and len(word) > 2]
                if names:
                    return type('Response', (),
                                {'content': f"Based on the text, the main character appears to be {names[0]}."})()

        elif any(word in question for word in ["what", "describe", "about"]):
            return type('Response', (), {
                'content': "Based on the book content, I can see this relates to the story and characters described in the text."})()

        elif any(word in question for word in ["where", "place", "location"]):
            return type('Response', (), {'content': "The events take place in the setting described in the book."})()

        elif any(word in question for word in ["why", "because", "reason"]):
            return type('Response', (), {'content': "The reasons are explained in the context of the story."})()

        elif any(word in question for word in ["how", "way", "method"]):
            return type('Response', (), {'content': "The method or process is described in the book content."})()

        # Default response
        return type('Response', (), {
            'content': "I found relevant information in the book. The answer depends on the specific context provided."})()
