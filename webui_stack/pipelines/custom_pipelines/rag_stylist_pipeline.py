# # pipelines/pipelines/rag_stylist_pipeline.py
from pydantic import BaseModel, Field
from typing import List, Union, Generator, Iterator
import ollama
import os

class Valves(BaseModel):
    RAG_SPECIALIST_MODEL: str = Field(default="llama3.2:latest", description="Model for Agent 1 (Stylistic Context Retriever).")
    STYLE_MAESTRO_MODEL: str = Field(default="gpt-oss:latest", description="Model for Agent 2 (Style Translator).")

class Pipeline:
    def __init__(self):
        self.valves = Valves()

        # ===================================================================
        # SYSTEM PROMPT FOR AGENT 1: STYLISTIC CONTEXT RETRIEVER
        # ===================================================================
        self.STYLISTIC_CONTEXT_RETRIEVER_PROMPT = """# **MISSION**
        You are a "Stylistic Context Retriever." Your only job is to receive a piece of text from the user and find relevant examples of writing from the attached knowledge base (`#wildfire_study`). These examples will be used by another AI to learn a specific writing style.

        # **OPERATIONAL MANDATE**
        1.  **Analyze Input Text:** Read the user's text and identify its core topics, keywords, and general sentiment.
        2.  **Formulate Search Queries:** Use the identified topics and keywords to search the `wildfire_study` knowledge base. Your goal is NOT to answer a question, but to find passages that are thematically similar to the user's text.
        3.  **Retrieve Stylistic Examples:** Select the top 10-15 most relevant passages that best represent the author's writing style on those topics.
        4.  **Format Output:** Present the retrieved passages cleanly under the specified heading.

        # **OUTPUT PROTOCOL**
        -   Your entire output must consist of ONLY the pristine, concatenated text from the final selected chunks.
        -   The output must be formatted under a single, machine-readable heading: `### STYLE EXAMPLES ###`.
        -   **DO NOT** add any conversational text, introductions, summaries, or explanations.
        -   **DO NOT** attempt to rewrite or modify the user's text yourself.
        """

        # ===================================================================
        # SYSTEM PROMPT FOR AGENT 2: STYLE TRANSLATOR 
        # ===================================================================
        self.STYLE_TRANSLATOR_PROMPT = """# **MISSION**
        You are an expert "Style Translator." Your sole function is to rewrite a given piece of text to perfectly match the writing style, tone, and vocabulary found in a set of provided examples from a target author.

        # **INCOMING DATA**
        You will receive two pieces of information:
        1.  **ORIGINAL TEXT:** The text that needs to be rewritten.
        2.  **STYLE EXAMPLES:** Passages written by the target author that you must emulate.

        # **CORE DIRECTIVE**
        -   You must rewrite the "ORIGINAL TEXT" completely.
        -   The rewritten text's core meaning, facts, and intent **must be preserved exactly.**
        -   The vocabulary, sentence structure, punctuation, and overall tone **must be changed** to match the "STYLE EXAMPLES."
        -   **DO NOT** add new information or ideas from the style examples.
        -   **DO NOT** treat the original text as a question to be answered.
        -   Your output must be **ONLY** the final, rewritten text. No introductions, explanations, or apologies.
        """
        
        ollama_host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = ollama.Client(host=ollama_host)
        
        print(f"Pipeline initialized. Connecting to Ollama at: {ollama_host}")

    async def on_startup(self):
        print(f"## RAG Stylist Pipeline Started ##")
        print(f"Agent 1 Model: {self.valves.RAG_SPECIALIST_MODEL}")
        print(f"Agent 2 Model: {self.valves.STYLE_MAESTRO_MODEL}")
        pass

    async def on_shutdown(self):
        print(f"## RAG Stylist Pipeline Shutting Down ##")
        pass

    def pipe(
        self, user_message: str, **kwargs,
    ) -> Union[str, Generator, Iterator]:
        print(f"Received text to be rewritten: '{user_message}'")

        print(f"Invoking Agent 1 ({self.valves.RAG_SPECIALIST_MODEL}) to find style examples...")
        
        agent1_message = f"#wildfire_study {user_message}"
        
        agent1_messages = [{"role": "system", "content": self.STYLISTIC_CONTEXT_RETRIEVER_PROMPT}, {"role": "user", "content": agent1_message}]
        try:
            rag_response = self.client.chat(model=self.valves.RAG_SPECIALIST_MODEL, messages=agent1_messages, stream=False)
            retrieved_style_examples = rag_response['message']['content']
            print(f"Successfully retrieved style examples from Agent 1.")
        except Exception as e:
            print(f"ERROR calling Agent 1 (Stylistic Context Retriever): {e}")
            return f"An error occurred while retrieving style examples: {e}"

        print(f"Invoking Agent 2 ({self.valves.STYLE_MAESTRO_MODEL}) to perform style transfer...")
        
        final_prompt_for_translator = f"""**ORIGINAL TEXT TO REWRITE:**
        '''
        {user_message}
        '''

        **STYLE EXAMPLES FROM TARGET AUTHOR:**
        '''
        {retrieved_style_examples}
        '''

        ---
        **TASK:** Rewrite the "ORIGINAL TEXT" to match the style of the "STYLE EXAMPLES". Preserve the original meaning exactly."""

        agent2_messages = [{"role": "system", "content": self.STYLE_TRANSLATOR_PROMPT}, {"role": "user", "content": final_prompt_for_translator}]
        try:
            stream = self.client.chat(model=self.valves.STYLE_MAESTRO_MODEL, messages=agent2_messages, stream=True)
            return (chunk['message']['content'] for chunk in stream)
        except Exception as e:
            print(f"ERROR calling Agent 2 (Style Translator): {e}")
            return f"An error occurred during style translation: {e}"