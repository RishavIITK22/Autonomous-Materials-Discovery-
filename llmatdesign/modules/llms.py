import os
import openai
from openai import OpenAI
import google.generativeai as genai
import groq
import requests
import time
# from krutrim_cloud import KrutrimCloud
# from dotenv import load_dotenv
# load_dotenv()
gemini_safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

class AskLLM:
    def __init__(
        self, 
        llm_model, 
        api_key=None,
        base_url=None,
        openai_organization=None, 
    ) -> None:
        self.llm_model = llm_model
        self.api_key = api_key
        if self.llm_model.startswith('gpt'):
            api_key = api_key if api_key is not None else os.environ.get('OPENAI_API_KEY')
            if openai_organization is not None:
                self.client = openai.OpenAI(
                    organization=openai_organization,
                    api_key=api_key
                )
            else:
                self.client = openai.OpenAI(api_key=api_key)
            self.model_name = self.get_openai_model_name()
            
        elif self.llm_model.startswith('gemini'):

            api_key = api_key if api_key is not None else os.environ.get('GOOGLE_API_KEY')

            if api_key is None:
                raise ValueError('Please provide a valid Google API key.')
            
            genai.configure(api_key=api_key)
            self.model_name = self.get_gemini_model_name()

        elif self.llm_model.startswith('groq'):
            api_key = api_key if api_key is not None else os.environ.get('GROQ_API_KEY')
            if api_key is None:
                raise ValueError('Please provide a valid Groq API key.')
            self.url = "https://api.groq.com/openai/v1/chat/completions"
            self.client = groq.Groq(api_key=api_key)
            self.model_name = self.get_groq_model_name()
        #For Krutrim/LLama-3.3-70b model
        elif self.llm_model.startswith('krutrim') or self.llm_model.startswith('Llama'):
            api_key = api_key if api_key is not None else os.environ.get('KRUTRIM_API_KEY')
            if api_key is None:
                raise ValueError('Please provide a valid Krutrim API key.')
            
            base_url = base_url if base_url is not None else "https://cloud.olakrutrim.com/v1"
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.model_name = self.get_krutrim_model_name()
        else:
            raise ValueError('Supported models are GPT, Gemini, and Llama-based models via KrutrimCloud.')

    def get_openai_model_name(self):
        if self.llm_model == 'gpt-4':
            return "gpt-4-0125-preview"
        elif self.llm_model == 'gpt-4o':
            return "gpt-4o"
        elif self.llm_model == 'gpt-3.5':
            return "gpt-3.5-turbo-0125"
        else:
            raise ValueError('Supported OpenAI models are GPT-3.5 and GPT-4.')
    
    def get_gemini_model_name(self):
        if self.llm_model == 'gemini-2.0-flash':
            return "gemini-2.0-flash"
        else:
            raise ValueError('Supported Gemini models: Gemini-2.0-flash.')
    def get_groq_model_name(self):
        if self.llm_model == 'Llama-3.3-70b-versatile':
            return "llama-3.3-70b-versatile"
        else:
            raise ValueError('Supported Groq models: llama3-70b.')
        
    def get_krutrim_model_name(self):
        # Return the model name as is for Krutrim
        if self.llm_model == 'krutrim-base':
            return "krutrim-base"
        elif self.llm_model == 'krutrim-pro':
            return "krutrim-pro"
        elif self.llm_model == 'Llama-3.3-70B-Instruct':
            return "Llama-3.3-70B-Instruct"
        else:
            raise ValueError('Supported Krutrim models: krutrim-base, krutrim-pro, Llama-3.3-70B-Instruct')

    def ask(self, prompt):
        if self.llm_model.startswith('gpt'):
            return self.ask_openai_compatible(prompt)
        elif self.llm_model.startswith('gemini'):
            result=self.ask_google(prompt)
        # elif self.llm_model.startswith('groq'):
        #     return self.ask_groq(prompt)
        elif self.llm_model.startswith('Llama'):
            result = self.ask_krutrim(prompt)
        else:
            raise ValueError('Supported models are GPT and Gemini LLM models.')
        #time.sleep(5)  # To avoid rate limiting
        return result
    
    def ask_openai_compatible(self, prompt):
        prompt_chat = [{"role": "user", "content": prompt.strip()}]
        
        # For Krutrim models, add a system message
        if self.llm_model.startswith('krutrim') or self.llm_model == 'Llama-3.3-70B-Instruct':
            prompt_chat.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt_chat,
            temperature=0,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            # Additional parameters specific to Krutrim if needed
            max_tokens=256 if (self.llm_model.startswith('krutrim') or self.llm_model == 'Llama-3.3-70B-Instruct') else None,
        )
        return response.choices[0].message.content.strip()
        
    def ask_google(self, prompt):
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt, safety_settings=gemini_safety_settings)
        return response.text.strip()
    def ask_groq(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": "You are an AI assistant."},
                         {"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(self.url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print("Error:", response.text)
            return None
        
    def ask_krutrim(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            # Extract the generated output
            txt_output_data = response.choices[0].message.content
            return txt_output_data.strip()
        except Exception as exc:
            raise RuntimeError(f"Krutrim API error: {exc}")
            return None