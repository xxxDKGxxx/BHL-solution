from interface.abstractmodel import AbstractModel
import google.generativeai as genai

class GeminiLLM(AbstractModel):
	def __init__(self):
		self.chat = None

	def generate_answer(self, prompt: str) -> str:
		if self.chat is None:
			api_key = "AIzaSyBWtYa-0DdN7rs9E_ovdgsbATjJK3NQXmI"
			genai.configure(api_key=api_key)
			model = genai.GenerativeModel('gemini-flash-latest')
			self.chat = model.start_chat()

		response = self.chat.send_message(prompt)
		return response.text