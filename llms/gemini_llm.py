from interface.abstractmodel import AbstractModel


class GeminiLLM(AbstractModel):
	def generate_answer(self, prompt: str, genai=None) -> str:
		api_key = "GEMINI_API_KEY"

		genai.configure(api_key)

		model = genai.GenerativeModel('gemini-flash-latest')

		response = model.generate_content(prompt)

		return response.text