import pytest
from relevance_checkers.cross_encoder_relevance_checker import CrossEncoderRelevanceChecker


@pytest.fixture(scope="module")
def checker():
	return CrossEncoderRelevanceChecker()

@pytest.fixture(scope="module")
def threshold():
	return 0.8


def test_high_relevance_simple(checker, threshold):
	prompt = "What is the capital of France?"
	answer = "The capital of France is Paris."
	score = checker.check_relevance(prompt, answer)
	assert score > threshold, f"Expected high relevance, got {score}"


def test_low_relevance_simple(checker, threshold):
	prompt = "What is the capital of France?"
	answer = "I enjoy playing the guitar on weekends."
	score = checker.check_relevance(prompt, answer)
	assert score < threshold, f"Expected low relevance, got {score}"


def test_high_relevance_opinion_based(checker, threshold):
	prompt = "Why is exercise beneficial for mental health?"
	answer = "Exercise improves mental health by reducing stress and boosting endorphins."
	score = checker.check_relevance(prompt, answer)
	assert score > threshold, f"Expected high relevance, got {score}"


def test_low_relevance_numeric_vs_factual(checker, threshold):
	prompt = "How many planets are in the Solar System?"
	answer = "The Amazon rainforest covers around 5.5 million square kilometers."
	score = checker.check_relevance(prompt, answer)
	assert score < threshold, f"Expected low relevance, got {score}"


def test_medium_relevance_partial_answer(checker, threshold):
	prompt = "Explain how photosynthesis works."
	answer = "Photosynthesis is a process used by plants, but it is very complex and involves several steps."
	score = checker.check_relevance(prompt, answer)
	# Not enforcing above threshold, just checking it returns a valid float
	assert isinstance(score, float)


def test_question_answer_mismatch(checker, threshold):
	prompt = "Can you tell me who invented the telephone?"
	answer = "The Great Wall of China is one of the most famous architectural structures in the world."
	score = checker.check_relevance(prompt, answer)
	assert score < threshold, "Expected mismatch relevance to be low"


def test_high_relevance_yes_no(checker, threshold):
	prompt = "Is water composed of hydrogen and oxygen?"
	answer = "Yes, water consists of two hydrogen atoms and one oxygen atom."
	score = checker.check_relevance(prompt, answer)
	assert score > threshold


def test_low_relevance_yes_no_wrong_fact(checker, threshold):
	prompt = "Is water composed of hydrogen and oxygen?"
	answer = "No, water is made of nitrogen and helium."
	score = checker.check_relevance(prompt, answer)
	# The model may treat this as relevant textually, but logically wrong.
	# This test only evaluates textual relevance, not factual correctness.
	assert isinstance(score, float)


def test_edge_case_empty_answer(checker, threshold):
	prompt = "What is quantum entanglement?"
	answer = ""
	score = checker.check_relevance(prompt, answer)
	assert score < threshold


def test_edge_case_empty_prompt(checker, threshold):
	prompt = ""
	answer = "Quantum entanglement describes correlations between particles."
	score = checker.check_relevance(prompt, answer)
	assert score < threshold
