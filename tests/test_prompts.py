from domain_llm.prompts import normalize_prediction, prompt, training_text


def test_prompt_contains_input_and_labels():
    assert training_text("hello", "card_arrival", ["card_arrival"]).endswith("card_arrival")

def test_normalize_prediction_is_safe():
    labels = ["card_arrival", "pending_transfer"]
    assert normalize_prediction("The answer is pending_transfer.", labels) == "pending_transfer"
    assert normalize_prediction("nonsense", labels) == "__unknown__"
