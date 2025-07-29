from transformers import pipeline, AutoTokenizer

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

summarizer = pipeline("summarization", model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def summarize_text(text, max_input_tokens=1024):
    """
    Tronque le texte à max_input_tokens avant le résumé. 
    Coupe aussi en cas de texte trop long pour éviter les erreurs internes du modèle.
    """
    if not text or not isinstance(text, str):
        return "[Texte invalide ou vide]"

    # Tokenisation et tronquage si trop long
    tokens = tokenizer.encode(text, truncation=True, max_length=max_input_tokens)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)

    try:
        summary = summarizer(
            truncated_text,
            max_length=180,
            min_length=50,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"[Erreur pendant le résumé] : {str(e)}"

# Tu peux remplacer le modèle par facebook/bart-large-cnn ou t5-small si besoin.
